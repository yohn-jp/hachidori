// Package releasestub is the deterministic local stand-in for the one release
// authority the updater talks to (the official GitHub Releases of
// yohn-jp/hachidori). It exists so the update E2E never reaches a real
// network: Server.Transport is handed to update.Service as its Transport, and
// that transport dials only the local listener whatever URL it is given. The
// fixed authority URLs stay exactly as production builds them
// (api.github.com, github.com, objects.githubusercontent.com over https); only
// the connection is local.
//
// Everything the E2E must prove about failure is injected here and is
// deterministic: a corrupted body, a checksum that does not match, a transfer
// that is cut after N bytes, a transfer or list response that is held until the
// test releases or aborts it, an unavailable release list. Holding is
// synchronized on observable state (Gate.Reached), never on a sleep.
package releasestub

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/update"
)

// Hosts of the fixed release authority, as the updater requests them.
const (
	APIHost      = "api.github.com"
	DownloadHost = "github.com"
	AssetHost    = "objects.githubusercontent.com"
)

// published is the fixed publication time of every stub release.
var published = time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)

// Request is one request the server received, as the updater sent it.
type Request struct {
	Method string
	Host   string
	Path   string
	Query  string
}

func (r Request) String() string {
	if r.Query != "" {
		return r.Host + r.Path + "?" + r.Query
	}
	return r.Host + r.Path
}

// Release is one stub release. Exe is what the release lists and what its
// checksum binds; Served, when set, is what the asset host actually sends.
type Release struct {
	Tag        string
	Prerelease bool
	Exe        []byte
	// Sum is the body of the checksum asset.
	Sum []byte
	// Served overrides the executable body the asset host sends. Its length
	// may differ from the listed size.
	Served []byte
	// MetadataDigest makes the release list report the executable's digest the
	// way GitHub does.
	MetadataDigest bool
}

func (r *Release) body() []byte {
	if r.Served != nil {
		return r.Served
	}
	return r.Exe
}

// Gate pauses one response at a chosen point until the test decides. Reached
// is closed when the response is paused, so a test synchronizes on it.
type Gate struct {
	reached chan struct{}
	release chan struct{}
	abort   chan struct{}
	once    [3]sync.Once
}

func newGate() *Gate {
	return &Gate{reached: make(chan struct{}), release: make(chan struct{}), abort: make(chan struct{})}
}

// Reached is closed once the held response has been paused.
func (g *Gate) Reached() <-chan struct{} { return g.reached }

// Release lets the held response continue and complete.
func (g *Gate) Release() { g.once[1].Do(func() { close(g.release) }) }

// Abort drops the connection of the held response without completing it.
func (g *Gate) Abort() { g.once[2].Do(func() { close(g.abort) }) }

func (g *Gate) markReached() { g.once[0].Do(func() { close(g.reached) }) }

// Server is the local release authority.
type Server struct {
	srv *httptest.Server

	mu       sync.Mutex
	releases []*Release
	reqs     []Request

	listGate   *Gate
	listStatus int
	listDrop   bool
	exeGate    map[string]*exeGate // tag -> hold
	exeCut     map[string]int      // tag -> bytes sent before the connection is dropped
	gates      []*Gate
}

type exeGate struct {
	after int
	gate  *Gate
}

// New starts the server; it is closed when the test ends.
func New(t testing.TB) *Server {
	t.Helper()
	s := &Server{exeGate: map[string]*exeGate{}, exeCut: map[string]int{}}
	s.srv = httptest.NewUnstartedServer(http.HandlerFunc(s.serve))
	s.srv.Config.ErrorLog = nil
	s.srv.StartTLS()
	t.Cleanup(func() {
		s.releaseAll()
		s.srv.Close()
	})
	return s
}

// Transport is the http.RoundTripper to give update.Service. It trusts only
// this server's certificate and dials only this server's listener, so no
// request can leave the machine whatever URL the updater builds.
func (s *Server) Transport() http.RoundTripper {
	pool := x509.NewCertPool()
	pool.AddCert(s.srv.Certificate())
	addr := s.srv.Listener.Addr().String()
	return &http.Transport{
		Proxy: nil,
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, "tcp", addr)
		},
		// httptest's certificate is valid for example.com and loopback.
		TLSClientConfig:   &tls.Config{RootCAs: pool, ServerName: "example.com", MinVersion: tls.VersionTLS12},
		DisableKeepAlives: true,
	}
}

// Publish adds a release whose checksum asset correctly binds exe. Releases
// are listed in publication order, newest last; the updater sorts them.
func (s *Server) Publish(tag string, exe []byte, mutate ...func(*Release)) *Release {
	r := &Release{Tag: tag, Exe: exe, Sum: ChecksumFile(exe)}
	for _, m := range mutate {
		m(r)
	}
	s.mu.Lock()
	s.releases = append(s.releases, r)
	s.mu.Unlock()
	return r
}

// ChecksumFile is the release workflow's checksum asset for exe: one line,
// "<sha256>  <asset name>".
func ChecksumFile(exe []byte) []byte {
	sum := sha256.Sum256(exe)
	return []byte(hex.EncodeToString(sum[:]) + "  " + update.ExeAsset + "\n")
}

// Requests returns every request received so far, in arrival order.
func (s *Server) Requests() []Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Request(nil), s.reqs...)
}

// Count is the number of requests received so far.
func (s *Server) Count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.reqs)
}

// Strings renders the request log for evidence and failure messages.
func (s *Server) Strings() []string {
	var out []string
	for _, r := range s.Requests() {
		out = append(out, r.Method+" "+r.String())
	}
	return out
}

// HoldList pauses the next release-list response before it is sent.
func (s *Server) HoldList() *Gate {
	g := newGate()
	s.mu.Lock()
	s.listGate = g
	s.gates = append(s.gates, g)
	s.mu.Unlock()
	return g
}

// HoldExe pauses the next download of tag's executable after after bytes of
// the body were sent.
func (s *Server) HoldExe(tag string, after int) *Gate {
	g := newGate()
	s.mu.Lock()
	s.exeGate[tag] = &exeGate{after: after, gate: g}
	s.gates = append(s.gates, g)
	s.mu.Unlock()
	return g
}

// CutExe makes downloads of tag's executable drop the connection after n bytes
// of the announced body (a network failure mid-transfer), until Heal.
func (s *Server) CutExe(tag string, n int) {
	s.mu.Lock()
	s.exeCut[tag] = n
	s.mu.Unlock()
}

// FailList makes the release list answer status (when nonzero) until Heal.
func (s *Server) FailList(status int) {
	s.mu.Lock()
	s.listStatus = status
	s.mu.Unlock()
}

// DropList makes the release list drop the connection without an answer until
// Heal.
func (s *Server) DropList() {
	s.mu.Lock()
	s.listDrop = true
	s.mu.Unlock()
}

// Heal removes every injected fault.
func (s *Server) Heal() {
	s.mu.Lock()
	s.listStatus, s.listDrop = 0, false
	s.exeCut = map[string]int{}
	s.mu.Unlock()
}

func (s *Server) releaseAll() {
	s.mu.Lock()
	gates := append([]*Gate(nil), s.gates...)
	s.mu.Unlock()
	for _, g := range gates {
		g.Release()
	}
}

func (s *Server) find(tag string) *Release {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, r := range s.releases {
		if r.Tag == tag {
			return r
		}
	}
	return nil
}

func (s *Server) serve(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	s.reqs = append(s.reqs, Request{Method: r.Method, Host: r.Host, Path: r.URL.Path, Query: r.URL.RawQuery})
	s.mu.Unlock()
	if r.Header.Get("Authorization") != "" || r.Header.Get("Cookie") != "" {
		http.Error(w, "the updater must send no credentials", http.StatusBadRequest)
		return
	}
	switch r.Host {
	case APIHost:
		s.serveList(w, r)
	case DownloadHost:
		s.serveDownloadRedirect(w, r)
	case AssetHost:
		s.serveAsset(w, r)
	default:
		http.Error(w, "host outside the release authority: "+r.Host, http.StatusMisdirectedRequest)
	}
}

func (s *Server) serveList(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet || r.URL.Path != "/repos/"+update.Owner+"/"+update.Repo+"/releases" {
		http.NotFound(w, r)
		return
	}
	s.mu.Lock()
	g, status, drop := s.listGate, s.listStatus, s.listDrop
	s.listGate = nil
	s.mu.Unlock()
	if g != nil {
		g.markReached()
		select {
		case <-g.release:
		case <-g.abort:
			panic(http.ErrAbortHandler)
		case <-r.Context().Done():
			return
		}
	}
	if drop {
		panic(http.ErrAbortHandler)
	}
	if status != 0 {
		http.Error(w, `{"message":"stub release authority unavailable"}`, status)
		return
	}
	if r.URL.Query().Get("page") != "1" {
		fmt.Fprint(w, "[]")
		return
	}
	s.mu.Lock()
	rels := append([]*Release(nil), s.releases...)
	s.mu.Unlock()
	var out []map[string]any
	for i := len(rels) - 1; i >= 0; i-- { // newest published first, like GitHub
		out = append(out, releaseJSON(rels[i]))
	}
	if out == nil {
		out = []map[string]any{}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

func releaseJSON(r *Release) map[string]any {
	asset := func(name string, size int, digest string) map[string]any {
		a := map[string]any{
			"name": name, "size": size, "state": "uploaded",
			"browser_download_url": fmt.Sprintf("https://%s/%s/%s/releases/download/%s/%s", DownloadHost, update.Owner, update.Repo, r.Tag, name),
		}
		if digest != "" {
			a["digest"] = "sha256:" + digest
		}
		return a
	}
	digest := ""
	if r.MetadataDigest {
		sum := sha256.Sum256(r.Exe)
		digest = hex.EncodeToString(sum[:])
	}
	return map[string]any{
		"tag_name": r.Tag, "draft": false, "prerelease": r.Prerelease,
		"published_at": published.Format(time.RFC3339),
		"assets": []map[string]any{
			asset(update.ExeAsset, len(r.Exe), digest),
			asset(update.SumAsset, len(r.Sum), ""),
		},
	}
}

// serveDownloadRedirect answers the canonical download URL the way GitHub does:
// a redirect to the release-asset host.
func (s *Server) serveDownloadRedirect(w http.ResponseWriter, r *http.Request) {
	prefix := "/" + update.Owner + "/" + update.Repo + "/releases/download/"
	rest, ok := strings.CutPrefix(r.URL.Path, prefix)
	if !ok || r.Method != http.MethodGet {
		http.NotFound(w, r)
		return
	}
	http.Redirect(w, r, "https://"+AssetHost+"/blob/"+rest, http.StatusFound)
}

func (s *Server) serveAsset(w http.ResponseWriter, r *http.Request) {
	rest, ok := strings.CutPrefix(r.URL.Path, "/blob/")
	tag, name, ok2 := strings.Cut(rest, "/")
	rel := s.find(tag)
	if !ok || !ok2 || rel == nil {
		http.NotFound(w, r)
		return
	}
	switch name {
	case update.SumAsset:
		w.Header().Set("Content-Length", fmt.Sprint(len(rel.Sum)))
		_, _ = w.Write(rel.Sum)
	case update.ExeAsset:
		s.serveExe(w, r, rel)
	default:
		http.NotFound(w, r)
	}
}

func (s *Server) serveExe(w http.ResponseWriter, r *http.Request, rel *Release) {
	body := rel.body()
	s.mu.Lock()
	hold := s.exeGate[rel.Tag]
	delete(s.exeGate, rel.Tag)
	cut, doCut := s.exeCut[rel.Tag]
	s.mu.Unlock()
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Length", fmt.Sprint(len(body)))
	fl, _ := w.(http.Flusher)
	switch {
	case hold != nil:
		n := min(hold.after, len(body))
		_, _ = w.Write(body[:n])
		if fl != nil {
			fl.Flush()
		}
		hold.gate.markReached()
		select {
		case <-hold.gate.release:
			_, _ = w.Write(body[n:])
		case <-hold.gate.abort:
			panic(http.ErrAbortHandler)
		case <-r.Context().Done():
		}
	case doCut:
		n := min(cut, len(body))
		_, _ = w.Write(body[:n])
		if fl != nil {
			fl.Flush()
		}
		panic(http.ErrAbortHandler)
	default:
		_, _ = w.Write(body)
	}
}
