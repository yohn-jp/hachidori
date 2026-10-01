package update

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"github.com/yohn-jp/hachidori/internal/home"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeGitHub is the deterministic stand-in for the release authority. It is
// the Service's Transport, so no test reaches a real network; it records every
// request it receives and fails the test on a host the authority never uses.
type fakeGitHub struct {
	t *testing.T

	mu    sync.Mutex
	reqs  []string
	pages map[int]string    // releases list pages (1-based)
	blobs map[string][]byte // "tag/name" -> body
	// overrides
	status     map[string]int // "tag/name" -> status for the asset download
	redirectTo map[string]string
	apiStatus  int
	apiRedir   string
}

func newFake(t *testing.T) *fakeGitHub {
	return &fakeGitHub{t: t, pages: map[int]string{}, blobs: map[string][]byte{}, status: map[string]int{}, redirectTo: map[string]string{}}
}

func hexSum(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }

// exeBytes is a fixture Windows executable.
func exeBytes(tag string) []byte {
	return append([]byte("MZ"), []byte("hachidori "+tag+strings.Repeat("x", 4096))...)
}

// publish adds a release with a well-formed executable and checksum asset and
// returns the executable bytes.
func (g *fakeGitHub) publish(tag string, prerelease bool) []byte {
	b := exeBytes(tag)
	g.blobs[tag+"/"+ExeAsset] = b
	g.blobs[tag+"/"+SumAsset] = []byte(hexSum(b) + "  " + ExeAsset + "\r\n")
	return b
}

// list renders all releases published so far (newest first as given).
func (g *fakeGitHub) setList(entries ...string) { g.pages[1] = "[" + strings.Join(entries, ",") + "]" }

func (g *fakeGitHub) entry(tag string, prerelease bool) string {
	exe := g.blobs[tag+"/"+ExeAsset]
	sum := g.blobs[tag+"/"+SumAsset]
	return rel(tag, prerelease, false, asset(tag, ExeAsset, int64(len(exe))), asset(tag, SumAsset, int64(len(sum))))
}

func (g *fakeGitHub) requests() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]string(nil), g.reqs...)
}

func (g *fakeGitHub) count() int { return len(g.requests()) }

func (g *fakeGitHub) RoundTrip(req *http.Request) (*http.Response, error) {
	g.mu.Lock()
	g.reqs = append(g.reqs, req.URL.Host+req.URL.Path)
	g.mu.Unlock()
	resp := func(code int, body string, hdr ...string) (*http.Response, error) {
		h := http.Header{}
		for i := 0; i+1 < len(hdr); i += 2 {
			h.Set(hdr[i], hdr[i+1])
		}
		return &http.Response{StatusCode: code, Status: fmt.Sprintf("%d %s", code, http.StatusText(code)), Header: h,
			Body: io.NopCloser(strings.NewReader(body)), ContentLength: int64(len(body)), Request: req}, nil
	}
	if req.Method != http.MethodGet || req.URL.Scheme != "https" {
		g.t.Errorf("unexpected request %s %s", req.Method, req.URL)
	}
	if req.Header.Get("Authorization") != "" {
		g.t.Error("the updater must send no credentials")
	}
	switch req.URL.Host {
	case apiHost:
		if req.URL.Path != "/repos/"+Owner+"/"+Repo+"/releases" {
			g.t.Errorf("unexpected API path %s", req.URL.Path)
			return resp(404, "")
		}
		if g.apiRedir != "" {
			return resp(302, "", "Location", g.apiRedir)
		}
		if g.apiStatus != 0 {
			return resp(g.apiStatus, `{"message":"API rate limit exceeded"}`)
		}
		var page int
		fmt.Sscan(req.URL.Query().Get("page"), &page)
		if b, ok := g.pages[page]; ok {
			return resp(200, b)
		}
		return resp(200, "[]")
	case downloadHost:
		prefix := "/" + Owner + "/" + Repo + "/releases/download/"
		key, ok := strings.CutPrefix(req.URL.Path, prefix)
		if !ok {
			g.t.Errorf("unexpected download path %s", req.URL.Path)
			return resp(404, "")
		}
		if code := g.status[key]; code != 0 {
			return resp(code, "")
		}
		if to := g.redirectTo[key]; to != "" {
			return resp(302, "", "Location", to)
		}
		return resp(302, "", "Location", "https://objects.githubusercontent.com/blob/"+key)
	case "objects.githubusercontent.com":
		key, _ := strings.CutPrefix(req.URL.Path, "/blob/")
		b, ok := g.blobs[key]
		if !ok {
			return resp(404, "")
		}
		return resp(200, string(b))
	}
	g.t.Errorf("request to a host outside the release authority: %s", req.URL)
	return nil, fmt.Errorf("unexpected host %s", req.URL.Host)
}

// memStore is an in-memory Store.
type memStore struct {
	mu sync.Mutex
	s  Settings
}

func (m *memStore) UpdateSettings() (Settings, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.s.Clone(), nil
}

func (m *memStore) ModifyUpdateSettings(fn func(*Settings) error) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	c := m.s.Clone()
	if err := fn(&c); err != nil {
		return err
	}
	if err := c.Validate(); err != nil {
		return err
	}
	m.s = c
	return nil
}

// env is one installation under test: a home, an executable and a service
// routed to the fake authority.
type env struct {
	t       *testing.T
	g       *fakeGitHub
	root    string
	exe     string
	exeData []byte
	set     *memStore
	svc     *Service
	started [][]string // helper starts: exe, args...
	clock   time.Time
}

func newEnv(t *testing.T) *env {
	t.Helper()
	dir := t.TempDir()
	e := &env{t: t, g: newFake(t), root: filepath.Join(dir, "home"), set: &memStore{}, clock: time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)}
	if err := os.MkdirAll(filepath.Join(e.root, "state"), 0o755); err != nil {
		t.Fatal(err)
	}
	e.exe = filepath.Join(dir, "app", "hachidori.exe")
	e.exeData = exeBytes("installed-build")
	if err := os.MkdirAll(filepath.Dir(e.exe), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(e.exe, e.exeData, 0o755); err != nil {
		t.Fatal(err)
	}
	e.svc = &Service{
		Home: func() string { return e.root }, Exe: e.exe, Settings: e.set, Transport: e.g,
		StartHelper: func(exe string, args []string) error {
			e.started = append(e.started, append([]string{exe}, args...))
			return nil
		},
		Now: func() time.Time { e.clock = e.clock.Add(time.Second); return e.clock },
	}
	return e
}

func (e *env) check() {
	e.t.Helper()
	if err := e.svc.Check(t0()); err != nil {
		e.t.Fatalf("check: %v", err)
	}
}

// download starts a download and waits for it to finish.
func (e *env) download(tag string) *Operation {
	e.t.Helper()
	if err := e.svc.StartDownload(tag); err != nil {
		e.t.Fatalf("StartDownload(%s): %v", tag, err)
	}
	return e.wait()
}

func (e *env) wait() *Operation {
	e.t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if st := e.svc.Status(); st.Busy == nil {
			return st.Last
		}
		time.Sleep(2 * time.Millisecond)
	}
	e.t.Fatal("download did not finish")
	return nil
}

type blockTransport struct{}

func (blockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	<-req.Context().Done()
	return nil, req.Context().Err()
}

func writeJSONFile(t *testing.T, path string, v any) {
	t.Helper()
	if err := home.WriteJSON(path, v); err != nil {
		t.Fatal(err)
	}
}
