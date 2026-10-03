// Package lab runs the diagnostics E2E scenarios against the real candidate
// executable.
//
// The executable is started as `hachidori dashboard` on a disposable home that
// is shaped like an activated CPU home but cannot start its worker, so the
// bundle describes a real, deterministic failure. The scenario then does what an
// operator does: it opens Diagnostics, presses Export bundle (the dashboard's
// own form, with its token) and inspects the archive the executable wrote.
// Nothing here talks to anything but that local loopback dashboard.
package lab

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// Reporter receives the scenario's bounded evidence; *e2e.Scenario satisfies it.
type Reporter interface {
	Logf(format string, args ...any)
	Attach(name string, data []byte, home string)
}

const (
	startWait = 90 * time.Second
	httpWait  = 30 * time.Second
	pollEvery = 20 * time.Millisecond
	// stderrTail is how much of the executable's own output is kept.
	stderrTail = 32 << 10
)

// until polls cond until it holds or the deadline passes; stop reports an early
// reason to give up (the process exited).
func until(t testing.TB, what string, within time.Duration, stop func() string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(within)
	for !cond() {
		if why := stop(); why != "" {
			t.Fatalf("while waiting for %s: %s", what, why)
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out after %s waiting for %s", within, what)
		}
		time.Sleep(pollEvery)
	}
}

// tailBuffer keeps the last stderrTail bytes written to it.
type tailBuffer struct {
	mu  sync.Mutex
	buf []byte
}

func (b *tailBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.buf = append(b.buf, p...)
	if len(b.buf) > stderrTail {
		b.buf = append([]byte(nil), b.buf[len(b.buf)-stderrTail:]...)
	}
	return len(p), nil
}

func (b *tailBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return string(b.buf)
}

// host is one running `hachidori dashboard` process.
type host struct {
	t         testing.TB
	cmd       *exec.Cmd
	dashAddr  string
	home      string
	stderr    *tailBuffer
	done      chan struct{}
	exitState string
	client    *http.Client
}

// freeAddr returns a loopback address that was free a moment ago.
func freeAddr() (string, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", err
	}
	defer l.Close()
	return l.Addr().String(), nil
}

// startDashboard starts exe's dashboard on home with exactly env as its
// environment. The process is ended when the test ends.
func startDashboard(t testing.TB, exe, home string, env []string) *host {
	t.Helper()
	api, err := freeAddr()
	if err != nil {
		t.Fatal(err)
	}
	dash, err := freeAddr()
	if err != nil {
		t.Fatal(err)
	}
	h := &host{t: t, dashAddr: dash, home: home, stderr: &tailBuffer{}, done: make(chan struct{}),
		client: &http.Client{Timeout: httpWait, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}}
	cmd := exec.Command(exe, "dashboard", "--home", home, "--listen", api, "--addr", dash)
	cmd.Env = env
	cmd.Stderr, cmd.Stdout = h.stderr, h.stderr
	if err := cmd.Start(); err != nil {
		t.Fatalf("start the executable under test: %v", err)
	}
	h.cmd = cmd
	go func() {
		err := cmd.Wait()
		h.exitState = fmt.Sprint(err)
		close(h.done)
	}()
	t.Cleanup(h.stop)
	return h
}

func (h *host) stop() {
	if h.cmd == nil {
		return
	}
	_ = h.cmd.Process.Kill()
	<-h.done
	h.cmd = nil
}

func (h *host) exited() string {
	select {
	case <-h.done:
		return "the executable exited (" + h.exitState + "): " + trim(h.stderr.String(), 600)
	default:
		return ""
	}
}

func trim(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) > n {
		return "..." + s[len(s)-n:]
	}
	return s
}

func (h *host) url(path string) string { return "http://" + h.dashAddr + path }

// status reads the executable's own status document.
func (h *host) status() (server.Status, error) {
	resp, err := h.client.Get(h.url("/api/status"))
	if err != nil {
		return server.Status{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return server.Status{}, fmt.Errorf("status: %s", resp.Status)
	}
	var st server.Status
	err = json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&st)
	return st, err
}

// waitFailedWorker waits until the dashboard answers and the worker has ended
// its first start in the failed state, and returns that status.
func (h *host) waitFailedWorker() server.Status {
	h.t.Helper()
	var st server.Status
	until(h.t, "the dashboard to report the failed worker start", startWait, h.exited, func() bool {
		s, err := h.status()
		if err != nil || s.Worker.State != worker.StateFailed {
			return false
		}
		st = s
		return true
	})
	return st
}

var tokenField = regexp.MustCompile(`name="token" value="([^"]+)"`)

// exportBundle presses Diagnostics > Export bundle and returns the files the
// executable wrote under state/diagnostics.
func (h *host) exportBundle() []string {
	h.t.Helper()
	resp, err := h.client.Get(h.url("/diagnostics"))
	if err != nil {
		h.t.Fatalf("open Diagnostics: %v", err)
	}
	page, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		h.t.Fatalf("open Diagnostics: %s", resp.Status)
	}
	m := tokenField.FindSubmatch(page)
	if m == nil {
		h.t.Fatal("the Diagnostics page carries no form token")
	}
	form := url.Values{"token": {string(m[1])}}
	req, err := http.NewRequest(http.MethodPost, h.url("/diagnostics/export"), strings.NewReader(form.Encode()))
	if err != nil {
		h.t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Origin", "http://"+h.dashAddr)
	resp, err = h.client.Do(req)
	if err != nil {
		h.t.Fatalf("Export bundle: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusSeeOther {
		h.t.Fatalf("Export bundle: %s, want a redirect back to Diagnostics", resp.Status)
	}
	ents, err := os.ReadDir(filepath.Join(h.home, "state", "diagnostics"))
	if err != nil {
		h.t.Fatalf("the export wrote no state/diagnostics folder: %v", err)
	}
	var names []string
	for _, e := range ents {
		names = append(names, e.Name())
	}
	return names
}

func readCapped(path string, max int) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, int64(max)+1))
	if err == nil && len(data) > max {
		err = fmt.Errorf("%s is larger than %d bytes", path, max)
	}
	return data, err
}

func indent(raw []byte) []byte {
	var b bytes.Buffer
	if json.Indent(&b, raw, "", "  ") != nil {
		return raw
	}
	return b.Bytes()
}
