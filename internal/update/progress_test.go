package update

import (
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

// gateTransport delays the authority the way a slow network does. A request
// whose host and path suffix match stalls until release is closed (before any
// response, or, with midBody, after the first half of the body).
type gateTransport struct {
	next    http.RoundTripper
	host    string
	suffix  string
	midBody bool

	release chan struct{}
	hit     chan struct{}
	once    sync.Once
}

func newGate(next http.RoundTripper, host, suffix string, midBody bool) *gateTransport {
	return &gateTransport{next: next, host: host, suffix: suffix, midBody: midBody, release: make(chan struct{}), hit: make(chan struct{})}
}

func (g *gateTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.URL.Host != g.host || !strings.HasSuffix(req.URL.Path, g.suffix) {
		return g.next.RoundTrip(req)
	}
	if !g.midBody {
		g.once.Do(func() { close(g.hit) })
		<-g.release
		return g.next.RoundTrip(req)
	}
	resp, err := g.next.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	half := len(body) / 2
	resp.Body = &gatedBody{head: body[:half], tail: body[half:], gate: g}
	return resp, nil
}

type gatedBody struct {
	head, tail []byte
	gate       *gateTransport
	passed     bool
}

func (b *gatedBody) Read(p []byte) (int, error) {
	if len(b.head) == 0 && !b.passed {
		b.gate.once.Do(func() { close(b.gate.hit) })
		<-b.gate.release
		b.passed = true
		b.head = b.tail
	}
	if len(b.head) == 0 {
		return 0, io.EOF
	}
	n := copy(p, b.head)
	b.head = b.head[n:]
	return n, nil
}

func (b *gatedBody) Close() error { return nil }

func waitFor(t *testing.T, what string, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if ok() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

// refused fails unless every other update action is rejected as already in
// progress, and reports how many requests the rejected actions made.
func (e *env) refusesEveryOtherAction(tag string) {
	e.t.Helper()
	before := e.g.count()
	errs := map[string]error{
		"StartCheck":    e.svc.StartCheck(),
		"Check":         e.svc.Check(t0()),
		"StartDownload": e.svc.StartDownload(tag),
		"SetChannel":    e.svc.SetChannel(Stable),
		"Install":       e.svc.Install(),
	}
	for name, err := range errs {
		var pe *Error
		if !asError(err, &pe) || pe.Class != ClassRefused {
			e.t.Errorf("%s while work is active: %v", name, err)
		}
	}
	for _, name := range []string{"StartCheck", "Check", "StartDownload", "SetChannel"} {
		if err := errs[name]; err == nil || !strings.Contains(err.Error(), "in progress") {
			e.t.Errorf("%s is not told the work is in progress: %v", name, err)
		}
	}
	if n := e.g.count(); n != before {
		e.t.Errorf("rejected actions made %d requests", n-before)
	}
}

// A delayed GitHub response must leave a visible, accepted check, not an idle
// subsystem, and a repeated action must not start second work.
func TestCheckIsAcceptedAtOnceAndVisibleWhileTheNetworkIsSlow(t *testing.T) {
	e := newEnv(t)
	e.svc.SetChannel(Development)
	e.catalog("0.2.5-dev")
	gate := newGate(e.g, apiHost, "/releases", false)
	e.svc.Transport = gate

	if err := e.svc.StartCheck(); err != nil {
		t.Fatal(err)
	}
	// Accepted: visible in Status before the network has answered anything.
	st := e.svc.Status()
	if st.Busy == nil || st.Busy.Kind != KindCheck || strings.Join(st.Busy.Plan, " ") != PhaseReleases || !st.Busy.Finished.IsZero() {
		t.Fatalf("accepted check not visible: %+v", st.Busy)
	}
	<-gate.hit
	waitFor(t, "the release phase", func() bool {
		b := e.svc.Status().Busy
		return b != nil && b.Phase == PhaseReleases && b.Step == StepFetching
	})
	if b := e.svc.Status().Busy; b.Total != 0 || b.Done != 0 {
		t.Errorf("a phase with no known size reported byte progress: %+v", b)
	}
	e.refusesEveryOtherAction("0.2.5-dev")
	if st := e.svc.Status(); st.Busy == nil || st.Last != nil || st.Check != nil {
		t.Fatalf("a rejected action disturbed the operation: %+v", st)
	}

	close(gate.release)
	waitFor(t, "the check to finish", func() bool { return e.svc.Status().Busy == nil })
	st = e.svc.Status()
	if st.Check == nil || len(st.Check.Releases) != 1 || st.LastCheck == nil || !st.LastCheck.OK {
		t.Fatalf("check result: %+v", st)
	}
	for _, r := range e.g.requests() {
		if !strings.HasPrefix(r, apiHost+"/repos/"+Owner+"/"+Repo+"/releases") {
			t.Errorf("Check requested %s", r)
		}
	}
	// Idle again: the next action is accepted.
	if err := e.svc.StartDownload("0.2.5-dev"); err != nil {
		t.Fatal(err)
	}
	e.wait()
}

func TestCheckFailureEndsTheOperationAndStaysVisible(t *testing.T) {
	e := newEnv(t)
	e.g.apiStatus = http.StatusForbidden
	if err := e.svc.StartCheck(); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the check to finish", func() bool { return e.svc.Status().Busy == nil })
	st := e.svc.Status()
	if st.Check != nil || st.LastCheck == nil || st.LastCheck.OK || !strings.Contains(st.LastCheck.Message, "403") {
		t.Fatalf("failed check: %+v %+v", st.Check, st.LastCheck)
	}
	if err := e.svc.StartCheck(); err != nil {
		t.Fatalf("a failed check must not leave the subsystem busy: %v", err)
	}
	waitFor(t, "the second check to finish", func() bool { return e.svc.Status().Busy == nil })
}

// The checksum request is the first download network wait: the operation is
// accepted, in the checksum phase, with no invented progress.
func TestDownloadIsAcceptedAtOnceAndVisibleWhileTheChecksumIsSlow(t *testing.T) {
	e := newEnv(t)
	e.svc.SetChannel(Development)
	e.catalog("0.2.5-dev")
	e.check()
	gate := newGate(e.g, "objects.githubusercontent.com", "/"+SumAsset, false)
	e.svc.Transport = gate

	if err := e.svc.StartDownload("0.2.5-dev"); err != nil {
		t.Fatal(err)
	}
	if b := e.svc.Status().Busy; b == nil || b.Kind != KindDownload || b.Tag != "0.2.5-dev" || strings.Join(b.Plan, " ") != "checksum download verify" {
		t.Fatalf("accepted download not visible: %+v", b)
	}
	<-gate.hit
	waitFor(t, "the checksum phase", func() bool { b := e.svc.Status().Busy; return b != nil && b.Phase == PhaseChecksum })
	if b := e.svc.Status().Busy; b.Total != 0 {
		t.Errorf("the checksum phase reported a total: %+v", b)
	}
	e.refusesEveryOtherAction("0.2.5-dev")

	close(gate.release)
	e.wait()
	if st := e.svc.Status(); st.Ready == nil || st.Last == nil || st.Last.Failure != nil {
		t.Fatalf("download did not complete: %+v", st)
	}
}

// Byte progress is real: it is the bytes received so far of the size the
// release states, and the phase is the download phase.
func TestActiveDownloadReportsRealBytesAndRefusesRepeatedActions(t *testing.T) {
	e := newEnv(t)
	e.svc.SetChannel(Development)
	e.catalog("0.2.5-dev")
	e.check()
	exe := exeBytes("0.2.5-dev")
	gate := newGate(e.g, "objects.githubusercontent.com", "/"+ExeAsset, true)
	e.svc.Transport = gate

	if err := e.svc.StartDownload("0.2.5-dev"); err != nil {
		t.Fatal(err)
	}
	<-gate.hit
	half := int64(len(exe) / 2)
	waitFor(t, "half of the bytes", func() bool { b := e.svc.Status().Busy; return b != nil && b.Phase == PhaseDownload && b.Done == half })
	b := e.svc.Status().Busy
	if b.Total != int64(len(exe)) || b.Step != StepDownloading || b.Detail != ExeAsset || strings.Join(b.Phases, " ") != "checksum download" {
		t.Fatalf("download progress: %+v", b)
	}
	e.refusesEveryOtherAction("0.2.5-dev")
	if e.svc.Status().Ready != nil {
		t.Fatal("ready before verification")
	}

	close(gate.release)
	op := e.wait()
	if op == nil || op.Failure != nil || strings.Join(op.Phases, " ") != "checksum download verify" || op.Done != op.Total {
		t.Fatalf("completed download: %+v", op)
	}
	if st := e.svc.Status(); st.Busy != nil || st.Ready == nil || st.ReadyProblem != "" {
		t.Fatalf("completion did not make the update ready: %+v", st)
	}
}

// A failure names its phase and leaves nothing partial; an earlier verified
// update stays ready.
func TestDownloadFailureKeepsItsPhaseAndLeavesNoPartialFile(t *testing.T) {
	e := newEnv(t)
	e.svc.SetChannel(Development)
	e.catalog("0.2.5-dev")
	e.check()
	e.g.status["0.2.5-dev/"+SumAsset] = http.StatusNotFound
	op := e.download("0.2.5-dev")
	if op == nil || op.Failure == nil || op.Failure.Phase != PhaseChecksum || !strings.Contains(op.Failure.Message, "404") || op.Finished.IsZero() {
		t.Fatalf("checksum failure: %+v", op)
	}
	delete(e.g.status, "0.2.5-dev/"+SumAsset)

	e.g.blobs["0.2.5-dev/"+ExeAsset] = append(exeBytes("0.2.5-dev"), 'x') // stated size no longer matches
	op = e.download("0.2.5-dev")
	if op == nil || op.Failure == nil || op.Failure.Phase != PhaseDownload {
		t.Fatalf("download failure: %+v", op)
	}
	st := e.svc.Status()
	if st.Busy != nil || st.Ready != nil {
		t.Fatalf("a failed download left work or a ready update: %+v", st)
	}
	if err := e.svc.StartCheck(); err != nil {
		t.Fatalf("a failed download must free the subsystem: %v", err)
	}
	waitFor(t, "the check to finish", func() bool { return e.svc.Status().Busy == nil })
}
