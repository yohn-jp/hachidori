package harness

import (
	"fmt"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/server"
)

// Check fails the test with what and err when err is not nil.
func Check(t testing.TB, what string, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: %v", what, err)
	}
}

// ExpectIdentity checks that the status document names the fixture home's
// model, device and runtime, and that the worker reports the fixture provider
// on the requested device.
func (s *Session) ExpectIdentity(st server.Status) {
	t := s.T
	t.Helper()
	r := st.Runtime
	if r.ModelID != s.Fix.ModelID || r.Device != s.Fix.Device || r.Runtime != s.Fix.RuntimeID {
		t.Fatalf("runtime identity model %q device %q runtime %q, want %q %q %q",
			r.ModelID, r.Device, r.Runtime, s.Fix.ModelID, s.Fix.Device, s.Fix.RuntimeID)
	}
	info := st.Worker.Info
	if info["provider"] != "laya" || info["device"] != s.Fix.Device ||
		!strings.Contains(fmt.Sprint(info["provider_version"]), "e2e-fixture") || !strings.Contains(fmt.Sprint(info["torch_version"]), "e2e-fixture") {
		t.Fatalf("the worker did not report the fixture provider on %s: provider %v device %v", s.Fix.Device, info["provider"], info["device"])
	}
	if r.Worker == nil || r.Worker.SHA256 == "" {
		t.Fatalf("the status carries no delivered worker identity")
	}
}

// WorkerScriptPath is where the executable delivered the worker script it runs
// for the build identity the status reports.
func (s *Session) WorkerScriptPath(st server.Status) string {
	return s.Fix.Home.Path("workers", st.Runtime.Worker.SHA256, "hachidori_worker.py")
}

// ExpectWorkerScript checks that the worker the executable runs is the script it
// delivered into the home, content addressed by the digest the status reports.
func (s *Session) ExpectWorkerScript(st server.Status) {
	t := s.T
	t.Helper()
	sum, err := FileSHA256(s.WorkerScriptPath(st))
	Check(t, "delivered worker script", err)
	if sum != st.Runtime.Worker.SHA256 {
		t.Fatalf("the delivered worker script has digest %s, its directory and the status say %s", sum, st.Runtime.Worker.SHA256)
	}
}

// Typed sends a typed decision over the real HTTP API and checks that the answer
// is the fixture model's, computed by the worker, for this exact state.
func (s *Session) Typed(state, qid string, choices []string) {
	t := s.T
	t.Helper()
	res, err := s.Client.Decide(TypedRequest(state, qid, choices))
	Check(t, "POST /v1/decide", err)
	if res.Code != 200 {
		t.Fatalf("POST /v1/decide answered %d: %s: %s", res.Code, res.Error.Error.Class, res.Error.Error.Message)
	}
	Check(t, "typed answer", CheckAnswer(res.Response, qid, state, choices))
}

// ExpectRefused checks that no typed decision is answered right now.
func (s *Session) ExpectRefused(what string) {
	t := s.T
	t.Helper()
	res, err := s.Client.Decide(TypedRequest("a state", "q", []string{"a", "b"}))
	if err == nil && res.Code == 200 {
		t.Fatalf("%s: a typed decision was answered", what)
	}
}

// ExpectNotServing checks that /health is not 200 (or unreachable).
func (s *Session) ExpectNotServing(what string) {
	t := s.T
	t.Helper()
	if code, h, err := s.Client.Health(); err == nil && (code == 200 || h.Ready) {
		t.Fatalf("%s: health %d %+v, want not ready", what, code, h)
	}
}
