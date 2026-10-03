package desktopkit

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

// Attached output bounds: a failed scenario keeps the whole retained tail of
// every process, a passing one only its end. Both stay far below the shard's
// evidence budget.
const (
	attachOnFail = TailBytes
	attachOnPass = 4 << 10
)

// Run is the disposable world of one scenario: a short temporary root, a
// disposable user profile inside it, a launcher for the verified candidate, and
// every process the scenario started, which it always ends and whose output it
// keeps as bounded evidence.
type Run struct {
	T       *testing.T
	S       *e2e.Scenario
	Root    string
	Profile Profile
	L       Launcher

	procs []tracked
	n     int
}

type tracked struct {
	label string
	p     *Proc
}

// NewRun creates the scenario's disposable world and registers its cleanup.
// Call it right after e2e.Begin.
func NewRun(t *testing.T, s *e2e.Scenario) *Run {
	t.Helper()
	version, err := EnsureWebView2()
	if err != nil {
		t.Fatal(err)
	}
	s.Logf("WebView2 Runtime %s", version)
	root, cleanup, err := MkdirTemp("hb-")
	if err != nil {
		t.Fatalf("disposable root: %v", err)
	}
	prof, err := NewProfile(root)
	if err != nil {
		cleanup()
		t.Fatal(err)
	}
	r := &Run{T: t, S: s, Root: root, Profile: prof, L: Launcher{Exe: s.Candidate().Path, Profile: prof}}
	t.Cleanup(func() {
		failed := t.Failed()
		for _, tp := range r.procs {
			tp.p.Kill()
			out := tp.p.Output()
			if !failed && len(out) > attachOnPass {
				out = "[end of output]\n" + out[len(out)-attachOnPass:]
			}
			s.Attach(fmt.Sprintf("%s-%s.log", t.Name(), tp.label), []byte(out), r.Root)
		}
		cleanup()
	})
	return r
}

// Track registers a process so it is ended and its output retained.
func (r *Run) Track(label string, p *Proc) *Proc {
	r.n++
	r.procs = append(r.procs, tracked{label: fmt.Sprintf("%02d-%s", r.n, label), p: p})
	return p
}

// Path is a path beneath the scenario root.
func (r *Run) Path(parts ...string) string {
	return filepath.Join(append([]string{r.Root}, parts...)...)
}

// Desktop starts `hachidori desktop` on fresh loopback addresses.
func (r *Run) Desktop(label string) *Instance {
	r.T.Helper()
	i, err := r.L.Desktop()
	if err != nil {
		r.T.Fatalf("start %s: %v", label, err)
	}
	r.Track(label, i.Proc)
	return i
}

// NoArg starts the no-argument (double-click) entry point.
func (r *Run) NoArg(label string) *Instance {
	r.T.Helper()
	i, err := r.L.NoArg()
	if err != nil {
		r.T.Fatalf("start %s: %v", label, err)
	}
	r.Track(label, i.Proc)
	return i
}

// Cmd starts a CLI command of the candidate.
func (r *Run) Cmd(label string, args ...string) *Proc {
	r.T.Helper()
	p, err := r.L.Run(args...)
	if err != nil {
		r.T.Fatalf("start %s: %v", label, err)
	}
	return r.Track(label, p)
}

// Stop ends one instance and everything it started, and waits for it.
func (r *Run) Stop(i *Instance) {
	r.T.Helper()
	i.Kill()
	if i.Alive() {
		r.T.Errorf("pid %d is still running after it was terminated", i.Pid)
	}
}
