package bootstrap

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/firstrun"
	"github.com/yohn-jp/hachidori/test/windows-e2e/desktopkit"
)

// startupWait bounds how long a cold candidate may take to serve its first
// screen on a hosted runner (loader, Defender scan, WebView2 start).
const startupWait = 90 * time.Second

// wantMode waits until the instance serves a first-run state and checks that it
// is the expected startup mode, in the log and in the served view. It then waits
// for the shell to finish its first navigation, so the application is known to
// be up and to stay up (the "not an exit" half of every recovery assertion).
func wantMode(t *testing.T, i *desktopkit.Instance, mode firstrun.Mode) firstrun.View {
	t.Helper()
	v, err := i.WaitWizard(startupWait, "a first-run state", func(v firstrun.View) bool { return v.Mode != "" })
	if err != nil {
		t.Fatal(err)
	}
	if v.Mode != mode {
		t.Fatalf("startup mode %q, want %q; view %+v\noutput:\n%s", v.Mode, mode, v, i.Output())
	}
	if got := i.StartMode(); got != string(mode) {
		t.Fatalf("the process logged startup mode %q, want %q; output:\n%s", got, mode, i.Output())
	}
	if err := i.WaitShell(startupWait); err != nil {
		t.Fatal(err)
	}
	if !i.Alive() {
		t.Fatalf("the process exited after reaching %s; output:\n%s", mode, i.Output())
	}
	return v
}

// noInstall fails if root holds any trace of an installation or of Hachidori's
// own staging.
func noInstall(t *testing.T, root string) {
	t.Helper()
	if m := desktopkit.InstallMarkers(root); len(m) != 0 {
		t.Errorf("an installation or setup run left %v under %s", m, root)
	}
	if r := desktopkit.Residue(root); len(r) != 0 {
		t.Errorf("owned staging or temporary files remain under %s: %v", root, r)
	}
}

// notExist fails if path exists.
func notExist(t *testing.T, path, what string) {
	t.Helper()
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("%s exists (err=%v), want it absent", what, err)
	}
}

func contains(t *testing.T, text, want, what string) {
	t.Helper()
	if !strings.Contains(text, want) {
		t.Errorf("%s = %q, want it to contain %q", what, text, want)
	}
}
