package bootstrap

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/firstrun"
	"github.com/yohn-jp/hachidori/test/windows-e2e/desktopkit"
	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

// TestMissingHomeGivesRecoveryNamingTheFolder proves W04.2: a remembered home
// that was renamed, moved, deleted or replaced by a file does not exit and is
// not a first run; the recovery screen names the folder, and nothing is
// installed, created or deleted.
func TestMissingHomeGivesRecoveryNamingTheFolder(t *testing.T) {
	s := e2e.Begin(t, "bootstrap-missing-home")
	r := desktopkit.NewRun(t, s)

	gone := r.Path("moved away", "Hachidori ハチドリ renamed")
	asFile := r.Path("files", "now a file")
	if err := os.MkdirAll(filepath.Dir(asFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(asFile, []byte("not a folder"), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, c := range []struct{ name, stored string }{{"missing-folder", gone}, {"folder-replaced-by-file", asFile}} {
		func() {
			locator, err := desktopkit.WriteLocator(r.Profile.LocatorPath(), c.stored)
			if err != nil {
				t.Fatal(err)
			}
			i := r.Desktop(c.name)
			v := wantMode(t, i, firstrun.ModeMissing)
			if v.State != app.Unconfigured || v.Stage != firstrun.StageSelect || v.StoredHome != c.stored ||
				v.Selection != nil || v.Operation != nil || v.Dashboard || v.Failure != nil {
				t.Errorf("%s: recovery view %+v, want an untouched selection screen naming %s", c.name, v, c.stored)
			}
			contains(t, v.Notice, c.stored, c.name+" notice")
			contains(t, v.Notice, "missing or unavailable", c.name+" notice")
			contains(t, v.Notice, "Nothing has been installed or deleted", c.name+" notice")
			s.Logf("%s: recovery shown, folder named, nothing installed", c.name)

			// Installs nothing, creates nothing, deletes nothing, rewrites nothing.
			if c.stored == gone {
				notExist(t, gone, "the missing home")
				notExist(t, filepath.Dir(gone), "the parent of the missing home")
			} else if b, err := os.ReadFile(asFile); err != nil || string(b) != "not a folder" {
				t.Errorf("the file at the stored home was changed: %q, %v", b, err)
			}
			if err := desktopkit.Unchanged(r.Profile.LocatorPath(), locator); err != nil {
				t.Error(err)
			}
			noInstall(t, r.Root)
			r.Stop(i)
			if err := desktopkit.Unchanged(r.Profile.LocatorPath(), locator); err != nil {
				t.Error(err)
			}
		}()
	}
}

// TestCorruptLocatorGivesDiagnosticRecovery proves W04.3: every corrupt
// bootstrap.json gives a diagnostic recovery state, not an exit and not a first
// run, and the corrupt record is left exactly as it was (never repaired or
// deleted behind the user's back).
func TestCorruptLocatorGivesDiagnosticRecovery(t *testing.T) {
	s := e2e.Begin(t, "bootstrap-corrupt-locator")
	r := desktopkit.NewRun(t, s)
	homeRoot := r.Path("homes", "Hachidori Data ハチドリ")
	if err := os.MkdirAll(homeRoot, 0o755); err != nil {
		t.Fatal(err)
	}

	for _, c := range desktopkit.Corruptions(homeRoot) {
		func() {
			if err := desktopkit.WriteRaw(r.Profile.LocatorPath(), c.Data); err != nil {
				t.Fatal(err)
			}
			i := r.Desktop(c.Name)
			v := wantMode(t, i, firstrun.ModeInvalid)
			if v.State != app.Unconfigured || v.Stage != firstrun.StageSelect || v.Selection != nil ||
				v.Operation != nil || v.Dashboard || v.Failure != nil {
				t.Errorf("%s: recovery view %+v, want an untouched selection screen", c.Name, v)
			}
			contains(t, v.Notice, "could not read its saved storage location", c.Name+" notice")
			contains(t, v.Notice, "bootstrap locator", c.Name+" notice diagnostic")
			contains(t, v.Notice, "Nothing has been installed or deleted", c.Name+" notice")
			s.Logf("%s: diagnostic recovery shown and the process stayed up", c.Name)

			if err := desktopkit.Unchanged(r.Profile.LocatorPath(), c.Data); err != nil {
				t.Errorf("%s: %v", c.Name, err)
			}
			noInstall(t, homeRoot)
			r.Stop(i)
			if err := desktopkit.Unchanged(r.Profile.LocatorPath(), c.Data); err != nil {
				t.Errorf("%s after exit: %v", c.Name, err)
			}
		}()
	}
}
