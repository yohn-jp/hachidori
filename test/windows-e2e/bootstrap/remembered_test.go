package bootstrap

import (
	"os"
	"testing"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/firstrun"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/test/windows-e2e/desktopkit"
	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

// TestRememberedHomeResumes certifies, for a home path with spaces, with
// Unicode and a supported deep path, that a home remembered in the bootstrap
// locator is the one the packaged executable offers: it is accepted by the
// first-run validation, reported with its free space, and nothing is installed
// or rewritten (W02.2 through the controller's own first-run projection).
func TestRememberedHomeResumes(t *testing.T) {
	s := e2e.Begin(t, "bootstrap-remembered-home")
	r := desktopkit.NewRun(t, s)

	for _, shape := range desktopkit.Shapes() {
		func() {
			homeRoot, err := shape.Path(r.Path("homes", shape.Name))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.MkdirAll(homeRoot, 0o755); err != nil {
				t.Fatalf("create the %s home: %v", shape.Name, err)
			}
			locator, err := desktopkit.WriteLocator(r.Profile.LocatorPath(), homeRoot)
			if err != nil {
				t.Fatal(err)
			}

			i := r.Desktop("resume-" + shape.Name)
			v := wantMode(t, i, firstrun.ModeResume)
			sel := v.Selection
			switch {
			case v.State != app.NotInstalled || v.Stage != firstrun.StageSelect || v.Operation != nil || v.Failure != nil:
				t.Errorf("a remembered, empty home must wait for Install: %+v", v)
			case sel == nil:
				t.Fatal("the remembered home was not offered as the selection")
			case !sel.OK() || sel.Picked != homeRoot || sel.Home != homeRoot || sel.Nested || sel.Existing != nil:
				t.Errorf("selection %+v, want the remembered %s home accepted as is", sel, shape.Name)
			case sel.FreeBytes == nil || *sel.FreeBytes == 0:
				t.Errorf("selection %+v reports no free space", sel)
			default:
				s.Logf("%s home (%d characters): accepted, free space reported (%d GiB)", shape.Name, len(homeRoot), *sel.FreeBytes>>30)
			}

			// Resuming installs nothing and rewrites nothing.
			if err := desktopkit.Unchanged(r.Profile.LocatorPath(), locator); err != nil {
				t.Error(err)
			}
			noInstall(t, homeRoot)
			r.Stop(i)
			if err := desktopkit.Unchanged(r.Profile.LocatorPath(), locator); err != nil {
				t.Error(err)
			}
		}()
	}
}

// TestRememberedInstalledHomeLaunches certifies that a remembered home that is
// already installed goes straight to normal startup, never the wizard, and
// binds exactly that home's runtime without starting its worker, for each path
// shape. The home is the installed-home fixture (stand-in worker).
func TestRememberedInstalledHomeLaunches(t *testing.T) {
	s := e2e.Begin(t, "bootstrap-remembered-installed-home")
	r := desktopkit.NewRun(t, s)

	for _, shape := range desktopkit.Shapes() {
		func() {
			homeRoot, err := shape.Path(r.Path("installed", shape.Name))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := desktopkit.InstallFixture(homeRoot); err != nil {
				t.Fatalf("install the fixture home: %v", err)
			}
			locator, err := desktopkit.WriteLocator(r.Profile.LocatorPath(), homeRoot)
			if err != nil {
				t.Fatal(err)
			}

			i := r.Desktop("launch-" + shape.Name)
			v, err := i.WaitWizard(startupWait, "the dashboard of the remembered home", func(v firstrun.View) bool { return v.Mode == firstrun.ModeLaunch && v.Dashboard })
			if err != nil {
				t.Fatal(err)
			}
			if got := i.StartMode(); got != string(firstrun.ModeLaunch) {
				t.Errorf("startup mode %q, want launch; output:\n%s", got, i.Output())
			}
			if v.Failure != nil || v.Stage == firstrun.StageFailed || v.Selection != nil {
				t.Errorf("normal startup must not show the wizard or a failure: %+v", v)
			}
			if err := i.WaitShell(startupWait); err != nil {
				t.Fatal(err)
			}

			// The runtime that was bound is the remembered home's, and no worker runs
			// until the operator starts one.
			var st server.Status
			if err := desktopkit.Poll(startupWait, "the bound runtime status", func() (bool, error) {
				var e error
				st, e = desktopkit.Status(i.API)
				return e == nil, e
			}); err != nil {
				t.Fatal(err)
			}
			if st.Runtime.Home != homeRoot || st.Runtime.ModelID != desktopkit.FixtureModel || st.Runtime.Device != desktopkit.FixtureDevice {
				t.Errorf("bound runtime %+v, want the remembered home %s", st.Runtime, homeRoot)
			}
			if st.Worker.PID != 0 || st.Worker.Ready {
				t.Errorf("normal startup started a worker: %+v", st.Worker)
			}
			if procs, err := desktopkit.Processes(); err != nil {
				t.Fatal(err)
			} else if w := desktopkit.ChildrenNamed(procs, i.Pid, "python.exe"); len(w) != 0 {
				t.Errorf("worker processes %v exist before Start", desktopkit.PIDs(w))
			}
			s.Logf("%s home (%d characters): launched from the locator, runtime %s bound, no worker", shape.Name, len(homeRoot), st.Runtime.Runtime)

			if err := desktopkit.Unchanged(r.Profile.LocatorPath(), locator); err != nil {
				t.Error(err)
			}
			if rs := desktopkit.Residue(homeRoot); len(rs) != 0 {
				t.Errorf("owned staging or temporary files remain under the home: %v", rs)
			}
			r.Stop(i)
			if _, err := os.Stat(homeRoot); err != nil {
				t.Errorf("the remembered home is gone after the run: %v", err)
			}
		}()
	}
}
