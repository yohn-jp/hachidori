package bootstrap

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/firstrun"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
	"github.com/yohn-jp/hachidori/test/windows-e2e/desktopkit"
	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

// choosing is the folder chooser of the flow: it answers with the folder the
// scenario chose, standing in for the native dialog (W02.1 stays physical).
type choosing struct{ path string }

func (c choosing) PickFolder(context.Context, string) (string, error) { return c.path, nil }

// fixtureSetup is the setup step of this scenario: it materializes the
// installed-home fixture instead of downloading a runtime and a model. Path
// validation, the free-space query, the controller's lifecycle, the bootstrap
// commit and the worker supervision around it are the production code.
func fixtureSetup(root, _, _ string, _ io.Writer, obs *setup.Observer) error {
	for _, ph := range []setup.Phase{setup.PhasePreparing, setup.PhaseRuntime, setup.PhaseModel, setup.PhaseActivation} {
		obs.OnPhase(ph)
	}
	_, err := desktopkit.InstallFixture(root)
	return err
}

// TestInstallCommitsOnlySchemaAndHome certifies, on the Windows filesystem,
// the bootstrap commit of the first-run flow for each path shape: the chosen
// folder (spaces, Unicode, supported deep path) is accepted and reported with
// its free space by the flow, and after the install reaches READY bootstrap.json
// in the (disposable) per-user profile holds exactly "schema" and "home", the
// heavy state lives under the chosen home and nowhere else, and the locator's
// atomic write left no temporary file behind. It runs the production flow and
// controller in this process, not the packaged executable, and its setup step
// is the fixture materializer; it does not certify a real setup.
func TestInstallCommitsOnlySchemaAndHome(t *testing.T) {
	s := e2e.Begin(t, "bootstrap-commit-record")
	r := desktopkit.NewRun(t, s)
	// The locator resolves %LOCALAPPDATA% of this process: point it into the
	// disposable profile for the duration of the scenario.
	t.Setenv("LOCALAPPDATA", r.Profile.LocalAppData)
	t.Setenv("HACHIDORI_HOME", "")

	for _, shape := range desktopkit.Shapes() {
		func() {
			chosen, err := shape.Path(r.Path("chosen", shape.Name))
			if err != nil {
				t.Fatal(err)
			}
			notExist(t, r.Profile.LocatorPath(), "bootstrap.json before the install")

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			env := firstrun.Env{FreeSpace: firstrun.DefaultFreeSpace}
			ctl := app.New(app.Config{
				Open:      app.WorkerRuntime(ctx, io.Discard, worker.DefaultPolicy, nil),
				Setup:     fixtureSetup,
				Installed: env.IsInstalled,
			})
			defer func() {
				cctx, ccancel := context.WithTimeout(context.Background(), 30*time.Second)
				defer ccancel()
				if err := ctl.Close(cctx); err != nil {
					t.Errorf("%s: close the controller: %v", shape.Name, err)
				}
			}()
			flow := firstrun.New(firstrun.Config{Ctl: ctl, Plan: firstrun.Plan{Mode: firstrun.ModeFirstRun},
				Picker: choosing{chosen}, Env: env, Remember: home.Remember})

			// The chosen folder is accepted and reported with its free space.
			v, err := flow.Select(ctx)
			if err != nil {
				t.Fatalf("%s: select: %v", shape.Name, err)
			}
			if !v.OK() || v.Picked != chosen || v.Home != chosen || v.Nested || v.Exists || v.Existing != nil || v.FreeBytes == nil || *v.FreeBytes == 0 {
				t.Fatalf("%s: validation of %s = %+v", shape.Name, chosen, v)
			}
			s.Logf("%s folder (%d characters) accepted, free space %d GiB", shape.Name, len(chosen), *v.FreeBytes>>30)
			notExist(t, r.Profile.LocatorPath(), "bootstrap.json after selecting a folder (nothing is written before Install)")
			notExist(t, chosen, "the chosen folder before Install")

			if err := flow.Install("cpu"); err != nil {
				t.Fatalf("%s: install: %v", shape.Name, err)
			}
			var view firstrun.View
			if err := desktopkit.Poll(60*time.Second, shape.Name+" install to READY", func() (bool, error) {
				view = flow.View()
				if view.Stage == firstrun.StageFailed {
					return false, desktopkit.Fatal{Err: fmt.Errorf("install failed: %+v", view.Failure)}
				}
				return view.Stage == firstrun.StageReady, nil
			}); err != nil {
				t.Fatal(err)
			}
			if view.Identity == nil || view.Identity.Home != chosen || view.Identity.Device != desktopkit.FixtureDevice || view.Identity.ModelID != desktopkit.FixtureModel {
				t.Errorf("%s: READY identity %+v, want the chosen home", shape.Name, view.Identity)
			}

			// bootstrap.json holds only schema and home, and names the chosen folder.
			data, err := os.ReadFile(r.Profile.LocatorPath())
			if err != nil {
				t.Fatalf("%s: bootstrap.json after READY: %v", shape.Name, err)
			}
			if err := desktopkit.CheckLocatorShape(data, chosen); err != nil {
				t.Errorf("%s: %v", shape.Name, err)
			}
			if h, found, err := (home.Locator{Path: r.Profile.LocatorPath()}).Lookup(); err != nil || !found || h.Root != chosen {
				t.Errorf("%s: the production reader resolves %v %v %v, want %s", shape.Name, h.Root, found, err, chosen)
			}
			// The profile holds the locator and nothing else; no atomic-write temporary remains.
			if got := desktopkit.Tree(r.Profile.LocatorDir(), 20); len(got) != 1 || got[0] != "bootstrap.json" {
				t.Errorf("%s: the profile's Hachidori folder holds %v, want only bootstrap.json", shape.Name, got)
			}
			// The heavy state is under the chosen home.
			tree := desktopkit.Tree(chosen, 0)
			for _, want := range []string{"state/active-runtime.json", "runtime/", "models/", "logs/setup.log"} {
				if !desktopkit.Has(tree, want) {
					t.Errorf("%s: the chosen home lacks %s", shape.Name, want)
				}
			}
			if rs := desktopkit.Residue(chosen); len(rs) != 0 {
				t.Errorf("%s: owned staging or temporary files remain under the home: %v", shape.Name, rs)
			}
			if m := desktopkit.InstallMarkers(r.Profile.UserProfile); len(m) != 0 {
				t.Errorf("%s: installation state leaked into the user profile: %v", shape.Name, m)
			}
			s.Logf("%s: bootstrap.json = schema+home only, heavy state under the chosen home", shape.Name)

			// Next shape starts from a clean locator.
			if err := os.Remove(r.Profile.LocatorPath()); err != nil && !errors.Is(err, os.ErrNotExist) {
				t.Fatal(err)
			}
		}()
	}
}
