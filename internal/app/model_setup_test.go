package app

import (
	"errors"
	"io"
	"slices"
	"sync"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// Model-aware setup -> application controller: the controller passes the
// caller's catalog model ID (empty means the catalog default) straight to the
// setup authority, reports the setup phases the authority entered, and never
// runs setup beside a running worker (activation cannot change under it).
func TestControllerSetupModelSelection(t *testing.T) {
	var mu sync.Mutex
	var gotModels []string
	record := func(root, device, model string, log io.Writer, onPhase func(setup.Phase)) error {
		mu.Lock()
		gotModels = append(gotModels, model)
		mu.Unlock()
		onPhase(setup.PhasePreparing)
		if _, err := setup.LookupModel(model); err != nil {
			return err
		}
		onPhase(setup.PhaseModel)
		onPhase(setup.PhaseActivation)
		return nil
	}

	e := newEnv(t, "/h", false)
	e.c.cfg.Setup = record
	e.installed.Store(true)

	// Default and every explicit catalog ID are accepted.
	ids := []string{""}
	for _, m := range setup.Models {
		ids = append(ids, m.ID)
	}
	for _, id := range ids {
		if err := e.c.Setup(SetupParams{Device: "cpu", Model: id}); err != nil {
			t.Fatalf("model %q rejected: %v", id, err)
		}
		s := waitIdle(t, e.c)
		if s.Last == nil || s.Last.Failure != nil || s.Last.Model != id ||
			!slices.Equal(s.Last.Phases, []string{"preparing", "model", "activation"}) {
			t.Fatalf("model %q: last %+v", id, s.Last)
		}
	}
	if !slices.Equal(gotModels, ids) {
		t.Fatalf("setup authority saw models %v, want %v", gotModels, ids)
	}
	if _, err := setup.LookupModel(""); err != nil {
		t.Fatalf("empty model must resolve to the catalog default %s: %v", setup.DefaultModel, err)
	}

	// A model outside the catalog fails as a setup failure after the real
	// phases that were entered; nothing is invented and the failure is kept.
	e.c.cfg.Setup = func(root, device, model string, log io.Writer, onPhase func(setup.Phase)) error {
		return setup.RunObserved(home.Home{Root: t.TempDir()}, device, model, log, onPhase)
	}
	if err := e.c.Setup(SetupParams{Device: "cpu", Model: "not-in-catalog"}); err != nil {
		t.Fatal(err)
	}
	s := waitIdle(t, e.c)
	if s.Last == nil || s.Last.Failure == nil || s.Last.Failure.Source != SourceSetup ||
		!slices.Equal(s.Last.Phases, []string{"preparing"}) {
		t.Fatalf("unsupported model: last %+v", s.Last)
	}
}

// Setup is refused while the worker runs, for default and explicit models,
// so the activation cannot change concurrently with a running worker.
func TestControllerSetupRefusedWhileWorkerRuns(t *testing.T) {
	e := newEnv(t, "/h", true)
	if err := e.c.Start(); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"", setup.DefaultModel} {
		if err := e.c.Setup(SetupParams{Device: "cpu", Model: id}); !errors.Is(err, ErrRuntimeBusy) {
			t.Fatalf("model %q under a running worker: %v", id, err)
		}
	}
	if n := e.setup.calls.Load(); n != 0 {
		t.Fatalf("setup ran %d times beside a running worker", n)
	}
}
