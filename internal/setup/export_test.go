package setup

import (
	"io"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
)

// MaterializeFake runs the real setup reconciliation against the fake uv and
// artifact server, for tests outside this package.
func MaterializeFake(t *testing.T, device string) home.Home {
	return MaterializeFakeModel(t, device, "")
}

// MaterializeFakeModel is MaterializeFake selecting a catalog model.
func MaterializeFakeModel(t *testing.T, device, model string) home.Home {
	h, _ := MaterializeFakeCounting(t, device, model)
	return h
}

// MaterializeFakeCounting is MaterializeFakeModel plus a counter of all
// artifact server requests (uv and model downloads) made after setup.
func MaterializeFakeCounting(t *testing.T, device, model string) (home.Home, func() int) {
	f := newFixture(t)
	f.mustRunModel(device, model)
	total := func() int {
		f.mu.Lock()
		defer f.mu.Unlock()
		n := 0
		for _, c := range f.hits {
			n += c
		}
		return n
	}
	base := total()
	return f.H, func() int { return total() - base }
}

// TunedModel is the ID of the fake catalog's second (non-default) model.
const TunedModel = tunedModel

// MaterializeFakeLegacy is MaterializeFake followed by legacyize: a home whose
// active runtime is a pre-declarative runtime that still has its worker and
// interpreter on disk.
func MaterializeFakeLegacy(t *testing.T, device string) home.Home {
	f := newFixture(t)
	f.mustRun(device)
	f.legacyize(device)
	return f.H
}

// MaterializeFakeOlderWorker is MaterializeFake followed by olderWorker: a home
// whose active runtime is consistent but carries an older build's worker script.
func MaterializeFakeOlderWorker(t *testing.T, device string) (home.Home, string) {
	f := newFixture(t)
	f.mustRun(device)
	name := f.olderWorker(device)
	return f.H, name
}

// MaterializeFakeResidents materializes and activates the default catalog
// model for device and, without activating it, the fake catalog's second
// model: a home from which both models can be launched as residents. It
// returns the home and the two catalog IDs (default first).
func MaterializeFakeResidents(t *testing.T, device string) (h home.Home, def, other string) {
	f := newFixture(t)
	f.mustRun(device)
	if err := Materialize(f.H, device, tunedModel, io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	return f.H, DefaultModel, tunedModel
}
