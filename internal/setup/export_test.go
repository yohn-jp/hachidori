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

// MaterializeFakeStale is MaterializeFake followed by staleRuntime: a home
// whose active runtime is consistent but is not the dependency runtime this
// build requires.
func MaterializeFakeStale(t *testing.T, device string) (home.Home, string) {
	f := newFixture(t)
	f.mustRun(device)
	name := f.staleRuntime(device)
	return f.H, name
}

// MaterializeFakeSchemaV1 is MaterializeFake followed by schemaV1Runtime: a
// home whose active runtime was written under the identity scheme that
// included the worker digest.
func MaterializeFakeSchemaV1(t *testing.T, device string) (home.Home, string) {
	f := newFixture(t)
	f.mustRun(device)
	name, _ := f.schemaV1Runtime(device)
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

// MaterializeFakeClef materializes the runtime and a System One fixture model
// under the real Clef-Flash catalog ID through the real setup path, activating
// it, and returns the home and the catalog entry. Variants of it can then be
// built with the fake optimizer.
func MaterializeFakeClef(t *testing.T, device string) (home.Home, home.ModelManifest) {
	f := newFixture(t)
	m := f.addClef()
	f.mustRunModel(device, ClefFlash)
	return f.H, m
}

// FileSHA256Bytes is the digest of b, for tests outside this package.
func FileSHA256Bytes(b []byte) (string, error) { return digest(b), nil }

// Fake is the fake uv, fake private python and artifact server of this
// package's tests, for tests outside it. Every method fails the test on an
// unexpected error unless it returns one.
type Fake struct {
	f *fixture
	H home.Home
}

// NewFake returns an empty fake home.
func NewFake(t *testing.T) *Fake {
	f := newFixture(t)
	return &Fake{f: f, H: f.H}
}

// Run is the real setup (materialize, verify, activate) of model on device.
func (x *Fake) Run(device, model string) error {
	_, err := x.f.runModel(device, model)
	return err
}

// AddClef adds the System One fixture model to the catalog.
func (x *Fake) AddClef() home.ModelManifest { return x.f.addClef() }

// Stale rewrites the active runtime as another dependency environment than
// this build requires and returns its directory name.
func (x *Fake) Stale(device string) string { return x.f.staleRuntime(device) }

// SchemaV1 rewrites the active runtime as one written under the identity
// scheme that included the worker digest and returns its directory name.
func (x *Fake) SchemaV1(device string) string { n, _ := x.f.schemaV1Runtime(device); return n }

// FailUV makes the fake uv fail the named subcommand ("" clears it).
func (x *Fake) FailUV(sub string) { x.f.control(fakeControl{Fail: sub}) }

// UVCalls is the number of uv invocations so far.
func (x *Fake) UVCalls() int { return len(x.f.calls()) }

// Requests is the number of artifact server requests so far (uv release and
// model files).
func (x *Fake) Requests() int {
	x.f.mu.Lock()
	defer x.f.mu.Unlock()
	n := 0
	for _, c := range x.f.hits {
		n += c
	}
	return n
}

// ActivateDir points the activation record at an existing runtime directory.
func (x *Fake) ActivateDir(name string) { x.f.activateRuntimeDir(name) }
