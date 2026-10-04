package optimize

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// Build checks the cuda optimizer runtime with the accelerator probe after it
// is materialized and before anything is loaded: only the pinned CUDA torch with
// a usable device passes, everything else is a typed *CUDAUnavailableError and
// the build stops (there is no cpu retry).
func TestRequireCUDARuntime(t *testing.T) {
	spec, err := setup.DesiredOptimizer("cuda")
	if err != nil {
		t.Fatal(err)
	}
	rt := setup.OptimizerRuntime{ID: spec.ID(), Python: "/py", Manifest: home.RuntimeManifest{Spec: spec}}
	good := setup.AcceleratorFacts{Torch: spec.Torch, TorchCUDA: "12.8", CUDAAvailable: true, DeviceCount: 1, DeviceName: "NVIDIA GeForce RTX 3060", VRAMTotal: 12 << 30, VRAMFree: 11 << 30}
	probe := func(f setup.AcceleratorFacts, err error) func(context.Context, home.Home, string) (setup.AcceleratorFacts, error) {
		return func(_ context.Context, _ home.Home, python string) (setup.AcceleratorFacts, error) {
			if python != "/py" {
				t.Errorf("probed %q, not the optimizer runtime's interpreter", python)
			}
			return f, err
		}
	}

	var log strings.Builder
	if err := requireCUDARuntime(context.Background(), home.Home{}, rt, probe(good, nil), &log); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"NVIDIA GeForce RTX 3060", "CUDA 12.8", "torch " + spec.Torch, "12.0 GiB"} {
		if !strings.Contains(log.String(), want) {
			t.Errorf("log %q does not record %q", log.String(), want)
		}
	}

	bad := map[string]setup.AcceleratorFacts{
		"no cuda device": {Torch: spec.Torch, TorchCUDA: "12.8"},
		"cpu torch":      {Torch: "2.11.0+cpu", CUDAAvailable: true, DeviceName: "x", VRAMTotal: 1},
		"unpinned torch": {Torch: "2.12.0+cu128", TorchCUDA: "12.8", CUDAAvailable: true, DeviceName: "x", VRAMTotal: 1},
		"torch error":    {Error: "ImportError: libcudart"},
		"no device name": {Torch: spec.Torch, TorchCUDA: "12.8", CUDAAvailable: true, VRAMTotal: 1},
		"no vram":        {Torch: spec.Torch, TorchCUDA: "12.8", CUDAAvailable: true, DeviceName: "x"},
	}
	for name, f := range bad {
		var u *CUDAUnavailableError
		err := requireCUDARuntime(context.Background(), home.Home{}, rt, probe(f, nil), io.Discard)
		if !errors.As(err, &u) || u.Runtime != spec.ID() || !strings.Contains(err.Error(), "not continued on the cpu") {
			t.Errorf("%s: err = %v", name, err)
		}
	}
	var u *CUDAUnavailableError
	if err := requireCUDARuntime(context.Background(), home.Home{}, rt, probe(setup.AcceleratorFacts{}, errors.New("probe died")), io.Discard); !errors.As(err, &u) {
		t.Fatalf("an unobservable device was accepted: %v", err)
	}
}
