package desktopkit

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// FixtureDevice and FixtureModel name the activation the installed-home
// fixture records: the CPU runtime and the catalog's default model.
const (
	FixtureDevice = "cpu"
	FixtureModel  = setup.DefaultModel
)

// pythonRel is where the runtime keeps its interpreter, exactly as setup
// records it in the runtime manifest.
func pythonRel() string {
	if runtime.GOOS == "windows" {
		return "env/Scripts/python.exe"
	}
	return "env/bin/python"
}

// Installed describes a fixture home.
type Installed struct {
	Home    home.Home
	Runtime string // runtime directory name (identity)
	Python  string // path of the stand-in interpreter
}

// InstallFixture writes, under root, a Hachidori home that the production
// code accepts as installed: an activation record, a runtime manifest whose
// identity is the runtime this build requires, and the catalog model's
// manifest. Nothing is downloaded and no model weights are written.
//
// The runtime's interpreter is a copy of the running test binary, which
// behaves as a protocol-speaking worker stand-in (see MaybeRunStub). The
// fixture therefore certifies process topology (which process owns which
// worker and endpoint), never inference quality.
func InstallFixture(root string) (Installed, error) {
	h := home.Home{Root: root}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return Installed{}, err
	}
	if err := h.Ensure(); err != nil {
		return Installed{}, fmt.Errorf("prepare home: %w", err)
	}
	spec, err := setup.Desired(FixtureDevice)
	if err != nil {
		return Installed{}, err
	}
	id := spec.ID()
	rtDir := h.Path("runtime", id)
	py := filepath.Join(rtDir, filepath.FromSlash(pythonRel()))
	if err := os.MkdirAll(filepath.Dir(py), 0o755); err != nil {
		return Installed{}, err
	}
	self, err := os.Executable()
	if err != nil {
		return Installed{}, err
	}
	if err := copyFile(self, py); err != nil {
		return Installed{}, fmt.Errorf("stage the stand-in interpreter: %w", err)
	}
	manifest := home.RuntimeManifest{Identity: id, Spec: spec, PythonVersion: spec.Python, PythonRelPath: pythonRel(), Installed: []string{}}
	if err := home.WriteJSON(filepath.Join(rtDir, "manifest.json"), manifest); err != nil {
		return Installed{}, err
	}
	model, err := setup.LookupModel(FixtureModel)
	if err != nil {
		return Installed{}, err
	}
	active := home.Active{Runtime: id, ModelID: model.ID, Model: setup.ModelDirName(model), Device: FixtureDevice}
	modelDir := h.ModelDir(active)
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		return Installed{}, err
	}
	if err := home.WriteJSON(filepath.Join(modelDir, "hachidori-model.json"), model); err != nil {
		return Installed{}, err
	}
	// The activation record is written last: its presence marks a complete home.
	if err := home.WriteJSON(h.Path("state", "active-runtime.json"), active); err != nil {
		return Installed{}, err
	}
	return Installed{Home: h, Runtime: id, Python: py}, nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer func() { _ = in.Close() }()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		_ = out.Close()
		return err
	}
	return out.Close()
}
