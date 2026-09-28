package doctor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
)

// A runtime whose manifest does not carry a Runtime Spec identity matching
// its directory (for example one created by the former pip-based setup) is
// reported as runtime_invalid.
func TestDoctorRejectsRuntimeWithoutSpecIdentity(t *testing.T) {
	h := home.Home{Root: t.TempDir()}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	os.MkdirAll(h.Path("runtime", "0.1.0-cpu"), 0o755)
	os.MkdirAll(h.Path("models", "m", "r"), 0o755)
	os.WriteFile(h.Path("runtime", "0.1.0-cpu", "manifest.json"), []byte(`{"version":"0.1.0","python":"python/bin/python3.12"}`), 0o644)
	os.WriteFile(filepath.Join(h.Path("models", "m", "r"), "hachidori-model.json"), []byte(`{}`), 0o644)
	home.WriteJSON(h.Path("state", "active-runtime.json"), home.Active{Runtime: "0.1.0-cpu", Model: "m/r", Device: "cpu"})

	var out strings.Builder
	if Run(h.Root, &out) {
		t.Fatal("doctor passed")
	}
	if !strings.Contains(out.String(), "FAIL runtime") || !strings.Contains(out.String(), "class: "+RuntimeInvalid) ||
		!strings.Contains(out.String(), "does not match its Runtime Spec") {
		t.Fatalf("output:\n%s", out.String())
	}
}
