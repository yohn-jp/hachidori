package setup

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
)

func TestClefKernelRuntimeContract(t *testing.T) {
	for _, plat := range []string{"linux/amd64", "windows/amd64"} {
		cpu, err := desiredFor("cpu", plat)
		if err != nil {
			t.Fatal(err)
		}
		cuda, err := desiredFor("cuda", plat)
		if err != nil {
			t.Fatal(err)
		}
		if len(clefKernelDistributions(cpu)) != 0 {
			t.Fatal("CPU runtime requires Clef CUDA kernels")
		}
		want := []string(nil)
		if plat == "windows/amd64" {
			want = []string{"fla-core==0.5.2", "triton-windows==3.6.0.post26"}
		}
		if !slices.Equal(clefKernelDistributions(cuda), want) {
			t.Fatalf("wrong Clef kernel pins for %s: %v", plat, clefKernelDistributions(cuda))
		}
		if cuda.Project != digest(specFile("pyproject.toml")) || cuda.Lock != digest(specFile("uv.lock")) {
			t.Fatal("kernel materialization/dispatch is outside runtime identity")
		}
		for _, field := range []string{"project", "lock", "worker_abi"} {
			old := cuda
			switch field {
			case "project":
				old.Project = strings.Repeat("0", 64)
			case "lock":
				old.Lock = strings.Repeat("0", 64)
			case "worker_abi":
				old.WorkerABI = "hachidori.worker-runtime/2"
			}
			if old.ID() == cuda.ID() {
				t.Fatalf("%s changes do not change identity", field)
			}
		}
	}
	project, lock := string(specFile("pyproject.toml")), string(specFile("uv.lock"))
	for _, pin := range []string{"fla-core==0.5.2; sys_platform == 'win32'", "triton-windows==3.6.0.post26; sys_platform == 'win32'"} {
		if !strings.Contains(project, pin) {
			t.Fatalf("project lacks %s", pin)
		}
	}
	if strings.Contains(project, "triton==3.6.0") {
		t.Fatal("project carries unsupported Linux Clef Triton runtime")
	}
	for _, pkg := range []string{"name = \"fla-core\"\nversion = \"0.5.2\"", "name = \"triton\"\nversion = \"3.6.0\"", "name = \"triton-windows\"\nversion = \"3.6.0.post26\"", "triton_windows-3.6.0.post26-cp312-cp312-win_amd64.whl"} {
		if !strings.Contains(lock, pkg) {
			t.Fatalf("lock lacks %s", pkg)
		}
	}
}
func TestClefKernelRuntimeVerification(t *testing.T) {
	f := newFixture(t)
	f.mustRun("cuda")
	spec, _ := Desired("cuda")
	dir := f.H.Path("runtime", spec.ID())
	installedPath := filepath.Join(dir, "env", "installed.json")
	var installed []string
	if err := home.ReadJSON(installedPath, &installed); err != nil {
		t.Fatal(err)
	}
	for _, required := range clefKernelDistributions(spec) {
		without := slices.DeleteFunc(slices.Clone(installed), func(pin string) bool { return pin == required })
		if err := home.WriteJSON(installedPath, without); err != nil {
			t.Fatal(err)
		}
		if _, err := verifyRuntime(f.H, dir, spec); err == nil || !strings.Contains(err.Error(), required+" not installed") {
			t.Fatalf("missing %s accepted: %v", required, err)
		}
	}
}
