package fixture

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// The fixture must be a home the real executable accepts: the same resolution
// the executable performs succeeds and names the fixture's runtime and model.
func TestMaterializeIsAnActivatedHomeTheRuntimeResolves(t *testing.T) {
	root := t.TempDir()
	info, err := Materialize(root)
	if err != nil {
		t.Fatal(err)
	}
	cfg, rt, err := server.WorkerConfig(home.Home{Root: root}, nil)
	if err != nil {
		t.Fatalf("the executable would refuse the fixture home: %v", err)
	}
	if rt.Runtime != info.Runtime || rt.ModelID != setup.DefaultModel || rt.Device != "cpu" || rt.Model != info.ModelDir || rt.Home != root {
		t.Fatalf("runtime %+v does not describe the fixture %+v", rt, info)
	}
	if rt.Worker == nil || rt.Worker.SHA256 != setup.BuildWorker().SHA256 || rt.Worker.ABI != home.WorkerABIServing {
		t.Fatalf("worker build %+v", rt.Worker)
	}
	if cfg.Preflight == nil || cfg.Preflight() != nil {
		t.Fatal("the fixture runtime must pass the launch preflight, so the failure is the worker's own start")
	}
	if fi, err := os.Stat(cfg.Python); err != nil || fi.Mode()&0o111 != 0 {
		t.Fatalf("the interpreter must exist but not be executable: %v %v", fi, err)
	}
}

func TestPlantPlacesEveryCanaryAtItsSource(t *testing.T) {
	root, profile := t.TempDir(), t.TempDir()
	info, err := Materialize(root)
	if err != nil {
		t.Fatal(err)
	}
	c, err := Plant(info, profile)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, can := range c.List {
		if len(can.Needle) < 16 || can.Label == "" || can.Where == "" {
			t.Errorf("weak canary %+v", can)
		}
		if seen[can.Needle] {
			t.Errorf("duplicate canary %q", can.Needle)
		}
		seen[can.Needle] = true
	}
	for _, s := range c.Sources {
		data, err := os.ReadFile(s.Path)
		if err != nil || !strings.Contains(string(data), s.Needle) {
			t.Errorf("%s does not hold its canary: %v", s.Path, err)
		}
	}
	if len(c.Env) != 4 {
		t.Errorf("environment canaries: %v", c.Env)
	}
	for _, rel := range []string{".ssh/id_ed25519", ".ssh/known_hosts"} {
		if _, err := os.Stat(filepath.Join(profile, filepath.FromSlash(rel))); err != nil {
			t.Error(err)
		}
	}
	if len(c.List) < 15 {
		t.Errorf("only %d canaries planted", len(c.List))
	}
	// A different run draws different canaries.
	c2, _ := Plant(info, t.TempDir())
	if c2.List[0].Needle == c.List[0].Needle {
		t.Error("canaries are not unique per run")
	}
}

func TestEnvironmentInheritsNothingButEssentialsAndCanaries(t *testing.T) {
	t.Setenv("HF_TOKEN", "must-not-leak")
	t.Setenv("SECRET_FROM_THE_HOST", "must-not-leak")
	c := Canaries{Env: []string{"HF_TOKEN=planted"}}
	env := Environment(`C:\scratch\profile`, `C:\scratch\tmp`, c)
	joined := strings.Join(env, "\n")
	if strings.Contains(joined, "must-not-leak") || strings.Contains(joined, "SECRET_FROM_THE_HOST") {
		t.Fatalf("the host environment leaked into the run: %v", env)
	}
	for _, want := range []string{"USERPROFILE=", "HOME=", "TEMP=", "HF_TOKEN=planted"} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %s", want)
		}
	}
}
