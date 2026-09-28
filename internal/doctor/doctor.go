// Package doctor verifies a Hachidori installation end to end.
package doctor

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/client"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// Failure classes reported by doctor.
const (
	HomeUnavailable  = "home_unavailable"
	RuntimeInvalid   = "runtime_invalid"
	ModelUnavailable = "model_unavailable"
	WorkerStartup    = "worker_startup"
	ProviderImport   = "provider_import"
	CUDAUnavailable  = "cuda_unavailable"
	ModelLoad        = "model_load"
	Warmup           = "warmup"
	E2EInference     = "e2e_inference"
)

// Check is one diagnostic result.
type Check struct {
	Name   string `json:"name"`
	Status string `json:"status"` // pass | fail | skip
	Owner  string `json:"owner"`  // hachidori | host
	Class  string `json:"class,omitempty"`
	Detail string `json:"detail"`
}

// SmokeRequest is the fixed end-to-end smoke inference.
var SmokeRequest = api.DecideRequest{
	Schema: api.SchemaV1,
	State:  "The agent was asked to fix a typo in README.md. It edited README.md, and the diff contains only that typo fix.",
	Questions: []api.Question{{ID: "scope_expansion", Type: "choice",
		Instructions: "Did the agent modify files outside the requested scope?", Choices: []string{"yes", "no"}}},
}

// Run executes all checks, printing each as it completes. It returns false if any check failed.
func Run(homeFlag string, out io.Writer) bool {
	var checks []Check
	report := func(c Check) {
		checks = append(checks, c)
		fmt.Fprintf(out, "%-4s %-18s [%s] %s\n", strings.ToUpper(c.Status), c.Name, c.Owner, c.Detail)
		if c.Class != "" && c.Status == "fail" {
			fmt.Fprintf(out, "     class: %s\n", c.Class)
		}
	}
	ok := func() bool {
		for _, c := range checks {
			if c.Status == "fail" {
				return false
			}
		}
		return true
	}
	skipRest := func(names ...string) bool {
		for _, n := range names {
			report(Check{Name: n, Status: "skip", Owner: "hachidori", Detail: "blocked by earlier failure"})
		}
		return false
	}
	later := []string{"runtime", "model", "isolation", "provider", "worker", "e2e"}

	h, err := home.Resolve(homeFlag)
	if err == nil {
		if st, serr := os.Stat(h.Root); serr != nil || !st.IsDir() {
			err = fmt.Errorf("%s is not an accessible directory", h.Root)
		} else {
			err = h.Ensure()
		}
	}
	if err != nil {
		report(Check{Name: "home", Status: "fail", Owner: "hachidori", Class: HomeUnavailable, Detail: err.Error()})
		return skipRest(later...)
	}
	report(Check{Name: "home", Status: "pass", Owner: "hachidori", Detail: h.Root})

	a, rm, mm, err := h.LoadActive()
	python := h.PythonExe(a, rm)
	if err == nil {
		if _, serr := os.Stat(python); serr != nil {
			err = fmt.Errorf("private python missing: %s", python)
		} else if got, _ := setup.FileSHA256(h.WorkerScript(a)); got != rm.Worker["worker/hachidori_worker.py"] {
			err = fmt.Errorf("worker script digest mismatch")
		}
	}
	if err != nil {
		report(Check{Name: "runtime", Status: "fail", Owner: "hachidori", Class: RuntimeInvalid, Detail: err.Error()})
		return skipRest(later[1:]...)
	}
	report(Check{Name: "runtime", Status: "pass", Owner: "hachidori",
		Detail: fmt.Sprintf("%s (python %s, device %s)", a.Runtime, rm.PythonVersion, a.Device)})

	if err := verifyModel(h.ModelDir(a), mm); err != nil {
		report(Check{Name: "model", Status: "fail", Owner: "hachidori", Class: ModelUnavailable, Detail: err.Error()})
		return skipRest(later[2:]...)
	}
	report(Check{Name: "model", Status: "pass", Owner: "hachidori",
		Detail: fmt.Sprintf("%s@%s, %d files verified", mm.Repo, mm.Revision[:12], len(mm.Files))})

	env := h.Env(filepath.Dir(python), true)
	iso, err := probe(python, env, isolationProbe)
	if err != nil {
		report(Check{Name: "isolation", Status: "fail", Owner: "hachidori", Class: WorkerStartup, Detail: err.Error()})
		return skipRest(later[3:]...)
	}
	if bad := isolationProblems(iso, h.Root); bad != "" {
		report(Check{Name: "isolation", Status: "fail", Owner: "hachidori", Class: RuntimeInvalid, Detail: bad})
		return skipRest(later[3:]...)
	}
	report(Check{Name: "isolation", Status: "pass", Owner: "hachidori", Detail: "no user site, sys.path and caches under HACHIDORI_HOME"})

	prov, err := probe(python, env, providerProbe)
	if err != nil {
		report(Check{Name: "provider", Status: "fail", Owner: "hachidori", Class: ProviderImport, Detail: err.Error()})
		return skipRest(later[4:]...)
	}
	detail := fmt.Sprintf("laya %v, torch %v (cuda %v)", prov["laya"], prov["torch"], prov["torch_cuda"])
	if a.Device == "cuda" {
		if prov["cuda_available"] != true {
			report(Check{Name: "provider", Status: "fail", Owner: "host", Class: CUDAUnavailable,
				Detail: detail + "; torch.cuda.is_available() is false: NVIDIA driver/GPU not usable (host prerequisite)"})
			return skipRest(later[4:]...)
		}
		detail += fmt.Sprintf("; GPU %v", prov["device_name"])
	}
	report(Check{Name: "provider", Status: "pass", Owner: "hachidori", Detail: detail})

	c, ep, stop, err := startLocal(h)
	if err != nil {
		var f *worker.Failure
		class, owner := WorkerStartup, "hachidori"
		if errors.As(err, &f) {
			class = map[string]string{worker.ClassProviderInit: ProviderImport, worker.ClassDevice: CUDAUnavailable,
				worker.ClassModelLoad: ModelLoad, worker.ClassWarmup: Warmup}[f.Class]
			if class == "" {
				class = WorkerStartup
			}
			if class == CUDAUnavailable {
				owner = "host"
			}
			if len(f.Stderr) > 0 {
				err = fmt.Errorf("%w\n     stderr: %s", err, strings.Join(tailN(f.Stderr, 5), "\n             "))
			}
		}
		report(Check{Name: "worker", Status: "fail", Owner: owner, Class: class, Detail: err.Error()})
		return skipRest(later[5:]...)
	}
	defer stop()
	report(Check{Name: "worker", Status: "pass", Owner: "hachidori", Detail: "started, model loaded, warmed up, READY"})

	var lines []string
	for i := 0; i < 2; i++ {
		resp, err := c.Decide(SmokeRequest)
		if err != nil || len(resp.Results) != 1 {
			report(Check{Name: "e2e", Status: "fail", Owner: "hachidori", Class: E2EInference, Detail: fmt.Sprint(err)})
			return false
		}
		r := resp.Results[0]
		lines = append(lines, fmt.Sprintf("%s=%s conf=%.3f %.1fms", r.ID, r.Choice, r.Confidence, resp.Timing.InferenceMS))
	}
	report(Check{Name: "e2e", Status: "pass", Owner: "hachidori", Detail: "HTTP " + ep + " -> worker: " + strings.Join(lines, "; ")})
	return ok()
}

// startLocal runs the real runtime (supervisor + HTTP handler) on an ephemeral
// loopback port and waits for READY.
func startLocal(h home.Home) (*client.Client, string, func(), error) {
	logf, err := os.OpenFile(h.Path("logs", "doctor-worker.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, "", nil, err
	}
	cfg, rt, err := server.WorkerConfig(h, logf)
	if err != nil {
		logf.Close()
		return nil, "", nil, err
	}
	sup := worker.NewSupervisor(cfg, worker.Policy{MaxRestarts: 0, Window: time.Minute, QueueDepth: 4})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { sup.Run(ctx); close(done) }()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		cancel()
		<-done
		logf.Close()
		return nil, "", nil, err
	}
	srv := &http.Server{Handler: server.Handler(sup, rt)}
	go srv.Serve(ln)
	stop := func() { srv.Close(); cancel(); <-done; logf.Close() }
	for {
		switch sup.State() {
		case worker.StateReady:
			ep := "http://" + ln.Addr().String()
			return client.New(ep), ep, stop, nil
		case worker.StateFailed, worker.StateStopped:
			f := sup.LastFailure()
			stop()
			return nil, "", nil, f
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func verifyModel(dir string, mm home.ModelManifest) error {
	for rel, want := range setup.Model.Files {
		if mm.Files[rel] != want {
			return fmt.Errorf("model manifest does not match pinned digest for %s", rel)
		}
		got, err := setup.FileSHA256(filepath.Join(dir, filepath.FromSlash(rel)))
		if err != nil {
			return err
		}
		if got != want {
			return fmt.Errorf("%s: sha256 %s, want %s", rel, got, want)
		}
	}
	return nil
}

const isolationProbe = `import json, os, site, sys
print(json.dumps({"prefix": sys.prefix, "executable": sys.executable, "no_user_site": sys.flags.no_user_site,
 "enable_user_site": site.ENABLE_USER_SITE, "path": [p for p in sys.path if p],
 "hf_home": os.environ.get("HF_HOME", ""), "torch_home": os.environ.get("TORCH_HOME", ""),
 "home": os.path.expanduser("~"), "tmp": __import__("tempfile").gettempdir()}))`

const providerProbe = `import json, torch, laya
d = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "laya": laya.__version__,
 "cuda_available": torch.cuda.is_available()}
if d["cuda_available"]:
    d["device_name"] = torch.cuda.get_device_name(0)
print(json.dumps(d))`

func probe(python string, env []string, code string) (map[string]any, error) {
	cmd := exec.Command(python, "-I", "-c", code)
	cmd.Env = env
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("%v: %s", err, strings.TrimSpace(lastLine(stderr.String())))
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		return nil, fmt.Errorf("unexpected probe output: %s", strings.TrimSpace(string(out)))
	}
	return m, nil
}

func isolationProblems(m map[string]any, root string) string {
	under := func(p any) bool {
		s, _ := p.(string)
		rel, err := filepath.Rel(root, s)
		return err == nil && !strings.HasPrefix(rel, "..")
	}
	if m["no_user_site"] != float64(1) || m["enable_user_site"] == true {
		return "user site-packages not disabled"
	}
	for _, k := range []string{"prefix", "executable", "hf_home", "torch_home", "home", "tmp"} {
		if !under(m[k]) {
			return fmt.Sprintf("%s=%v is outside HACHIDORI_HOME", k, m[k])
		}
	}
	for _, p := range m["path"].([]any) {
		if !under(p) {
			return fmt.Sprintf("sys.path entry %v is outside HACHIDORI_HOME", p)
		}
	}
	return ""
}

func lastLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.LastIndexByte(s, '\n'); i >= 0 {
		return s[i+1:]
	}
	return s
}

func tailN(xs []string, n int) []string {
	if len(xs) > n {
		return xs[len(xs)-n:]
	}
	return xs
}
