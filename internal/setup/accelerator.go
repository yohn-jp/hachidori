package setup

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/subprocess"
)

// AcceleratorFacts is what the private runtime's torch reports about CUDA.
// It is observed by running the private interpreter, isolated and offline;
// nothing is inferred from the runtime's name or from the host.
type AcceleratorFacts struct {
	Python        string `json:"python,omitempty"`
	Torch         string `json:"torch,omitempty"`
	TorchCUDA     string `json:"torch_cuda,omitempty"`
	CUDAAvailable bool   `json:"cuda_available"`
	DeviceCount   int    `json:"device_count,omitempty"`
	DeviceName    string `json:"device_name,omitempty"`
	Capability    []int  `json:"capability,omitempty"`
	VRAMFree      uint64 `json:"vram_free_bytes,omitempty"`
	VRAMTotal     uint64 `json:"vram_total_bytes,omitempty"`
	// Error is torch failing to import or to report; the facts above are then
	// absent, not zero.
	Error string `json:"error,omitempty"`
}

// acceleratorProbeCode reports the torch/CUDA facts of the interpreter.
const acceleratorProbeCode = `import json, sys
out = {"python": "%d.%d.%d" % sys.version_info[:3]}
try:
    import torch
    out["torch"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        i = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(i)
        out.update({"device_count": torch.cuda.device_count(), "device_name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i)), "vram_free_bytes": int(free), "vram_total_bytes": int(total)})
except Exception as e:
    out["error"] = "%s: %s" % (type(e).__name__, e)
print(json.dumps(out))`

// acceleratorProbeTimeout bounds the probe: importing torch is slow on a cold
// disk but never unbounded.
const acceleratorProbeTimeout = 2 * time.Minute

// ProbeAccelerator runs the torch/CUDA probe with the runtime's private
// interpreter. It never starts a worker and never loads a model.
func ProbeAccelerator(ctx context.Context, h home.Home, python string) (AcceleratorFacts, error) {
	ctx, cancel := context.WithTimeout(ctx, acceleratorProbeTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, python, "-I", "-X", "utf8", "-c", acceleratorProbeCode)
	subprocess.Configure(cmd)
	cmd.Env = h.Env(filepath.Dir(python), true)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return AcceleratorFacts{}, fmt.Errorf("accelerator probe: %v: %s", err, strings.TrimSpace(tailString(stderr.String(), 400)))
	}
	var f AcceleratorFacts
	if err := json.Unmarshal(out, &f); err != nil {
		return AcceleratorFacts{}, fmt.Errorf("accelerator probe: unexpected output %q", tailString(strings.TrimSpace(string(out)), 200))
	}
	return f, nil
}

func tailString(s string, n int) string {
	if len(s) > n {
		return "..." + s[len(s)-n:]
	}
	return s
}
