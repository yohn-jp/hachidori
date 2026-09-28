// Package setup materializes the pinned runtime and model under HACHIDORI_HOME.
package setup

import (
	"fmt"
	"runtime"

	"github.com/yohn-jp/hachidori/internal/home"
)

// RuntimeVersion is the version of the runtime bundle this binary materializes.
const RuntimeVersion = "0.1.0"

// Pinned private CPython (python-build-standalone, install_only).
const (
	pythonRelease = "20250902"
	pythonVersion = "3.12.11"
	pbsBase       = "https://github.com/astral-sh/python-build-standalone/releases/download/" + pythonRelease + "/"
)

type pythonDist struct {
	home.Artifact
	RelPath string // interpreter path inside the extracted archive
}

var pythonDists = map[string]pythonDist{
	"linux/amd64": {home.Artifact{
		URL:    pbsBase + "cpython-" + pythonVersion + "%2B" + pythonRelease + "-x86_64-unknown-linux-gnu-install_only.tar.gz",
		SHA256: "59c2827a4385741d04ea3971a3e6a845f951e96b2d168284534cc0d465391eeb"}, "python/bin/python3.12"},
	"windows/amd64": {home.Artifact{
		URL:    pbsBase + "cpython-" + pythonVersion + "%2B" + pythonRelease + "-x86_64-pc-windows-msvc-install_only.tar.gz",
		SHA256: "9ff8fddfd39b518d3902f204ceb9b6cef4213c6d78cf4bc4507c28079afece7c"}, "python/python.exe"},
}

// Flavors select the PyTorch build. CUDA 12.8 wheels support RTX 20xx..50xx
// (Turing..Blackwell) and need an NVIDIA driver >= 570.
var flavors = map[string]struct{ Name, Torch, Index string }{
	"cuda": {"cu128", "torch==2.11.0+cu128", "https://download.pytorch.org/whl/cu128"},
	"cpu":  {"cpu", "torch==2.11.0+cpu", "https://download.pytorch.org/whl/cpu"},
}

// packages is the exact package set resolved for laya 0.3.21 on CPython 3.12.
// On Linux CUDA, torch additionally pulls its own exactly pinned NVIDIA wheels.
var packages = []string{
	"laya==0.3.21",
	"transformers==5.17.0",
	"tokenizers==0.23.2",
	"safetensors==0.8.0",
	"huggingface_hub==1.33.0",
	"hf-xet==1.6.0",
	"numpy==2.5.3",
	"annotated-doc==0.0.5",
	"anyio==4.15.1",
	"certifi==2026.7.22",
	"click==8.5.0",
	"filelock==4.0.5",
	"fsspec==2026.9.0",
	"h11==0.16.0",
	"httpcore==1.0.9",
	"httpx==0.28.1",
	"idna==3.20",
	"Jinja2==3.1.6",
	"markdown-it-py==4.2.0",
	"MarkupSafe==3.0.3",
	"mdurl==0.1.2",
	"mpmath==1.3.0",
	"networkx==3.7",
	"packaging==26.3",
	"Pygments==2.21.0",
	"PyYAML==6.0.3",
	"regex==2026.9.10",
	"rich==15.0.0",
	"setuptools==81.0.0",
	"shellingham==1.5.4",
	"sympy==1.14.0",
	"tqdm==4.70.1",
	"typer==0.27.2",
	"typing_extensions==4.16.0",
	`colorama==0.4.6; sys_platform == "win32"`,
}

// Model is the pinned Laya checkpoint (English, ModernBERT-large) at the
// revision Laya itself lists as reviewed in laya.revisions.PINNED_REVISIONS.
var Model = home.ModelManifest{
	Repo:     "convaiinnovations/laya",
	Revision: "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
	Files: map[string]string{
		"rl_agent_config.json":            "ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd",
		"model.safetensors":               "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
		"encoder/config.json":             "bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979",
		"tokenizer/tokenizer.json":        "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
		"tokenizer/tokenizer_config.json": "50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12",
	},
}

// ModelDirName is the activation name of the pinned model.
func ModelDirName() string { return "convaiinnovations--laya/" + Model.Revision }

// RuntimeName is the runtime directory name for a device.
func RuntimeName(device string) (string, error) {
	f, ok := flavors[device]
	if !ok {
		return "", fmt.Errorf("device must be cuda or cpu, got %q", device)
	}
	return RuntimeVersion + "-" + f.Name, nil
}

func platform() string { return runtime.GOOS + "/" + runtime.GOARCH }
