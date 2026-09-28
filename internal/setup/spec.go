// Package setup reconciles HACHIDORI_HOME with the desired Runtime Spec and
// the pinned model: it bootstraps Hachidori's private uv, lets uv materialize
// the locked Python environment into staging, verifies it, and activates it.
package setup

import (
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"fmt"
	"runtime"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/worker/py"
)

// SpecSchema versions the Runtime Spec contract and its identity derivation.
const SpecSchema = "hachidori.runtime-spec/1"

// Runtime environment intent. The authoritative package set is the uv
// project in runtimespec/ (pyproject.toml + uv.lock); these values are what
// verification requires of the materialized environment.
const (
	pythonVersion   = "3.12.11"
	providerName    = "laya"
	providerVersion = "0.3.21"
	torchVersion    = "2.11.0"
)

// flavors maps a device to the uv extra (and torch local version) that
// selects the PyTorch build in runtimespec/pyproject.toml.
var flavors = map[string]string{
	"cuda": "cu128",
	"cpu":  "cpu",
}

//go:embed runtimespec/pyproject.toml runtimespec/uv.lock
var specFS embed.FS

func specFile(name string) []byte {
	b, err := specFS.ReadFile("runtimespec/" + name)
	if err != nil {
		panic(err)
	}
	return b
}

// Private uv: the exact release Hachidori bootstraps under
// HACHIDORI_HOME/tools/uv/<version>/. Both the published archive digest and the
// digest of the extracted executable are pinned; the executable is verified
// before every invocation. uv from PATH is never used.
var uvVersion = "0.12.19"

type uvArtifact struct {
	URL          string // release archive
	SHA256       string // archive digest (as published next to the release asset)
	Member       string // executable path inside the archive
	BinarySHA256 string // digest of the extracted executable
}

const uvBase = "https://github.com/astral-sh/uv/releases/download/"

var uvArtifacts = map[string]uvArtifact{
	"linux/amd64": {
		URL:          uvBase + "0.12.19/uv-x86_64-unknown-linux-gnu.tar.gz",
		SHA256:       "23bf5552d220e0842b65c862097b2ebaeba0064b74eda5e565e77fd25969d8c8",
		Member:       "uv-x86_64-unknown-linux-gnu/uv",
		BinarySHA256: "242e462a63f5a3c0421d68557006193ecbfb61321cba0fe8542213ac62d92563",
	},
	"windows/amd64": {
		URL:          uvBase + "0.12.19/uv-x86_64-pc-windows-msvc.zip",
		SHA256:       "6dbb02d79e419522f1c500f0adb1cddcff0cda7d59b0d66ea7f5e3b4a1b2f5f0",
		Member:       "uv.exe",
		BinarySHA256: "f94eddb81f3addca6ef8f2361a70c3edde31adcbd1000a55fc6674306ae0b1e7",
	},
}

// uvFor selects the pinned uv artifact for a platform.
func uvFor(plat string) (uvArtifact, error) {
	a, ok := uvArtifacts[plat]
	if !ok {
		return uvArtifact{}, fmt.Errorf("no pinned uv for %s (supported: linux/amd64, windows/amd64)", plat)
	}
	return a, nil
}

// Desired is the Runtime Spec this executable materializes for a device on
// the current platform.
func Desired(device string) (home.RuntimeSpec, error) {
	return desiredFor(device, platform())
}

func desiredFor(device, plat string) (home.RuntimeSpec, error) {
	flavor, ok := flavors[device]
	if !ok {
		return home.RuntimeSpec{}, fmt.Errorf("device must be cuda or cpu, got %q", device)
	}
	uv, err := uvFor(plat)
	if err != nil {
		return home.RuntimeSpec{}, err
	}
	return home.RuntimeSpec{
		Schema:   SpecSchema,
		Platform: plat,
		Python:   pythonVersion,
		Provider: providerName + "==" + providerVersion,
		Torch:    torchVersion + "+" + flavor,
		Flavor:   flavor,
		UV:       uvVersion,
		UVSHA256: uv.BinarySHA256,
		Project:  digest(specFile("pyproject.toml")),
		Lock:     digest(specFile("uv.lock")),
		Worker:   digest(py.Script),
	}, nil
}

// RuntimeName is the runtime identity (directory name under runtime/) for a device.
func RuntimeName(device string) (string, error) {
	s, err := Desired(device)
	if err != nil {
		return "", err
	}
	return s.ID(), nil
}

// Model is the pinned Laya checkpoint (English, ModernBERT-large) at the
// revision Laya itself lists as reviewed in laya.revisions.PINNED_REVISIONS.
// It is materialized independently of the Python runtime and is not part of
// the runtime identity.
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

var modelBaseURL = "https://huggingface.co/"

// ModelDirName is the activation name of the pinned model.
func ModelDirName() string { return "convaiinnovations--laya/" + Model.Revision }

func platform() string { return runtime.GOOS + "/" + runtime.GOARCH }

func digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
