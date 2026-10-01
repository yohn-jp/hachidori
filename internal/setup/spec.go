// Package setup reconciles HACHIDORI_HOME with the desired Runtime Spec and
// the pinned model: it bootstraps Hachidori's private uv, lets uv materialize
// the locked Python environment into staging, verifies it, and activates it.
package setup

import (
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"errors"
	"fmt"
	"runtime"
	"sort"
	"strings"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/worker/py"
)

// SpecSchema versions the Runtime Spec contract and its identity derivation.
const SpecSchema = "hachidori.runtime-spec/1"

// Runtime environment intent. The authoritative package set is the uv
// project in runtimespec/ (pyproject.toml + uv.lock); these values are what
// verification requires of the materialized environment.
const (
	pythonVersion = "3.12.11"
	torchVersion  = "2.11.0"
)

// Model providers: the packages that load a catalog model for the resident
// worker, in canonical order. One runtime carries all of them, so any
// catalog model can be activated on any materialized runtime; each catalog
// model names exactly one. The packages are pinned in runtimespec/.
const (
	providerLaya        = "laya"
	providerOpenDecider = "opendecider"

	layaVersion        = "0.3.21"
	openDeciderVersion = "0.3.0"
)

// providerPins is RuntimeSpec.Provider: the pinned providers as
// name==version, comma separated.
const providerPins = providerLaya + "==" + layaVersion + "," + providerOpenDecider + "==" + openDeciderVersion

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
		Provider: providerPins,
		Torch:    torchVersion + "+" + flavor,
		Flavor:   flavor,
		UV:       uvVersion,
		UVSHA256: uv.BinarySHA256,
		Project:  digest(specFile("pyproject.toml")),
		Lock:     digest(specFile("uv.lock")),
		Worker:   digest(py.Script),
	}, nil
}

// WorkerDigest is the digest of the worker script this build embeds and
// materializes into every runtime it creates.
func WorkerDigest() string { return digest(py.Script) }

// ErrWorkerContract marks an activated runtime whose worker script is not the
// one this build embeds.
var ErrWorkerContract = errors.New("runtime carries a different worker script")

// CheckWorkerContract is the launch-contract authority for an activated
// runtime. The worker script is half of the contract between Hachidori and its
// private Python process (its command line and its protocol); the other half
// is built into this executable. A runtime is immutable, so one materialized
// by an older build keeps its older script, which can be internally
// consistent (its manifest, identity and digests all agree) and still not
// understand the arguments this build passes. Starting it would only fail in
// the interpreter's argument parser, so it is refused here with the cause and
// the recovery instead.
func CheckWorkerContract(a home.Active, rm home.RuntimeManifest) error {
	if rm.Spec.Worker == WorkerDigest() {
		return nil
	}
	return fmt.Errorf("%w: runtime %s was materialized by an older Hachidori (worker %s, this build %s) and cannot be started by this build. "+
		"Materialize the current runtime and activate it: Settings, Models & runtimes, choose %s and your model, Materialize, Activate, then Restart; "+
		"or run `hachidori setup --device %s`. Installed models are reused",
		ErrWorkerContract, a.Runtime, shortDigest(rm.Spec.Worker), shortDigest(WorkerDigest()), a.Device, a.Device)
}

func shortDigest(d string) string {
	if len(d) > 12 {
		return d[:12]
	}
	if d == "" {
		return "none"
	}
	return d
}

// RuntimeName is the runtime identity (directory name under runtime/) for a device.
func RuntimeName(device string) (string, error) {
	s, err := Desired(device)
	if err != nil {
		return "", err
	}
	return s.ID(), nil
}

// DefaultModel is the catalog model setup activates when none is selected.
// Another model becomes the default only on recorded comparative evidence
// (docs/certification.md), never because it is newer or smaller.
const DefaultModel = "laya-base"

// ProviderOpenDecider is the provider kind of OpenDecider catalog models.
const ProviderOpenDecider = providerOpenDecider

// OpenDeciderNano is the catalog ID of the OpenDecider-nano candidate model.
const OpenDeciderNano = "opendecider-nano"

// Models is the decision-model catalog: every checkpoint Hachidori can
// materialize, each an immutable identity (upstream repository, revision and
// the exact files with their pinned digests). A model is materialized
// independently of the Python runtime and is not part of the runtime
// identity; every entry must be loadable by one of the runtime's providers
// (its Provider field). Adding a checkpoint means declaring another entry
// here, nothing else.
var Models = []home.ModelManifest{
	{
		// The upstream Laya checkpoint (English, ModernBERT-large) at the
		// revision Laya itself lists as reviewed in laya.revisions.PINNED_REVISIONS.
		ID:          DefaultModel,
		Provider:    providerLaya,
		Repo:        "convaiinnovations/laya",
		Revision:    "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
		Description: "upstream Laya checkpoint (English, ModernBERT-large)",
		Files: map[string]string{
			"rl_agent_config.json":            "ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd",
			"model.safetensors":               "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
			"encoder/config.json":             "bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979",
			"tokenizer/tokenizer.json":        "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
			"tokenizer/tokenizer_config.json": "50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12",
		},
	},
	{
		// OpenDecider-nano (Apache-2.0, Ettin-encoder-400m backbone with an MLP
		// decision head): a candidate, not the default. It scores the request's
		// closed options in one encoder pass and never generates text. The
		// revision is the upstream commit these digests were taken from;
		// model.safetensors is the Hub's LFS digest for that revision.
		ID:          OpenDeciderNano,
		Provider:    providerOpenDecider,
		Repo:        "manjunathshiva/opendecider-nano",
		Revision:    "7e42a1508d2beef44717d044831e87f2fc4db9f2",
		Description: "OpenDecider-nano candidate (English, Ettin-encoder-400m, Apache-2.0)",
		Files: map[string]string{
			"LICENSE":               "30f149868450e5288b6ed3b6fc0257a9597b5ffcdd961fdf3863a14a40b63b8d",
			"config.json":           "5dc491f5dafeb79b653c76cb3e5699a7958306ca8d85304f91ca16928c67dbaf",
			"head.safetensors":      "b57ad139c1ca8984a5205dc57aebeb921c9c59eaa536880d6b5e81d7734eee56",
			"model.safetensors":     "da243ae586e17ee87b5aa68e1bd06112c1c4cf756ea651d335ae7a2d6097cb38",
			"opendecider.json":      "ec0f4e9caa4cd95e2f30ab0e849cf62ce197ce0dcd3bf5fb1eb3eb12f7b480ba",
			"tokenizer.json":        "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
			"tokenizer_config.json": "5926e6ec4294f80294bd98d9176defa9fbae7f140525d06a1da987488e74a973",
		},
	},
}

var modelBaseURL = "https://huggingface.co/"

// LookupModel resolves a catalog model ID; the empty ID selects DefaultModel.
// Only catalog identities can be selected: there is no way to name an
// arbitrary repository or revision.
func LookupModel(id string) (home.ModelManifest, error) {
	if id == "" {
		id = DefaultModel
	}
	var known []string
	for _, m := range Models {
		if m.ID == id {
			return m, nil
		}
		known = append(known, m.ID)
	}
	sort.Strings(known)
	return home.ModelManifest{}, fmt.Errorf("unsupported model %q (supported: %s)", id, strings.Join(known, ", "))
}

// ModelDirName is the activation name (directory under models/, slash
// separated) of a catalog model, derived from its immutable repository and
// revision.
func ModelDirName(m home.ModelManifest) string {
	return strings.ReplaceAll(m.Repo, "/", "--") + "/" + m.Revision
}

// ActiveModel resolves the catalog model an activation record selects and
// checks that the record's model directory is the one that model derives.
// Records written before model selection existed carry no model ID; they
// are resolved by their directory.
func ActiveModel(a home.Active) (home.ModelManifest, error) {
	if a.ModelID == "" {
		for _, m := range Models {
			if ModelDirName(m) == a.Model {
				return m, nil
			}
		}
		return home.ModelManifest{}, fmt.Errorf("active model %s is not a catalog model (run `hachidori setup`)", a.Model)
	}
	m, err := LookupModel(a.ModelID)
	if err != nil {
		return m, fmt.Errorf("active model: %w (run `hachidori setup`)", err)
	}
	if ModelDirName(m) != a.Model {
		return m, fmt.Errorf("active model %s: directory %s does not match its catalog entry %s", a.ModelID, a.Model, ModelDirName(m))
	}
	return m, nil
}

func platform() string { return runtime.GOOS + "/" + runtime.GOARCH }

func digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
