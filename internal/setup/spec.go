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
	optpy "github.com/yohn-jp/hachidori/internal/optimize/py"
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

	// The Clef System One provider is carried by the model, not by a package:
	// the upstream joint_schema_model.py of the digest-pinned release is
	// imported from the model directory by the worker's clef adapter, on top
	// of the runtime's transformers. Its pin is the adapter contract version;
	// the adapter itself is part of the worker script, whose digest is part of
	// the runtime identity, and the distributions it needs are verified
	// explicitly (clefDistributions).
	providerClef       = home.ProviderClef
	clefAdapterVersion = "1"
)

// providerPins is RuntimeSpec.Provider: the pinned providers as
// name==version, comma separated.
const providerPins = providerLaya + "==" + layaVersion + "," + providerOpenDecider + "==" + openDeciderVersion + "," + providerClef + "==" + clefAdapterVersion

// clefDistributions are the packages the clef adapter loads a release with,
// besides torch. They are pinned in runtimespec/ and verified in every
// materialized runtime that declares the clef provider.
var clefDistributions = []string{"transformers==5.17.0", "safetensors==0.8.0", "tokenizers==0.23.2", "accelerate==1.15.0", "compressed-tensors==0.19.0"}

// CUDA carries FLA's two serving kernels. The backend distributions expose the
// same triton module; only the platform-appropriate one may be materialized.
func clefKernelDistributions(spec home.RuntimeSpec) []string {
	if spec.Flavor != "cu128" {
		return nil
	}
	backend := "triton==3.6.0"
	if spec.Platform == "windows/amd64" {
		backend = "triton-windows==3.6.0.post26"
	}
	return []string{"fla-core==0.5.2", backend}
}

// carriedProviders are the providers a runtime declares without a package of
// their own.
var carriedProviders = map[string]bool{providerClef: true}

// The optimizer runtime: the bounded environment that builds System One
// variants. It is separate from the serving runtime so that the compression
// stack never enters it. Its packages are pinned in optimizerspec/.
const (
	optimizerEngine = "llmcompressor"

	llmCompressorVersion   = "0.14.0"
	compressedTensorsPin   = "0.19.0"
	optimizerProviderPins  = optimizerEngine + "==" + llmCompressorVersion + ",compressed-tensors==" + compressedTensorsPin
	optimizerDeviceFlavour = "cpu"
)

//go:embed optimizerspec/pyproject.toml optimizerspec/uv.lock
var optimizerSpecFS embed.FS

// runtimeKind selects the embedded uv project and private script a Runtime
// Spec is materialized from: the serving runtime or the optimizer runtime.
type runtimeKind struct {
	file      func(name string) []byte
	script    []byte
	scriptRel string // slash separated, relative to the runtime directory
}

func kindOf(spec home.RuntimeSpec) runtimeKind {
	if spec.Role == home.RoleOptimizer {
		return runtimeKind{
			file: func(name string) []byte {
				b, err := optimizerSpecFS.ReadFile("optimizerspec/" + name)
				if err != nil {
					panic(err)
				}
				return b
			},
			script: optpy.Script, scriptRel: "worker/hachidori_optimizer.py",
		}
	}
	return runtimeKind{file: specFile, script: py.Script, scriptRel: "worker/hachidori_worker.py"}
}

// DesiredOptimizer is the Runtime Spec of the optimizer runtime on the
// current platform. The optimizer's first recipe transforms on the CPU, so
// the runtime is always the CPU flavor.
func DesiredOptimizer() (home.RuntimeSpec, error) { return desiredOptimizerFor(platform()) }

func desiredOptimizerFor(plat string) (home.RuntimeSpec, error) {
	uv, err := uvFor(plat)
	if err != nil {
		return home.RuntimeSpec{}, err
	}
	k := kindOf(home.RuntimeSpec{Role: home.RoleOptimizer})
	return home.RuntimeSpec{
		Schema:   SpecSchema,
		Role:     home.RoleOptimizer,
		Platform: plat,
		Python:   pythonVersion,
		Provider: optimizerProviderPins,
		Torch:    torchVersion + "+" + optimizerDeviceFlavour,
		Flavor:   optimizerDeviceFlavour,
		UV:       uvVersion,
		UVSHA256: uv.BinarySHA256,
		Project:  digest(k.file("pyproject.toml")),
		Lock:     digest(k.file("uv.lock")),
		Worker:   digest(k.script),
	}, nil
}

// OptimizerEngine is the optimizer backend and OptimizerEngineVersion its
// pinned version: recorded in every variant it builds.
const (
	OptimizerEngine        = optimizerEngine
	OptimizerEngineVersion = llmCompressorVersion
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

// ClefFlash is the catalog ID of Clef-Flash, the first System One model with
// derived execution variants.
const ClefFlash = "clef-flash"

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
	{
		// Clef-Flash (Apache-2.0): a 9B multimodal System One model, a Qwen3.5-9B
		// post-train with a joint schema head that returns one logit per allowed
		// option of every question in a single forward pass. Its typed-decision
		// use is text-only and never generates text. The revision is the upstream
		// commit these digests were taken from; the safetensors digests are the
		// Hub's LFS digests for that revision, the others were computed from the
		// files at it. joint_schema_model.py is upstream code that the clef
		// adapter imports from the model directory, only after its digest has
		// been verified. README.md is documentation and is not pinned.
		ID:          ClefFlash,
		Provider:    providerClef,
		Repo:        "Cloudflare/clef-flash",
		Revision:    "17f0b0ad64efb65d273590632833508766b2aae6",
		Description: "Clef-Flash System One model (Qwen3.5-9B backbone + joint schema head, Apache-2.0)",
		Files: map[string]string{
			"LICENSE":                          "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
			"chat_template.jinja":              "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
			"config.json":                      "66f87f6fb2616b46604daf2a9c67ddc87938296d07156efa34d59b5be49e3238",
			"generation_config.json":           "45707f8467bf4e54e112c6095a1b2d1f2d63651cf86816344cfff21f02dda0d4",
			"joint_head.safetensors":           "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0",
			"joint_head_config.json":           "77efe959a38b5b17b241543e129e695f3c77465ece55a25985279bd8176279a0",
			"joint_schema_model.py":            "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3",
			"model-00001-of-00004.safetensors": "8b45a8e968141cdcc58fb71c9adfc258e2c77b5f062bc636c1fd5bc5d916b565",
			"model-00002-of-00004.safetensors": "7590856c713eed844a2dcf48e6c43c4de165b788bc3f80e328311183cdbc7db8",
			"model-00003-of-00004.safetensors": "e6eac2467952c33361ed7dcb3c7959d1086bbe57201cd3749c3d769fdc17fe63",
			"model-00004-of-00004.safetensors": "9fcecc6556b39171238373a465f409794b7f821fb4cd1e6459e3a9c0fe317af7",
			"model.safetensors.index.json":     "941305ff9f77551e145a6cea976ef456cd5cb208cbece99f168376c752fcf96c",
			"processor_config.json":            "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
			"tokenizer.json":                   "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
			"tokenizer_config.json":            "91a08f825d370d085d692e04cf117cdd7faad7bf18e996f1e6031b6dab03db72",
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
