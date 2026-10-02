package setup

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
)

// KindVariant is the artifact kind of a derived System One variant, addressed
// by its variant ID.
const KindVariant = "variant"

// OptimizerReportFile is the optimizer's own record of what it did, written
// into every variant it builds and listed among the variant's artifacts.
const OptimizerReportFile = "hachidori-optimizer-report.json"

// Variant phases of an operation.
const (
	PhaseVariant Phase = "variant" // validating a variant against its source and artifacts
)

// SupportsVariants reports whether variants of the catalog model exist as a
// concept: only System One models are optimized.
func SupportsVariants(m home.ModelManifest) bool { return m.Provider == home.ProviderClef }

// VariantEntry is one variant under HACHIDORI_HOME as the inventory reports
// it. Certification is the state resolved from the variant's records, never
// from its manifest.
type VariantEntry struct {
	ID                 string   `json:"id"`
	SourceID           string   `json:"source_id"`
	SourceRevision     string   `json:"source_revision"`
	Recipe             string   `json:"recipe"`
	Scheme             string   `json:"scheme"`
	Bits               int      `json:"bits"`
	GroupSize          int      `json:"group_size"`
	DType              string   `json:"dtype"`
	Engine             string   `json:"engine"`
	EngineVersion      string   `json:"engine_version"`
	BuildID            string   `json:"build_id"`
	ManifestSHA256     string   `json:"manifest_sha256"`
	Files              int      `json:"files"`
	Preserved          []string `json:"preserved"` // preserved module patterns / carried files
	Certification      string   `json:"certification"`
	CertificationNote  string   `json:"certification_note,omitempty"`
	SourceMaterialized bool     `json:"source_materialized"`
	Active             bool     `json:"active"`
	Experimental       bool     `json:"experimental,omitempty"`
	Verified           bool     `json:"verified"` // artifact digests verified (only when requested)
	Problem            string   `json:"problem,omitempty"`
}

// variantDirs lists the variant directories of one source model.
func variantDirs(h home.Home, sourceID string) []string {
	entries, err := os.ReadDir(h.VariantsDir(sourceID))
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() && !strings.HasPrefix(e.Name(), ".") {
			out = append(out, e.Name())
		}
	}
	sort.Strings(out)
	return out
}

// FindVariant resolves a variant ID to the catalog model it derives from and
// its manifest, validating the manifest's identity and its source link
// against the catalog. It reads only manifests: artifact bytes are not
// touched.
func FindVariant(h home.Home, variantID string) (home.ModelManifest, home.VariantManifest, error) {
	for _, m := range Models {
		if !SupportsVariants(m) {
			continue
		}
		dir := h.VariantDir(m.ID, variantID)
		if _, err := os.Stat(dir); err != nil {
			continue
		}
		return loadVariant(h, m, variantID)
	}
	return home.ModelManifest{}, home.VariantManifest{}, fmt.Errorf("variant %q is not in this home (see `hachidori variant list`)", variantID)
}

func loadVariant(h home.Home, m home.ModelManifest, variantID string) (home.ModelManifest, home.VariantManifest, error) {
	v, err := home.ReadVariant(h.VariantDir(m.ID, variantID))
	if err != nil {
		return m, v, fmt.Errorf("variant %s: %w", variantID, err)
	}
	if v.ID != variantID {
		return m, v, fmt.Errorf("variant directory %s holds variant %s", variantID, v.ID)
	}
	if err := v.CheckSource(m); err != nil {
		return m, v, err
	}
	return m, v, nil
}

// ListVariants reports every variant of every catalog model that supports
// them. It reads only; with verify it also hashes every artifact.
func ListVariants(h home.Home, verify bool, active *home.Active) []VariantEntry {
	var out []VariantEntry
	for _, m := range Models {
		if !SupportsVariants(m) {
			continue
		}
		_, srcErr := os.Stat(h.Path("models", filepath.FromSlash(ModelDirName(m)), "hachidori-model.json"))
		for _, id := range variantDirs(h, m.ID) {
			e := VariantEntry{ID: id, SourceID: m.ID, SourceRevision: m.Revision, SourceMaterialized: srcErr == nil}
			_, v, err := loadVariant(h, m, id)
			if err != nil {
				e.Problem = err.Error()
				e.Certification = eval.StateUncertified
				out = append(out, e)
				continue
			}
			e.Recipe, e.Scheme, e.Bits, e.GroupSize, e.DType = v.Recipe.Name, v.Weights.Scheme, v.Weights.Bits, v.Weights.GroupSize, v.Weights.DType
			e.Engine, e.EngineVersion, e.BuildID, e.ManifestSHA256, e.Files = v.Optimizer.Engine, v.Optimizer.Version, v.BuildID, v.ManifestSHA256(), len(v.Files)
			for _, p := range v.Preserved {
				e.Preserved = append(e.Preserved, p.Pattern)
			}
			st := eval.ResolveCertification(h, v)
			e.Certification = st.State
			if len(st.Problems) > 0 {
				e.CertificationNote = strings.Join(st.Problems, "; ")
			}
			if active != nil && active.ModelID == m.ID && active.Variant == id {
				e.Active, e.Experimental = true, active.Experimental
			}
			if verify {
				if err := VerifyVariantArtifacts(h, m, v, nil); err != nil {
					e.Problem = err.Error()
				} else {
					e.Verified = true
				}
			}
			out = append(out, e)
		}
	}
	return out
}

// VerifyVariantArtifacts is the full integrity check of a variant: its
// directory holds exactly the files the manifest lists, every file matches its
// digest, files the variant carries over from the source are byte-identical to
// the source's pinned files, and the preserved modules the manifest declares
// are present in the output metadata at the precision it declares.
func VerifyVariantArtifacts(h home.Home, m home.ModelManifest, v home.VariantManifest, obs *Observer) error {
	dir := h.VariantDir(m.ID, v.ID)
	if err := v.CheckSource(m); err != nil {
		return err
	}
	if err := checkVariantFiles(dir, v, obs); err != nil {
		return err
	}
	return CheckPreserved(dir, m, v)
}

// checkVariantFiles verifies a directory holds exactly the artifacts of v.
func checkVariantFiles(dir string, v home.VariantManifest, obs *Observer) error {
	want := map[string]bool{home.VariantManifestFile: true}
	for rel := range v.Files {
		want[rel] = true
	}
	var extra []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, rerr := filepath.Rel(dir, p)
		if rerr != nil || rel == "." {
			return rerr
		}
		rel = filepath.ToSlash(rel)
		if d.Type()&fs.ModeSymlink != 0 {
			return fmt.Errorf("%s is a symbolic link", rel)
		}
		if d.IsDir() {
			return nil
		}
		if !want[rel] {
			extra = append(extra, rel)
		}
		return nil
	})
	if err != nil {
		return err
	}
	if len(extra) > 0 {
		sort.Strings(extra)
		return fmt.Errorf("variant %s holds files its manifest does not list: %s", v.ID, strings.Join(extra, ", "))
	}
	names := v.VariantFileNames()
	for i, rel := range names {
		got, err := fileSHA256Observed(filepath.Join(dir, filepath.FromSlash(rel)), obs,
			Progress{Step: StepVerify, Detail: rel, Item: i + 1, Items: len(names)})
		if err != nil {
			return fmt.Errorf("variant %s: %w", v.ID, err)
		}
		if got != v.Files[rel] {
			return fmt.Errorf("variant %s: %s: sha256 %s, want %s", v.ID, rel, got, v.Files[rel])
		}
	}
	return nil
}

// optimizerReport is the part of the optimizer's report the preserved-module
// check reads.
type optimizerReport struct {
	Schema           string `json:"schema"`
	Scheme           string `json:"scheme"`
	QuantizedModules int    `json:"quantized_modules"`
	Preserved        map[string]struct {
		Patterns     []string `json:"patterns"`
		WrittenDType string   `json:"written_dtype"`
	} `json:"preserved_modules"`
	Ignore []string `json:"quantization_config_ignore"`
}

// CheckPreserved verifies, from the variant's own output metadata, that what
// its manifest declares was preserved really is: every backbone pattern
// matched modules and those modules were written at a higher precision, the
// quantization config ignores exactly them, and every carried file is
// byte-identical to the source file of the same name. It reads the saved
// config and the optimizer's report, both of which are artifacts covered by
// the variant's digests.
func CheckPreserved(dir string, m home.ModelManifest, v home.VariantManifest) error {
	var rep optimizerReport
	if err := home.ReadJSON(filepath.Join(dir, OptimizerReportFile), &rep); err != nil {
		return fmt.Errorf("variant %s: optimizer report: %w", v.ID, err)
	}
	var cfg struct {
		Quantization struct {
			Method string   `json:"quant_method"`
			Ignore []string `json:"ignore"`
			Groups map[string]struct {
				Weights struct {
					Bits      int    `json:"num_bits"`
					Type      string `json:"type"`
					GroupSize int    `json:"group_size"`
					Symmetric bool   `json:"symmetric"`
				} `json:"weights"`
				Input any `json:"input_activations"`
			} `json:"config_groups"`
			Format string `json:"format"`
		} `json:"quantization_config"`
	}
	if err := home.ReadJSON(filepath.Join(dir, "config.json"), &cfg); err != nil {
		return fmt.Errorf("variant %s: config.json: %w", v.ID, err)
	}
	q := cfg.Quantization
	if q.Method != "compressed-tensors" || len(q.Groups) != 1 {
		return fmt.Errorf("variant %s: saved config is not a single compressed-tensors group", v.ID)
	}
	for _, g := range q.Groups {
		w := v.Weights
		if g.Weights.Bits != w.Bits || g.Weights.Type != "int" || g.Weights.GroupSize != w.GroupSize || g.Weights.Symmetric != w.Symmetric || g.Input != nil {
			return fmt.Errorf("variant %s: saved scheme (%d-bit %s group %d) is not the declared %s", v.ID, g.Weights.Bits, g.Weights.Type, g.Weights.GroupSize, w.Scheme)
		}
	}
	if q.Format != packFormat(v.Weights.Format) {
		return fmt.Errorf("variant %s: saved format %q, declared %q", v.ID, q.Format, v.Weights.Format)
	}
	saved := map[string]bool{}
	for _, n := range q.Ignore {
		saved[n] = true
	}
	if len(saved) != len(rep.Preserved) {
		return fmt.Errorf("variant %s: the saved config ignores %d modules, the optimizer report preserves %d", v.ID, len(saved), len(rep.Preserved))
	}
	covered := map[string]bool{}
	for name, p := range rep.Preserved {
		if !saved[name] {
			return fmt.Errorf("variant %s: preserved module %s is not in the saved config's ignore list", v.ID, name)
		}
		switch p.WrittenDType {
		case "BF16", "F16", "F32":
		default:
			return fmt.Errorf("variant %s: preserved module %s was written as %q", v.ID, name, p.WrittenDType)
		}
		for _, pat := range p.Patterns {
			covered[pat] = true
		}
	}
	for _, p := range v.Preserved {
		switch p.Scope {
		case home.ScopeBackbone:
			if !covered[p.Pattern] {
				return fmt.Errorf("variant %s: declared preserved pattern %q matched no module in the output", v.ID, p.Pattern)
			}
		case home.ScopeCarried:
			if v.Files[p.Pattern] == "" || v.Files[p.Pattern] != m.Files[p.Pattern] {
				return fmt.Errorf("variant %s: carried file %s is not byte-identical to the source file", v.ID, p.Pattern)
			}
		}
	}
	for _, rel := range v.Recipe.Carry {
		if v.Files[rel] != m.Files[rel] {
			return fmt.Errorf("variant %s: carried file %s differs from the source", v.ID, rel)
		}
	}
	return nil
}

// packFormat is the compressed-tensors format name inside a manifest's
// "compressed-tensors/<format>" serialization format.
func packFormat(f string) string { _, after, _ := strings.Cut(f, "/"); return after }

// ErrVariantNotCertified is returned (wrapped) when a variant is refused
// activation for want of an accepted certification record.
var ErrVariantNotCertified = errors.New("variant is not certified")

// DigestTree digests every regular file under dir (slash separated relative
// paths) except the files named in skip, reporting determinate byte progress
// for each. It rejects symbolic links and anything that is not a regular file.
func DigestTree(dir string, skip map[string]bool, obs *Observer) (map[string]string, error) {
	var rels []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel, rerr := filepath.Rel(dir, p)
		if rerr != nil {
			return rerr
		}
		rel = filepath.ToSlash(rel)
		if !d.Type().IsRegular() {
			return fmt.Errorf("%s is not a regular file", rel)
		}
		if !skip[rel] {
			rels = append(rels, rel)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(rels)
	out := make(map[string]string, len(rels))
	for i, rel := range rels {
		d, err := fileSHA256Observed(filepath.Join(dir, filepath.FromSlash(rel)), obs, Progress{Step: StepVerify, Detail: rel, Item: i + 1, Items: len(rels)})
		if err != nil {
			return nil, err
		}
		out[rel] = d
	}
	return out, nil
}
