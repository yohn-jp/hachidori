package home

import (
	"bytes"
	"embed"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// EvaluationSample is one operator-visible evaluation bundle installed at a
// stable path beneath HACHIDORI_HOME.
type EvaluationSample struct {
	Name      string
	Dataset   string
	Questions string
	Policy    string
}

// InariSampleEvaluation returns the fixed install-local Inari sample paths.
// The files are Hachidori-owned resources, not mutable application state.
func (h Home) InariSampleEvaluation() EvaluationSample {
	base := h.Path("resources", "evaluation", "inari")
	return EvaluationSample{
		Name:      "Inari sample",
		Dataset:   filepath.Join(base, "dataset.jsonl"),
		Questions: filepath.Join(base, "questions"),
		Policy:    filepath.Join(base, "policy.json"),
	}
}

//go:embed sample_evaluation/dataset.jsonl sample_evaluation/policy.json sample_evaluation/questions/*.json
var inariSampleFS embed.FS

var inariSampleFiles = []string{
	"dataset.jsonl",
	"policy.json",
	"questions/authority_status.json",
	"questions/blocker_class.json",
	"questions/failure_class.json",
	"questions/finalization_status.json",
	"questions/verification_status.json",
}

// EnsureEvaluationResources materializes the Hachidori-owned Inari sample at
// its fixed path. Rerunning is deterministic: unchanged files are left alone,
// owned sample files are refreshed, and custom resources elsewhere are never
// inspected or modified.
func (h Home) EnsureEvaluationResources() error {
	if h.Root == "" {
		return errors.New("cannot provision evaluation resources without HACHIDORI_HOME")
	}
	base := h.InariSampleEvaluation()
	root := filepath.Dir(base.Dataset)
	for _, rel := range inariSampleFiles {
		src := "sample_evaluation/" + filepath.ToSlash(rel)
		data, err := inariSampleFS.ReadFile(src)
		if err != nil {
			return fmt.Errorf("embedded evaluation resource %s: %w", rel, err)
		}
		dst := filepath.Join(root, filepath.FromSlash(rel))
		if old, err := os.ReadFile(dst); err == nil && bytes.Equal(old, data) {
			continue
		} else if err != nil && !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("read evaluation resource %s: %w", dst, err)
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		if err := WriteFileAtomic(dst, data, 0o644); err != nil {
			return fmt.Errorf("write evaluation resource %s: %w", dst, err)
		}
	}
	return nil
}
