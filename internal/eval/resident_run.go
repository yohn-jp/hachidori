package eval

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/yohn-jp/hachidori/internal/question"
)

// ResidentRunSchema identifies the evidence of one resident's pass over a
// dataset, kept as a file so that it can later be paired with another run:
// a reference run and a candidate run need not be resident at the same time
// (the reference may run on the CPU for as long as it takes).
const ResidentRunSchema = "hachidori.resident-run.v1"

// ResidentRun is one ModelRun with the context needed to audit it alone. It
// is the same evidence a resident comparison records for each of its models;
// no new measurement exists here. Labelled says whether the dataset carried
// expected labels: an unlabelled run has Observations and no Quality (its
// Expected and Correct fields mean nothing and quality is not computed).
type ResidentRun struct {
	Schema        string              `json:"schema"`
	Endpoint      string              `json:"endpoint"`
	Dataset       string              `json:"dataset"`
	DatasetSHA256 string              `json:"dataset_sha256"`
	Definitions   []question.Identity `json:"question_definitions,omitempty"`
	StartedAt     string              `json:"started_at"`
	Labelled      bool                `json:"labelled"`
	Declared      Declared            `json:"declared"`
	Run           ModelRun            `json:"run"`
}

// RunResident evaluates one already-resident model over cases through direct
// resident targeting, exactly as RunResidents does for each model of a
// comparison: it only sends decide requests naming the model and reads status,
// and never starts, restarts or reloads anything. labelled is whether cases
// carry expected labels (LoadAny).
func RunResident(e Endpoint, cases []Case, datasetSHA256 string, labelled bool, model string, opt ResidentOptions) (ResidentRun, error) {
	opt.Models = []string{model}
	cmp, err := runResidents(e, cases, datasetSHA256, opt, 1)
	if err != nil {
		return ResidentRun{}, err
	}
	if len(cmp.Runs) != 1 {
		return ResidentRun{}, errors.New("a resident run produced no run")
	}
	r := ResidentRun{Schema: ResidentRunSchema, DatasetSHA256: datasetSHA256, Definitions: cmp.Definitions,
		StartedAt: time.Now().UTC().Format(time.RFC3339), Labelled: labelled, Declared: cmp.Declared, Run: cmp.Runs[0]}
	if !labelled {
		// Labels were never given: nothing about correctness was measured.
		for i := range r.Run.Observations {
			r.Run.Observations[i].Expected, r.Run.Observations[i].Correct = "", false
		}
		r.Run.Quality, r.Run.PerQuestion, r.Run.PerFamily = Quality{N: len(r.Run.Observations)}, []Slice{}, []Slice{}
	}
	return r, nil
}

// ValidateResidentRun checks the internal consistency of a run document.
func ValidateResidentRun(r ResidentRun) error {
	if r.Schema != ResidentRunSchema {
		return fmt.Errorf("resident run schema %q, want %q", r.Schema, ResidentRunSchema)
	}
	run := r.Run
	switch {
	case run.Model == "":
		return errors.New("run has no model")
	case run.Identity.Digest == "":
		return errors.New("resident identity is missing")
	case r.DatasetSHA256 == "" || r.DatasetSHA256 != run.DatasetSHA256:
		return errors.New("dataset digest is missing or differs from the run's")
	case run.Quality.N != len(run.Observations):
		return fmt.Errorf("quality.n is %d but there are %d observations", run.Quality.N, len(run.Observations))
	case run.ErrorCount != len(run.Errors):
		return fmt.Errorf("error_count is %d but there are %d errors", run.ErrorCount, len(run.Errors))
	}
	for j, o := range run.Observations {
		if o.ServedModel != run.Model || o.IdentitySHA256 != run.Identity.Digest {
			return fmt.Errorf("observations[%d]: served by %q, not the run's resident %q", j, o.ServedModel, run.Model)
		}
		if o.InputSHA256 == "" {
			return fmt.Errorf("observations[%d]: input_sha256 is missing", j)
		}
	}
	return nil
}

// DecodeResidentRun parses a resident run strictly: another schema, unknown
// fields, trailing data or an inconsistent document are rejected.
func DecodeResidentRun(data []byte) (ResidentRun, error) {
	var probe struct {
		Schema string `json:"schema"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return ResidentRun{}, fmt.Errorf("not a resident run: %w", err)
	}
	if probe.Schema != ResidentRunSchema {
		return ResidentRun{}, fmt.Errorf("resident run schema %q, want %q", probe.Schema, ResidentRunSchema)
	}
	var r ResidentRun
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&r); err != nil {
		return ResidentRun{}, fmt.Errorf("malformed %s: %w", ResidentRunSchema, err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return ResidentRun{}, fmt.Errorf("malformed %s: trailing data", ResidentRunSchema)
	}
	if err := ValidateResidentRun(r); err != nil {
		return ResidentRun{}, fmt.Errorf("malformed %s: %w", ResidentRunSchema, err)
	}
	return r, nil
}

// LoadResidentRun reads and decodes a resident run file.
func LoadResidentRun(path string) (ResidentRun, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return ResidentRun{}, err
	}
	r, err := DecodeResidentRun(b)
	if err != nil {
		return r, fmt.Errorf("%s: %w", path, err)
	}
	return r, nil
}
