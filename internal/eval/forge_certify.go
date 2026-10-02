package eval

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
)

// CertificationProducerSchema versions the linkage between a certification and
// the internal Forge runs it was computed from. A certification without a
// producer block (a low-level `certify evaluate` of run files, or one recorded
// before this linkage existed) stays valid and readable.
const CertificationProducerSchema = "hachidori.certification-producer.v1"

// ProducerRun is one producer run a certification binds: its stable evidence
// ID, the digest of the ResidentRun inside it, and the exact execution target
// it was recorded on (source/variant identity, runtime, requested and actual
// device and dtype, quantized execution facts).
type ProducerRun struct {
	EvidenceID string         `json:"evidence_id"`
	RunSHA256  string         `json:"run_sha256"`
	Target     ForgeRunTarget `json:"target"`
}

// ProducerCheck is the outcome of a readiness gate that ran before the runs.
type ProducerCheck struct {
	Result string `json:"result"`
	Device string `json:"device"`
	At     string `json:"at,omitempty"`
}

// CertificationProducer binds a certification to the Forge execution evidence
// that produced both of its runs.
type CertificationProducer struct {
	Schema    string         `json:"schema"`
	Reference ProducerRun    `json:"reference"`
	Candidate ProducerRun    `json:"candidate"`
	Preflight *ProducerCheck `json:"preflight,omitempty"`
	Probe     *ProducerCheck `json:"probe,omitempty"`
}

// RecordProducer is the part of the producer linkage a certification record
// keeps: the evidence IDs and run digests, to be checked against the report.
type RecordProducer struct {
	ReferenceEvidenceID string `json:"reference_evidence_id"`
	ReferenceRunSHA256  string `json:"reference_run_sha256"`
	CandidateEvidenceID string `json:"candidate_evidence_id"`
	CandidateRunSHA256  string `json:"candidate_run_sha256"`
}

func (p *CertificationProducer) record() *RecordProducer {
	if p == nil {
		return nil
	}
	return &RecordProducer{ReferenceEvidenceID: p.Reference.EvidenceID, ReferenceRunSHA256: p.Reference.RunSHA256,
		CandidateEvidenceID: p.Candidate.EvidenceID, CandidateRunSHA256: p.Candidate.RunSHA256}
}

// ProducerOf is the producer linkage of a certification computed from two
// recorded Forge runs.
func ProducerOf(ref, cand ForgeRun) *CertificationProducer {
	return &CertificationProducer{Schema: CertificationProducerSchema,
		Reference: ProducerRun{EvidenceID: ref.ID, RunSHA256: ref.RunSHA256, Target: ref.Target},
		Candidate: ProducerRun{EvidenceID: cand.ID, RunSHA256: cand.RunSHA256, Target: cand.Target}}
}

// ForgeInputIdentity is the semantic evaluation input a run was recorded over,
// as the authoritative digests of its content and never as a file name: the
// normalized dataset, the Question Definitions, the questions as sent, the
// normalized and the sent requests, and the order of the (case, question)
// observations.
type ForgeInputIdentity struct {
	DatasetSHA256             string
	QuestionDefinitionsSHA256 string
	QuestionsSHA256           string
	InputSHA256               string
	SentSHA256                string
	ObservationOrderSHA256    string
	Cases                     int
	Observations              int
	Labelled                  bool
}

// InputIdentity is the evaluation input identity of the run.
func (r ForgeRun) InputIdentity() ForgeInputIdentity {
	keys := make([][2]string, 0, len(r.Run.Run.Observations))
	for _, o := range r.Run.Run.Observations {
		keys = append(keys, [2]string{o.CaseID, o.QuestionID})
	}
	return ForgeInputIdentity{DatasetSHA256: r.DatasetSHA256, QuestionDefinitionsSHA256: r.QuestionDefinitionsSHA256,
		QuestionsSHA256: digestJSON(r.Run.Run.Questions), InputSHA256: r.Run.Run.InputSHA256, SentSHA256: r.Run.Run.SentSHA256,
		ObservationOrderSHA256: digestJSON(keys), Cases: r.Run.Run.Cases, Observations: len(r.Run.Run.Observations), Labelled: r.Run.Labelled}
}

// Differences names every field in which two input identities differ.
func (a ForgeInputIdentity) Differences(b ForgeInputIdentity) []string {
	var d []string
	diff := func(name string, x, y any) {
		if x != y {
			d = append(d, fmt.Sprintf("%s differs (%v, %v)", name, x, y))
		}
	}
	diff("normalized dataset digest", a.DatasetSHA256, b.DatasetSHA256)
	diff("Question Definition digest", a.QuestionDefinitionsSHA256, b.QuestionDefinitionsSHA256)
	diff("question digest", a.QuestionsSHA256, b.QuestionsSHA256)
	diff("normalized input digest", a.InputSHA256, b.InputSHA256)
	diff("digest of the requests sent", a.SentSHA256, b.SentSHA256)
	diff("case/question observation order", a.ObservationOrderSHA256, b.ObservationOrderSHA256)
	diff("case count", a.Cases, b.Cases)
	diff("observation count", a.Observations, b.Observations)
	diff("labelled", a.Labelled, b.Labelled)
	return d
}

// AlignmentError is a refusal to pair two runs: every reason is listed.
type AlignmentError struct{ Reasons []string }

func (e *AlignmentError) Error() string {
	return "the reference and candidate runs are not the same evaluation input: " + strings.Join(e.Reasons, "; ")
}

// AlignForgeRuns proves that two Forge runs represent the same semantic
// evaluation input before any certification is computed: the same normalized
// dataset, Question Definitions, questions, requests and (case, question)
// order, from identical declared controls. A difference is a refusal, not a
// finding. It does not replace the checks of Certify, which still run.
func AlignForgeRuns(ref, cand ForgeRun) error {
	reasons := ref.InputIdentity().Differences(cand.InputIdentity())
	if !reflect.DeepEqual(ref.Run.Definitions, cand.Run.Definitions) {
		reasons = append(reasons, "the Question Definition identities differ")
	}
	if !equalControls(ref.Run.Declared, cand.Run.Declared) {
		reasons = append(reasons, "the declared controls differ")
	}
	if len(reasons) > 0 {
		return &AlignmentError{Reasons: reasons}
	}
	return nil
}

// checkProducer verifies a record's producer linkage against its report: both
// carry the same evidence IDs and run digests, or neither carries any.
func checkProducer(rec CertificationRecord, c Certification) error {
	switch {
	case rec.Producer == nil && c.Producer == nil:
		return nil
	case rec.Producer == nil || c.Producer == nil:
		return errors.New("the record and its report disagree about the producer runs")
	case c.Producer.Schema != CertificationProducerSchema:
		return fmt.Errorf("report producer schema %q", c.Producer.Schema)
	case *rec.Producer != *c.Producer.record():
		return errors.New("report binds different producer runs")
	}
	return nil
}
