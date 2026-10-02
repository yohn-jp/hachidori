package eval

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/yohn-jp/hachidori/internal/home"
)

// RecordSchema identifies a certification record: the small, durable binding
// between one certification report and the exact variant it is about. A
// record of this schema carries Sequence, its position in the variant's total
// order of certifications.
const RecordSchema = "hachidori.certification-record/2"

// LegacyRecordSchema is the record schema written before records carried a
// sequence. Its records stay readable; they are ordered only by their
// second-precision creation time, so two of them created within the same
// second have no order, and that ambiguity is reported, never resolved.
const LegacyRecordSchema = "hachidori.certification-record/1"

// Certification states of a variant, as resolved from its records.
const (
	StateUncertified = "uncertified" // no valid record
	StateAccepted    = VerdictAccepted
	StateRejected    = VerdictRejected
	// StateExperimental is what status reports for a variant that was
	// activated on an explicit operator request without an accepted record.
	StateExperimental = "experimental/uncertified"
	// StateAmbiguous: valid records exist but the latest of them cannot be
	// determined (legacy records created within the same second, or two
	// records claiming the same sequence). The variant is neither accepted nor
	// uncertified: it is not activated, not even experimentally, until it is
	// certified again.
	StateAmbiguous = "ambiguous"
)

// PolicyRef names the policy a record applied.
type PolicyRef struct {
	ID     string `json:"id"`
	SHA256 string `json:"sha256"`
}

// CertificationRecord is state/certifications/<variant id>/<report>.record.json.
// It is written after the report it names, so its presence marks a complete
// certification. It carries the identities activation checks and the verdict;
// everything else is in the report, whose digest it records.
//
// Sequence is the ordering authority: the n-th certification of the variant
// has sequence n, claimed exclusively on disk before the record is written
// (see claimSequence). A later certification therefore always has a larger
// sequence, whatever the wall clock, the report content or the order a
// directory is listed in. Legacy records have none (zero).
type CertificationRecord struct {
	Schema                  string             `json:"schema"`
	Sequence                uint64             `json:"sequence,omitempty"`
	CreatedAt               string             `json:"created_at"`
	VariantID               string             `json:"variant_id"`
	VariantManifestSHA256   string             `json:"variant_manifest_sha256"`
	Source                  home.VariantSource `json:"source"`
	ReferenceIdentitySHA256 string             `json:"reference_identity_sha256"`
	CandidateIdentitySHA256 string             `json:"candidate_identity_sha256"`
	DatasetSHA256           string             `json:"dataset_sha256"`
	QuestionsSHA256         string             `json:"questions_sha256"`
	InputSHA256             string             `json:"input_sha256"`
	Policy                  PolicyRef          `json:"policy"`
	Verdict                 string             `json:"verdict"`
	Failed                  []string           `json:"failed_criteria"`
	ReportSHA256            string             `json:"report_sha256"`
	Report                  string             `json:"report"` // file name beside the record
	// Producer is the evidence of the Forge runs the certification was
	// computed from (additive: absent for run-file certifications and for
	// records written before it existed).
	Producer *RecordProducer `json:"producer,omitempty"`
}

func certDir(h home.Home, variantID string) string {
	return h.Path(filepath.FromSlash(home.CertificationDir(variantID)))
}

// SaveCertification writes the certification report and then its record under
// the variant's certification directory, both atomically, and returns the
// record. Accepted and rejected certifications are both kept: the evidence is
// preserved whatever the policy decided. It never touches the variant.
func SaveCertification(h home.Home, c Certification) (CertificationRecord, error) {
	if c.Schema != CertificationSchema {
		return CertificationRecord{}, fmt.Errorf("certification schema %q, want %q", c.Schema, CertificationSchema)
	}
	body, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return CertificationRecord{}, err
	}
	body = append(body, '\n')
	sum := sha256.Sum256(body)
	reportSHA := hex.EncodeToString(sum[:])
	name := reportSHA[:16]
	rec := CertificationRecord{Schema: RecordSchema, CreatedAt: c.CreatedAt, VariantID: c.Variant.ID,
		VariantManifestSHA256: c.Variant.ManifestSHA256, Source: c.Source,
		ReferenceIdentitySHA256: c.Reference.IdentitySHA256, CandidateIdentitySHA256: c.Candidate.IdentitySHA256,
		DatasetSHA256: c.Dataset.SHA256, QuestionsSHA256: c.Dataset.QuestionsSHA256, InputSHA256: c.Dataset.InputSHA256,
		Policy:  PolicyRef{ID: c.Policy.ID, SHA256: c.PolicySHA256},
		Verdict: c.Verdict.Status, Failed: append([]string{}, c.Verdict.Failed()...),
		ReportSHA256: reportSHA, Report: name + ".report.json", Producer: c.Producer.record()}
	dir := certDir(h, c.Variant.ID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return rec, err
	}
	if err := home.WriteFileAtomic(filepath.Join(dir, rec.Report), body, 0o644); err != nil {
		return rec, err
	}
	seq, err := claimSequence(dir)
	if err != nil {
		return rec, fmt.Errorf("claiming the certification sequence: %w", err)
	}
	rec.Sequence = seq
	if err := home.WriteJSON(filepath.Join(dir, name+".record.json"), rec); err != nil {
		return rec, err
	}
	return rec, nil
}

// orderSuffix names the sequence claims of a certification directory:
// <sequence, 20 digits>.order, empty files created exclusively.
const orderSuffix = ".order"

// claimSequence claims the next certification sequence of dir: one more than
// the highest claim or record sequence present, taken by creating its claim
// file exclusively, so two writers never hold the same sequence and a claim
// is never reused, even when its record was never written. The claims are
// files of the home, so the order survives a restart.
func claimSequence(dir string) (uint64, error) {
	var next uint64
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, err
	}
	for _, e := range entries {
		var n uint64
		switch name := e.Name(); {
		case strings.HasSuffix(name, orderSuffix):
			n, err = strconv.ParseUint(strings.TrimSuffix(name, orderSuffix), 10, 64)
			if err != nil {
				continue
			}
		case strings.HasSuffix(name, ".record.json"):
			var rec struct {
				Sequence uint64 `json:"sequence"`
			}
			if b, err := os.ReadFile(filepath.Join(dir, name)); err == nil && json.Unmarshal(b, &rec) == nil {
				n = rec.Sequence
			}
		}
		next = max(next, n)
	}
	for attempt := 0; attempt < 64; attempt++ {
		next++
		f, err := os.OpenFile(filepath.Join(dir, fmt.Sprintf("%020d%s", next, orderSuffix)), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if err == nil {
			return next, f.Close()
		}
		if !errors.Is(err, fs.ErrExist) {
			return 0, err
		}
	}
	return 0, errors.New("no certification sequence could be claimed")
}

// DecodeCertification parses a certification report strictly.
func DecodeCertification(data []byte) (Certification, error) {
	var c Certification
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return Certification{}, fmt.Errorf("malformed %s: %w", CertificationSchema, err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return Certification{}, fmt.Errorf("malformed %s: trailing data", CertificationSchema)
	}
	if c.Schema != CertificationSchema {
		return Certification{}, fmt.Errorf("certification schema %q, want %q", c.Schema, CertificationSchema)
	}
	return c, nil
}

// CertificationState is the certification state of a variant resolved from
// its records: the verdict of the latest valid record, uncertified when there
// is none, or ambiguous when the latest cannot be determined. Problems lists
// records that were found and not trusted, and why, and any ordering
// ambiguity; they never make a variant certified.
type CertificationState struct {
	State    string               `json:"state"`
	Record   *CertificationRecord `json:"record,omitempty"`
	Problems []string             `json:"problems,omitempty"`
}

// ResolveCertification is the one reader of certification state. A record
// counts only if everything it claims is verified: it names this variant by ID
// and manifest digest and its source; its report exists, hashes to the
// recorded digest and binds the same identities; the report's own policy
// digests to the recorded one; and the verdict recomputed from that policy and
// the report's evidence is the recorded verdict. A record that fails any of
// that is ignored, so a hand-edited verdict can never certify a variant.
//
// The latest valid record is the one with the highest sequence. Every
// sequenced record is later than every legacy record (sequences were
// introduced after them). Among legacy records only, the latest is the one
// with the latest creation time; when several share it, they have no order
// and the state is ambiguous. Neither the report digest nor the order of the
// directory listing ever decides which record is the latest.
func ResolveCertification(h home.Home, v home.VariantManifest) CertificationState {
	st := CertificationState{State: StateUncertified}
	dir := certDir(h, v.ID)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if !errors.Is(err, fs.ErrNotExist) {
			st.Problems = append(st.Problems, err.Error())
		}
		return st
	}
	var valid []CertificationRecord
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".record.json") {
			continue
		}
		rec, err := verifyRecord(dir, e.Name(), v)
		if err != nil {
			st.Problems = append(st.Problems, e.Name()+": "+err.Error())
			continue
		}
		valid = append(valid, rec)
	}
	if len(valid) == 0 {
		return st
	}
	latest, tied := latestRecords(valid)
	if len(tied) > 1 {
		names := make([]string, 0, len(tied))
		for _, r := range tied {
			names = append(names, r.Report)
		}
		sort.Strings(names)
		st.State = StateAmbiguous
		st.Problems = append(st.Problems, fmt.Sprintf("records of reports %s %s: their order cannot be determined; certify the variant again",
			strings.Join(names, ", "), latest))
		sort.Strings(st.Problems)
		return st
	}
	rec := tied[0]
	st.State, st.Record = rec.Verdict, &rec
	sort.Strings(st.Problems)
	return st
}

// latestRecords returns the records that are latest under the total order
// (one, unless the order is ambiguous) and how they tie.
func latestRecords(valid []CertificationRecord) (string, []CertificationRecord) {
	var top []CertificationRecord
	var best uint64
	for _, r := range valid {
		switch {
		case r.Sequence == 0:
		case r.Sequence > best:
			best, top = r.Sequence, []CertificationRecord{r}
		case r.Sequence == best:
			top = append(top, r)
		}
	}
	if best > 0 {
		return fmt.Sprintf("claim the same sequence %d", best), top
	}
	var at string
	for _, r := range valid {
		switch {
		case r.CreatedAt > at:
			at, top = r.CreatedAt, []CertificationRecord{r}
		case r.CreatedAt == at:
			top = append(top, r)
		}
	}
	return "are legacy records created within the same second (" + at + ")", top
}

func verifyRecord(dir, name string, v home.VariantManifest) (CertificationRecord, error) {
	var rec CertificationRecord
	b, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return rec, err
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&rec); err != nil {
		return rec, err
	}
	switch {
	case rec.Schema == RecordSchema && rec.Sequence == 0:
		return rec, errors.New("record carries no sequence")
	case rec.Schema == LegacyRecordSchema && rec.Sequence != 0:
		return rec, errors.New("legacy record claims a sequence")
	case rec.Schema != RecordSchema && rec.Schema != LegacyRecordSchema:
		return rec, fmt.Errorf("record schema %q", rec.Schema)
	}
	if rec.VariantID != v.ID || rec.VariantManifestSHA256 != v.ManifestSHA256() || rec.Source != v.Source {
		return rec, errors.New("record is not about this variant (id, manifest digest or source differ)")
	}
	if rec.Verdict != VerdictAccepted && rec.Verdict != VerdictRejected {
		return rec, fmt.Errorf("record verdict %q", rec.Verdict)
	}
	if strings.ContainsAny(rec.Report, `/\`) || rec.Report == "" {
		return rec, fmt.Errorf("record names report %q", rec.Report)
	}
	body, err := os.ReadFile(filepath.Join(dir, rec.Report))
	if err != nil {
		return rec, fmt.Errorf("report: %w", err)
	}
	if sum := sha256.Sum256(body); hex.EncodeToString(sum[:]) != rec.ReportSHA256 {
		return rec, errors.New("report does not match the recorded digest")
	}
	c, err := DecodeCertification(body)
	if err != nil {
		return rec, err
	}
	producerMismatch := checkProducer(rec, c)
	switch {
	case c.Variant.ID != v.ID || c.Variant.ManifestSHA256 != rec.VariantManifestSHA256 || c.Source != v.Source:
		return rec, errors.New("report binds a different variant or source")
	case c.Reference.IdentitySHA256 != rec.ReferenceIdentitySHA256 || c.Candidate.IdentitySHA256 != rec.CandidateIdentitySHA256:
		return rec, errors.New("report binds different run identities")
	case c.Dataset.SHA256 != rec.DatasetSHA256 || c.Dataset.QuestionsSHA256 != rec.QuestionsSHA256 || c.Dataset.InputSHA256 != rec.InputSHA256:
		return rec, errors.New("report binds a different dataset, questions or inputs")
	case c.Candidate.VariantID != v.ID || c.Reference.VariantID != "":
		return rec, errors.New("report's runs are not the source reference and this variant")
	case producerMismatch != nil:
		return rec, producerMismatch
	case c.Policy.Validate() != nil || c.Policy.SHA256() != c.PolicySHA256 || c.PolicySHA256 != rec.Policy.SHA256 || c.Policy.ID != rec.Policy.ID:
		return rec, errors.New("report's policy does not match the recorded policy")
	}
	if want := c.Policy.Evaluate(c); !reflect.DeepEqual(want, c.Verdict) || want.Status != rec.Verdict {
		return rec, errors.New("the verdict does not follow from the policy and the evidence")
	}
	return rec, nil
}

// LoadCertification reads the report a record names.
func LoadCertification(h home.Home, rec CertificationRecord) (Certification, error) {
	b, err := os.ReadFile(filepath.Join(certDir(h, rec.VariantID), rec.Report))
	if err != nil {
		return Certification{}, err
	}
	return DecodeCertification(b)
}

// CertificationSummary renders a short human-readable account: evidence first,
// then the verdict. It states measurements and the policy's decision only.
func CertificationSummary(w io.Writer, c Certification) {
	f := func(p *float64) string {
		if p == nil {
			return "n/a"
		}
		return fmt.Sprintf("%+.4f", *p)
	}
	fmt.Fprintf(w, "variant        %s (manifest %.12s)\n", c.Variant.ID, c.Variant.ManifestSHA256)
	fmt.Fprintf(w, "source         %s %s@%.12s\n", c.Source.ID, c.Source.Repo, c.Source.Revision)
	fmt.Fprintf(w, "scheme         %s  engine %s %s  recipe %s\n", c.Variant.Scheme, c.Variant.Engine, c.Variant.EngineVersion, c.Variant.Recipe)
	for _, e := range []Execution{c.Reference, c.Candidate} {
		fmt.Fprintf(w, "%-14s device %s  dtype %s  %s\n", e.Role, e.Device, e.DType, e.Quantization)
	}
	fmt.Fprintf(w, "dataset        sha256 %.12s  cases %d  observations %d  labelled=%v\n", c.Dataset.SHA256, c.Dataset.Cases, c.Dataset.Observations, c.Dataset.Labelled)
	fd := c.Fidelity
	fmt.Fprintf(w, "\nfidelity (reference as the authority for fidelity only; not ground truth)\n")
	fmt.Fprintf(w, "  paired %d  unpaired %d\n", fd.Paired, fd.Unpaired)
	fmt.Fprintf(w, "  choice flips   %d  rate %.4f  top choice preserved %.4f  ranking preserved %.4f\n", fd.ChoiceFlips, fd.FlipRate, fd.TopChoicePreservedRate, fd.RankPreservedRate)
	fmt.Fprintf(w, "  confidence     mean change %+.4f  |change| mean %.4f p95 %.4f max %.4f\n", fd.Confidence.MeanSigned, fd.Confidence.Abs.Mean, fd.Confidence.Abs.P95, fd.Confidence.Abs.Max)
	fmt.Fprintf(w, "  probability    mean |dp| %.4f  max |dp| mean %.4f p95 %.4f max %.4f\n", fd.ProbabilityMeanAbs.Mean, fd.ProbabilityMaxAbs.Mean, fd.ProbabilityMaxAbs.P95, fd.ProbabilityMaxAbs.Max)
	fmt.Fprintf(w, "  JS divergence  mean %.5f bits  p95 %.5f  max %.5f\n", fd.JS.Mean, fd.JS.P95, fd.JS.Max)
	fmt.Fprintf(w, "  high-confidence reference decisions (>= %.2f)  %d, flipped %d\n", fd.HighConfidenceRef.Threshold, fd.HighConfidenceRef.N, fd.HighConfidenceRef.Flips)
	if l := c.Labelled; l != nil {
		fmt.Fprintf(w, "\nlabelled quality, candidate minus reference\n")
		fmt.Fprintf(w, "  accuracy %s  macro-F1 %s  ECE %s  Brier %s  NLL %s  high-confidence errors %+d\n",
			f(l.Delta.Accuracy), f(l.Delta.MacroF1), f(l.Delta.ECE), f(l.Delta.Brier), f(l.Delta.NLL), l.Delta.HighConfidenceErrors)
	}
	r := c.Resources
	fmt.Fprintf(w, "\nresources (NOT_CHECKED means the resident did not report it)\n")
	var notChecked []string
	for k, v := range r.Checks {
		if v == FactNotChecked {
			notChecked = append(notChecked, k)
		}
	}
	sort.Strings(notChecked)
	fmt.Fprintf(w, "  request latency p50 %.1f / %.1f ms  p95 %.1f / %.1f ms (reference / candidate)\n",
		r.Reference.RequestLatency.P50, r.Candidate.RequestLatency.P50, r.Reference.RequestLatency.P95, r.Candidate.RequestLatency.P95)
	fmt.Fprintf(w, "  NOT_CHECKED    %s\n", strings.Join(notChecked, ", "))
	fmt.Fprintf(w, "\npolicy         %s (%.12s)\n", c.Policy.ID, c.PolicySHA256)
	for _, cr := range c.Verdict.Criteria {
		switch {
		case !cr.Applied:
			fmt.Fprintf(w, "  not applied  %-36s %s\n", cr.Name, cr.Detail)
		case cr.Pass:
			fmt.Fprintf(w, "  pass         %-36s %s <= %s\n", cr.Name, num(cr.Observed), num(cr.Limit))
		default:
			fmt.Fprintf(w, "  FAIL         %-36s %s > %s\n", cr.Name, num(cr.Observed), num(cr.Limit))
		}
	}
	fmt.Fprintf(w, "verdict        %s\n", c.Verdict.Status)
}

func num(p *float64) string {
	if p == nil {
		return "-"
	}
	return fmt.Sprintf("%.5g", *p)
}
