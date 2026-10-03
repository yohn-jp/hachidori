package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/tuning"
)

// Candidate reports what the home holds about the candidate built from one
// exact profile and about the accepted baseline it is compared with: the
// active variant of the source, when its latest certification is accepted.
// Every figure is read from an artifact, a probe of the exact manifest or a
// certification record; nothing is estimated, and a figure with no record is
// NOT_CHECKED.
func (s tuningStore) Candidate(p tuning.Profile) (dashboard.TuningCandidate, error) {
	h, err := s.home()
	if err != nil {
		return dashboard.TuningCandidate{}, err
	}
	var out dashboard.TuningCandidate
	bv, bdir, why := acceptedBaseline(h, p.Source)
	out.Baseline, out.BaselineWhy = bv.ID, why
	if bv.ID != "" {
		out.BaselineCanonical = bv.Tuning == nil
		if ev, err := home.ReadTuningEvidence(bdir); err == nil {
			out.BaselineEvidence = &ev
		}
	}
	cv, cdir, ok := tunedVariant(h, p)
	if !ok {
		return out, nil
	}
	out.Variant = cv.ID
	switch ev, err := home.ReadTuningEvidence(cdir); {
	case err == nil:
		out.Evidence = &ev
	case !errors.Is(err, os.ErrNotExist):
		out.EvidenceErr = err.Error()
	}
	if bv.ID == "" || bv.ID == cv.ID {
		if bv.ID == cv.ID {
			out.BaselineWhy = "this candidate is the accepted baseline"
		}
		return out, nil
	}
	base, cand := variantFigures(h, bv, bdir), variantFigures(h, cv, cdir)
	out.Comparable, out.ComparableWhy = base.sameEvaluation(cand)
	for _, f := range []struct{ key, label string }{
		{"accuracy", "Choice accuracy"}, {"fidelity", "Fidelity to the source model"}, {"latency", "Request latency"},
		{"vram", "VRAM"}, {"size", "Artifact size"},
	} {
		out.Figures = append(out.Figures, dashboard.TuningFigure{Key: f.key, Label: f.label, Baseline: base.values[f.key], Candidate: cand.values[f.key], Evaluated: f.key != "size"})
	}
	return out, nil
}

// acceptedBaseline is the active variant of the source if its latest
// certification is accepted, else why there is none.
func acceptedBaseline(h home.Home, source home.VariantSource) (home.VariantManifest, string, string) {
	var a home.Active
	if err := home.ReadJSON(h.Path("state", "active-runtime.json"), &a); err != nil {
		return home.VariantManifest{}, "", "no runtime is active, so there is no accepted baseline"
	}
	if a.ModelID != source.ID || a.Variant == "" {
		return home.VariantManifest{}, "", "the active model is not a variant of " + source.ID + ", so there is no accepted baseline variant"
	}
	v, ok, err := h.LoadVariant(a)
	if err != nil || !ok || v.Source != source {
		return home.VariantManifest{}, "", "the active variant cannot be verified as a variant of this exact source"
	}
	if st := eval.ResolveCertification(h, v); st.State != eval.StateAccepted {
		return home.VariantManifest{}, "", "the active variant " + v.ID + " is not certified accepted (" + string(st.State) + ")"
	}
	return v, h.VariantDir(v.Source.ID, v.ID), ""
}

// figures are the measured figures of one variant and the evaluation they came
// from.
type figures struct {
	values            map[string]dashboard.ImpactValue
	dataset, question string
}

// sameEvaluation reports whether two variants' certification evidence came
// from the same dataset and question identities, the only basis on which
// their accuracy, fidelity, latency and VRAM are comparable.
func (f figures) sameEvaluation(o figures) (bool, string) {
	switch {
	case f.dataset == "" || o.dataset == "":
		return false, "at least one of the two variants has no certification record, so their evaluated figures are not compared"
	case f.dataset != o.dataset || f.question != o.question:
		return false, "the two variants were certified on different datasets or questions, so their evaluated figures are not comparable"
	}
	return true, ""
}

func variantFigures(h home.Home, v home.VariantManifest, dir string) figures {
	none := func(why string) dashboard.ImpactValue {
		return dashboard.ImpactValue{State: dashboard.NotChecked, Basis: why}
	}
	label := "variant " + v.ID
	f := figures{values: map[string]dashboard.ImpactValue{
		"size": none(label + ": its artifact files could not be read"), "accuracy": none(label + " has no certification"),
		"fidelity": none(label + " has no certification"), "latency": none(label + " has no certification or passed probe"),
		"vram": none(label + " has no certification or passed probe"),
	}}
	var total int64
	for rel := range v.Files {
		info, err := os.Stat(filepath.Join(dir, filepath.FromSlash(rel)))
		if err != nil {
			total = -1
			break
		}
		total += info.Size()
	}
	if total > 0 {
		f.values["size"] = dashboard.ImpactValue{State: dashboard.Measured, Value: gib(total), Basis: label + ": bytes of its artifact files"}
	}
	if pr, ok := app.LatestProbe(h, v.ID); ok && pr.VariantManifestSHA256 == v.ManifestSHA256() && pr.Result == "passed" {
		basis := "probe of " + label + " on " + pr.Device + " (one typed decision)"
		if pr.Timing.RequestMS > 0 {
			f.values["latency"] = dashboard.ImpactValue{State: dashboard.Measured, Value: fmt.Sprintf("%.1f ms per request", pr.Timing.RequestMS), Basis: basis}
		}
		if r := pr.Resources; r.VRAMAllocated > 0 || r.VRAMReserved > 0 {
			f.values["vram"] = dashboard.ImpactValue{State: dashboard.Measured, Value: "VRAM reserved " + gib(int64(max(r.VRAMAllocated, r.VRAMReserved))), Basis: basis}
		}
	}
	cs := eval.ResolveCertification(h, v)
	if cs.Record == nil {
		return f
	}
	cert, err := eval.LoadCertification(h, *cs.Record)
	if err != nil {
		return f
	}
	f.dataset, f.question = cert.Dataset.SHA256, cert.Dataset.QuestionsSHA256
	basis := "certification of " + label + " (" + cs.Record.Report + ")"
	if cert.Fidelity.Paired > 0 {
		f.values["fidelity"] = dashboard.ImpactValue{State: dashboard.Measured,
			Value: fmt.Sprintf("%.2f%% choice flips over %d paired observations", 100*cert.Fidelity.FlipRate, cert.Fidelity.Paired), Basis: basis + " against its source"}
	}
	f.values["accuracy"] = none(basis + " has no labelled evidence")
	if l := cert.Labelled; l != nil && l.Candidate.Accuracy != nil {
		f.values["accuracy"] = dashboard.ImpactValue{State: dashboard.Measured,
			Value: fmt.Sprintf("%.4f accuracy over %d labelled observations", *l.Candidate.Accuracy, l.Candidate.N), Basis: basis}
	}
	if lat := cert.Resources.Candidate.RequestLatency; lat.N > 0 {
		f.values["latency"] = dashboard.ImpactValue{State: dashboard.Measured,
			Value: fmt.Sprintf("p50 %.1f ms · p95 %.1f ms over %d requests", lat.P50, lat.P95, lat.N), Basis: basis}
	}
	if m := cert.Resources.Candidate.Memory; m.Available && m.PeakRsrv != nil && *m.PeakRsrv > 0 {
		f.values["vram"] = dashboard.ImpactValue{State: dashboard.Measured, Value: "peak VRAM reserved " + gib(*m.PeakRsrv), Basis: basis}
	}
	return f
}
