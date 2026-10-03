package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/trial"
)

// The tuning trial commands: RAM-resident, ephemeral experiment execution and
// the path from a measured Candidate to a Forge-built Variant. A trial changes
// a resident model in place and records a Candidate with trial Evidence; it
// builds no artifact and certifies nothing. A finalist is built by the normal
// Forge optimizer from the exact plan the candidate recorded, and is then
// certified from a clean load with `hachidori forge certify`.

const forgeTrialUsage = `usage: hachidori forge trial [flags] <dataset.jsonl> <profile-id>...
       hachidori forge candidates [flags]
       hachidori forge finalist [flags] <candidate-id>
`

// parseBytes reads an explicit byte budget: a number with an optional unit
// (B, KiB, MiB, GiB, TiB; KB, MB, GB, TB are the same binary sizes).
func parseBytes(s string) (int64, error) {
	s = strings.TrimSpace(s)
	units := []struct {
		suffix string
		n      int64
	}{{"TiB", 1 << 40}, {"GiB", 1 << 30}, {"MiB", 1 << 20}, {"KiB", 1 << 10}, {"TB", 1 << 40}, {"GB", 1 << 30}, {"MB", 1 << 20}, {"KB", 1 << 10}, {"B", 1}}
	mult := int64(1)
	for _, u := range units {
		if strings.HasSuffix(s, u.suffix) {
			s, mult = strings.TrimSpace(strings.TrimSuffix(s, u.suffix)), u.n
			break
		}
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || f <= 0 {
		return 0, fmt.Errorf("%q is not a positive size (examples: 40GiB, 512MiB, 8589934592)", s)
	}
	return int64(f * float64(mult)), nil
}

// forgeTrial runs one tuning trial session (app.RunTrials) over saved
// layer-wise profiles. Like `forge execute` it does not know the residents of a
// running Hachidori: stop the runtime first, or use the desktop, whose
// controller owns the accelerator for the session.
func forgeTrial(args []string) error {
	fs := flag.NewFlagSet("forge trial", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	device := fs.String("device", "", "device the trial worker runs on: cuda or cpu (required; never substituted)")
	model := fs.String("model", setup.ClefFlash, "catalog model ID of the source to tune")
	budget := fs.String("ram-budget", "", "system RAM the session may use for the canonical source and transformed components, e.g. 40GiB (required; nothing is assumed about this host)")
	reconstruct := fs.Bool("allow-reconstruct", false, "when a delta cannot be applied in place, rebuild the model from RAM (slower; recorded in the evidence) instead of refusing")
	var defPaths pathList
	fs.Var(&defPaths, "questions", "Question Definition file or directory resolving question_refs (repeatable)")
	policy := fs.String("policy", "", "evaluation policy file; default is the built-in profile")
	warmup := fs.Int("warmup", 1, "warmup requests excluded from latency")
	passes := fs.Int("passes", 1, "passes over the dataset for latency (the decisions of pass 1 are recorded)")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: hachidori forge trial [flags] <dataset.jsonl> <profile-id>...")
		fs.PrintDefaults()
	}
	fs.Parse(args)
	if fs.NArg() < 2 {
		fs.Usage()
		return errors.New("usage: hachidori forge trial [flags] <dataset.jsonl> <profile-id>...")
	}
	bytes, err := parseBytes(*budget)
	if err != nil {
		return fmt.Errorf("--ram-budget: %w", err)
	}
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	p := app.TrialParams{Source: *model, Device: *device, Profiles: fs.Args()[1:], Dataset: fs.Arg(0), Questions: defPaths, Policy: *policy,
		Warmup: *warmup, Passes: *passes, BudgetBytes: bytes, AllowReconstruct: *reconstruct}
	log, closeLog := app.OpenSetupLog(h.Root, app.OpTuningTrial, *device, *model, fmt.Sprintf("%d profiles", len(p.Profiles)))
	defer closeLog()
	run, err := app.RunTrials(ctx, h, p, app.TrialDeps{}, io.MultiWriter(os.Stderr, log), cliObserver(os.Stderr))
	printTrialRun(os.Stdout, run)
	return err
}

func printTrialRun(w io.Writer, run app.TrialRun) {
	for _, o := range run.Outcomes {
		if o.Result == nil {
			state := "the model was restored to the previous trial's state"
			if !o.Restored {
				state = "the model state could not be restored"
			}
			fmt.Fprintf(w, "profile %.12s: not measured: %s (%s)\n", o.Profile, o.Failure, state)
			continue
		}
		r := o.Result
		acc := "not checked"
		if r.Measurement.Accuracy != nil {
			acc = fmt.Sprintf("%.4f", *r.Measurement.Accuracy)
		}
		fmt.Fprintf(w, "profile %.12s: trial measured (ephemeral, not certification): accuracy %s, %s assembly %.0f ms, evaluation %.0f ms, total %.0f ms\n",
			o.Profile, acc, r.Assembly.Mode, r.Assembly.AssemblyMS, r.EvaluationMS, r.TotalMS)
		fmt.Fprintf(w, "  replaced %d groups, reused %d; cache %d hits, %d misses; RAM to GPU %d bytes, released %d bytes\n",
			len(r.Assembly.Changed), r.Assembly.Reused, r.Assembly.CacheHits, r.Assembly.CacheMisses, r.Assembly.BytesToGPU, r.Assembly.BytesReleased)
		if o.CandidateID != "" {
			fmt.Fprintf(w, "  candidate %s evidence %.12s\n", o.CandidateID, o.EvidenceID)
		}
		if o.Failure != "" {
			fmt.Fprintf(w, "  %s\n", o.Failure)
		}
	}
	st := run.Session.Cache
	fmt.Fprintf(w, "session: %d trials; RAM %d of %d bytes (%d canonical, %d transformed); %d hits, %d misses, %d evictions; worker start %.0f ms, session %.0f ms\n",
		run.Session.Trials, st.CurrentBytes, st.BudgetBytes, st.CanonicalBytes, st.TransformedBytes, st.Hits, st.Misses, st.Evictions, run.StartMS, run.SessionMS)
}

// forgeCandidates lists the trial candidates recorded for a source.
func forgeCandidates(args []string) error {
	fs := flag.NewFlagSet("forge candidates", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	model := fs.String("model", "", "list only the candidates of this source (default: every source)")
	fs.Parse(args)
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	list, err := trial.List(h, *model)
	if err != nil {
		return err
	}
	if len(list) == 0 {
		fmt.Println("no trial candidates recorded")
		return nil
	}
	for _, c := range list {
		st, err := trial.StatusOf(h, c.ID)
		if err != nil {
			return err
		}
		line := fmt.Sprintf("%s  %s  profile %.12s  plan %.12s  %s (%d measurements)", c.ID, c.Source.ID, c.Profile.ID, c.PlanSHA256, st.Stage(), st.Measurements)
		if len(st.Variants) > 0 {
			line += "  variant " + strings.Join(st.Variants, ",")
		}
		fmt.Println(line)
	}
	return nil
}

// forgeFinalist builds a measured candidate as a normal, immutable Variant: the
// Forge optimizer reproduces the exact resolved plan the candidate recorded and
// refuses a profile that no longer resolves to it. The variant is then
// certified from a clean load with `hachidori forge certify`; nothing is
// applied.
func forgeFinalist(args []string) error {
	fs := flag.NewFlagSet("forge finalist", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: hachidori forge finalist [flags] <candidate-id>")
		fs.PrintDefaults()
	}
	fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
		return errors.New("usage: hachidori forge finalist [flags] <candidate-id>")
	}
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	cand, err := trial.LoadCandidate(h, fs.Arg(0))
	if err != nil {
		return err
	}
	source, err := setup.LookupModel(cand.Source.ID)
	if err != nil {
		return err
	}
	f, err := trial.PrepareFinalist(h, source, cand.ID)
	if err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	run := newForgeRun(h, app.OpOptimize, source.ID, "", f.Request.Recipe, "cpu")
	log, closeLog := run.log()
	defer closeLog()
	res, err := optimize.Build(ctx, h, f.Request, optimize.Deps{}, log, run.observer())
	if err != nil {
		return run.fail(err)
	}
	if err := trial.RecordMaterialization(h, cand.ID, res.Variant, time.Now()); err != nil {
		return fmt.Errorf("variant %s was built but could not be linked to candidate %s: %w", res.Variant.ID, cand.ID, err)
	}
	fmt.Printf("variant %s (plan %.12s of candidate %s)\n", res.Variant.ID, cand.PlanSHA256, cand.ID)
	fmt.Fprintln(os.Stderr, "built, not certified and not applied: hachidori forge certify evaluates it from a clean load")
	return nil
}
