package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/yohn-jp/hachidori/internal/client"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/question"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// The System One model forge commands: activate, variant and certify. They
// are the CLI face of the same setup, optimize and eval authorities the
// desktop uses; none of them keeps state of its own.

const forgeUsage = `usage: hachidori variant <list|show|verify|optimize|remove|recipes> [flags]
       hachidori certify <run|evaluate|show> [flags]
       hachidori activate [flags]
`

// cliObserver prints the real phases and progress of an operation to w. A
// percentage is printed only for a step that reported a total.
func cliObserver(w io.Writer) *setup.Observer {
	last := ""
	return &setup.Observer{
		OnPhase: func(p setup.Phase) { fmt.Fprintf(w, "== %s\n", p) },
		OnProgress: func(p setup.Progress) {
			line := fmt.Sprintf("   %s %s", p.Step, p.Detail)
			if p.Items > 0 {
				line += fmt.Sprintf(" (%d of %d)", p.Item, p.Items)
			}
			if p.Determinate() && p.Done == p.Total {
				line += fmt.Sprintf(" done, %d bytes", p.Total)
			} else if !p.Determinate() {
				line += " ..."
			} else {
				return // byte progress is shown only when a step completes: no flood
			}
			if line != last {
				fmt.Fprintln(w, line)
				last = line
			}
		},
	}
}

// cmdActivate makes an already materialized catalog choice active: the same
// operation as Settings, Models & runtimes, Activate. It materializes nothing,
// touches no network and never starts a worker; a running runtime applies the
// change on its next restart.
func cmdActivate(args []string) error {
	fs := flag.NewFlagSet("activate", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	device := fs.String("device", "cuda", "inference device: cuda or cpu")
	model := fs.String("model", setup.DefaultModel, "catalog model ID ("+strings.Join(modelIDs(), ", ")+")")
	variant := fs.String("variant", "", "execute this variant of the model instead of its source artifact (see `hachidori variant list`); requires an accepted certification record")
	experimental := fs.Bool("experimental", false, "with -variant: allow a variant that has no certification record at all; the activation is marked experimental/uncertified in status. A variant with a rejecting record is never activated")
	fs.Parse(args)
	if *experimental && *variant == "" {
		return errors.New("-experimental applies to a variant: name it with -variant")
	}
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	changed, err := setup.ActivateTarget(h, *device, *model, setup.ActivateOptions{Variant: *variant, AllowUncertified: *experimental}, os.Stderr, cliObserver(os.Stderr))
	if err != nil {
		return err
	}
	if changed {
		fmt.Fprintln(os.Stderr, "activation changed; restart the runtime (`hachidori serve`, or Restart in the desktop) to apply it")
	} else {
		fmt.Fprintln(os.Stderr, "already active; nothing changed")
	}
	return nil
}

func cmdVariant(args []string) error {
	if len(args) == 0 {
		fmt.Fprint(os.Stderr, forgeUsage)
		return errors.New("variant needs a subcommand")
	}
	switch args[0] {
	case "list":
		return variantList(args[1:])
	case "show":
		return variantShow(args[1:])
	case "verify":
		return variantVerify(args[1:])
	case "optimize":
		return variantOptimize(args[1:])
	case "remove":
		return variantRemove(args[1:])
	case "recipes":
		for _, m := range setup.Models {
			for _, r := range optimize.RecipeNames(m.ID) {
				fmt.Printf("%s\t%s\n", m.ID, r)
			}
		}
		return nil
	}
	fmt.Fprint(os.Stderr, forgeUsage)
	return fmt.Errorf("unknown variant subcommand %q", args[0])
}

func variantList(args []string) error {
	fs := flag.NewFlagSet("variant list", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	verify := fs.Bool("verify", false, "also hash every artifact (slow)")
	fs.Parse(args)
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	inv := setup.Inspect(h, *verify)
	return printJSON(inv.Variants)
}

func variantShow(args []string) error {
	fs := flag.NewFlagSet("variant show", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	fs.Parse(args)
	if fs.NArg() != 1 {
		return errors.New("usage: hachidori variant show [-home DIR] <variant-id>")
	}
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	_, v, err := setup.FindVariant(h, fs.Arg(0))
	if err != nil {
		return err
	}
	return printJSON(struct {
		Manifest       home.VariantManifest    `json:"manifest"`
		ManifestSHA256 string                  `json:"manifest_sha256"`
		Certification  eval.CertificationState `json:"certification"`
	}{v, v.ManifestSHA256(), eval.ResolveCertification(h, v)})
}

func variantVerify(args []string) error {
	fs := flag.NewFlagSet("variant verify", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	fs.Parse(args)
	if fs.NArg() != 1 {
		return errors.New("usage: hachidori variant verify [-home DIR] <variant-id>")
	}
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	if err := setup.Verify(h, setup.KindVariant, fs.Arg(0), cliObserver(os.Stderr)); err != nil {
		return err
	}
	fmt.Println("verified", fs.Arg(0))
	return nil
}

func variantRemove(args []string) error {
	fs := flag.NewFlagSet("variant remove", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	fs.Parse(args)
	if fs.NArg() != 1 {
		return errors.New("usage: hachidori variant remove [-home DIR] <variant-id>")
	}
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	return setup.Remove(h, setup.KindVariant, fs.Arg(0), cliObserver(os.Stderr))
}

// variantOptimize builds a variant of a materialized catalog model with a
// canonical recipe. It is the one command that may materialize the optimizer
// runtime (the only network access of the optimization lifecycle); the
// optimization itself is offline and cannot be pointed at any model but the
// verified catalog source.
func variantOptimize(args []string) error {
	fs := flag.NewFlagSet("variant optimize", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	model := fs.String("model", setup.ClefFlash, "catalog model ID to optimize")
	recipe := fs.String("recipe", optimize.RecipeClefFlashW4A16, "canonical recipe name (see `hachidori variant recipes`)")
	reproduce := fs.Bool("reproduce", false, "rebuild a contract that already has a variant and compare the artifacts with it; nothing is published, and a difference is reported as an error")
	fs.Parse(args)
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	fmt.Fprintln(os.Stderr, "== accepted")
	res, err := optimize.Build(ctx, h, optimize.Request{Model: *model, Recipe: *recipe, Reproduce: *reproduce}, optimize.Deps{}, os.Stderr, cliObserver(os.Stderr))
	if err != nil {
		fmt.Fprintln(os.Stderr, "== failed")
		return err
	}
	fmt.Fprintln(os.Stderr, "== completed")
	switch {
	case res.Reproduced:
		fmt.Fprintf(os.Stderr, "reproduced: the rebuilt artifacts are byte-identical to variant %s\n", res.Variant.ID)
	case res.Existing:
		fmt.Fprintf(os.Stderr, "variant %s already exists for this contract; nothing was built (use -reproduce to rebuild and compare)\n", res.Variant.ID)
	}
	return printJSON(map[string]any{"variant": res.Variant.ID, "source": res.Variant.Source.ID, "scheme": res.Variant.Weights.Scheme,
		"dir": res.Dir, "files": len(res.Variant.Files), "existing": res.Existing})
}

func cmdCertify(args []string) error {
	if len(args) == 0 {
		fmt.Fprint(os.Stderr, forgeUsage)
		return errors.New("certify needs a subcommand")
	}
	switch args[0] {
	case "run":
		return certifyRun(args[1:])
	case "evaluate":
		return certifyEvaluate(args[1:])
	case "show":
		return certifyShow(args[1:])
	}
	fmt.Fprint(os.Stderr, forgeUsage)
	return fmt.Errorf("unknown certify subcommand %q", args[0])
}

// certifyRun records one resident's pass over a dataset as a
// hachidori.resident-run.v1 file. It is run once against the reference and
// once against the variant, each time against the runtime serving that
// execution; the dataset may carry no labels.
func certifyRun(args []string) error {
	fs := flag.NewFlagSet("certify run", flag.ExitOnError)
	endpoint := fs.String("endpoint", "", "endpoint (default: $HACHIDORI_ENDPOINT or "+client.DefaultEndpoint+")")
	model := fs.String("model", setup.ClefFlash, "catalog model ID of the resident to record; it must already be resident and ready")
	out := fs.String("out", "", "write the resident run to this local file (required)")
	var defPaths pathList
	fs.Var(&defPaths, "questions", "Question Definition file or directory resolving question_refs (repeatable)")
	warmup := fs.Int("warmup", 1, "warmup requests excluded from latency")
	passes := fs.Int("passes", 1, "passes over the dataset for latency (the decisions of pass 1 are recorded)")
	rf := residentFlags{}
	fs.Float64Var(&rf.highConfidence, "high-confidence", eval.DefaultHighConfidence, "confidence threshold for high-confidence errors")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: hachidori certify run [flags] -out run.json <dataset.jsonl>")
		fs.PrintDefaults()
	}
	fs.Parse(args)
	if fs.NArg() != 1 || *out == "" {
		fs.Usage()
		os.Exit(2)
	}
	var defs *question.Set
	if len(defPaths) > 0 {
		var err error
		if defs, err = question.Load(defPaths...); err != nil {
			return err
		}
	}
	cases, sum, labelled, err := eval.LoadAny(fs.Arg(0), defs)
	if err != nil {
		return err
	}
	c := client.New(*endpoint)
	run, err := eval.RunResident(c, cases, sum, labelled, *model, eval.ResidentOptions{
		Options: eval.Options{Warmup: *warmup, Passes: *passes}, HighConfidence: rf.highConfidence})
	if err != nil {
		return err
	}
	run.Endpoint, run.Dataset = c.Endpoint, fs.Arg(0)
	if err := writeJSONFile(*out, run); err != nil {
		return err
	}
	r := run.Run
	fmt.Printf("recorded %s: %d observations (labelled=%v), %d request errors, stable=%v, device %v dtype %v variant %q\n", r.Model, len(r.Observations),
		labelled, r.ErrorCount, r.ResidentStable, r.Identity.Provider["device"], r.Identity.Provider["dtype"], r.Identity.Provider["variant_id"])
	fmt.Printf("resident run written to %s\n", *out)
	if r.ErrorCount > 0 {
		return fmt.Errorf("%d request errors; the run is recorded, and certification will refuse it unless the policy allows them", r.ErrorCount)
	}
	return nil
}

// certifyEvaluate compares a reference run and a variant run and records the
// certification under the home. The evidence is recorded whatever the policy
// decides; a rejecting verdict is reported as an error so scripts can see it.
func certifyEvaluate(args []string) error {
	fs := flag.NewFlagSet("certify evaluate", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	variant := fs.String("variant", "", "variant ID to certify (required)")
	refPath := fs.String("reference", "", "resident run of the source model at high precision (required)")
	candPath := fs.String("candidate", "", "resident run of the variant (required)")
	policyPath := fs.String("policy", "", "certification policy file ("+eval.PolicySchema+"); default is the built-in "+eval.DefaultPolicyID+" profile")
	out := fs.String("out", "", "also write the certification report to this local file")
	fs.Parse(args)
	if *variant == "" || *refPath == "" || *candPath == "" {
		fs.Usage()
		return errors.New("certify evaluate needs -variant, -reference and -candidate")
	}
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	policy := eval.DefaultPolicy()
	if *policyPath != "" {
		if policy, err = eval.LoadPolicy(*policyPath); err != nil {
			return err
		}
	}
	src, v, err := setup.FindVariant(h, *variant)
	if err != nil {
		return err
	}
	ref, err := eval.LoadResidentRun(*refPath)
	if err != nil {
		return err
	}
	cand, err := eval.LoadResidentRun(*candPath)
	if err != nil {
		return err
	}
	cert, err := eval.Certify(eval.CertifyInput{Source: src, Variant: v, Reference: ref, Candidate: cand, Policy: policy})
	if err != nil {
		return err
	}
	rec, err := eval.SaveCertification(h, cert)
	if err != nil {
		return err
	}
	eval.CertificationSummary(os.Stdout, cert)
	if *out != "" {
		if err := writeJSONFile(*out, cert); err != nil {
			return err
		}
	}
	fmt.Printf("certification recorded: %s (report sha256 %.12s)\n", rec.Report, rec.ReportSHA256)
	if cert.Verdict.Status != eval.VerdictAccepted {
		return fmt.Errorf("verdict %s under %s: failed %s (the evidence is recorded; the variant is not activated)", cert.Verdict.Status, cert.Policy.ID, strings.Join(cert.Verdict.Failed(), ", "))
	}
	return nil
}

func certifyShow(args []string) error {
	fs := flag.NewFlagSet("certify show", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	asJSON := fs.Bool("json", false, "print the full report as JSON")
	fs.Parse(args)
	if fs.NArg() != 1 {
		return errors.New("usage: hachidori certify show [-home DIR] [-json] <variant-id>")
	}
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	_, v, err := setup.FindVariant(h, fs.Arg(0))
	if err != nil {
		return err
	}
	st := eval.ResolveCertification(h, v)
	fmt.Printf("variant %s: certification %s\n", v.ID, st.State)
	for _, p := range st.Problems {
		fmt.Printf("  record not trusted: %s\n", p)
	}
	if st.Record == nil {
		return nil
	}
	c, err := eval.LoadCertification(h, *st.Record)
	if err != nil {
		return err
	}
	if *asJSON {
		return printJSON(c)
	}
	eval.CertificationSummary(os.Stdout, c)
	return nil
}

func writeJSONFile(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}
