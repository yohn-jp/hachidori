package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/diagnostics"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/question"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// The Forge readiness commands: preflight, probe and the failure diagnostics.
// They are the CLI face of the authorities the desktop uses (app.RunPreflight,
// app.Probe, internal/diagnostics); none keeps state of its own.

const forgeCmdUsage = `usage: hachidori forge preflight <materialize|optimize|probe|certify> [flags]
       hachidori forge probe [flags] <variant-id>
       hachidori forge execute [flags] <dataset.jsonl>
       hachidori forge diagnostics <list|show|export> [flags]
`

func cmdForge(args []string) error {
	if len(args) == 0 {
		fmt.Fprint(os.Stderr, forgeCmdUsage)
		return errors.New("forge needs a subcommand")
	}
	switch args[0] {
	case "preflight":
		return forgePreflight(args[1:])
	case "probe":
		return forgeProbe(args[1:])
	case "execute":
		return forgeExecute(args[1:])
	case "diagnostics":
		return forgeDiagnostics(args[1:])
	}
	fmt.Fprint(os.Stderr, forgeCmdUsage)
	return fmt.Errorf("unknown forge subcommand %q", args[0])
}

// forgeRun carries what a failed Forge command needs to leave its diagnostic:
// the operation, the phase and step it was in, and the explicit identities.
type forgeRun struct {
	h       home.Home
	failure app.ForgeFailure
	started time.Time
	phase   string
	step    string
}

func newForgeRun(h home.Home, kind, model, variant, recipe, device string) *forgeRun {
	return &forgeRun{h: h, started: time.Now(),
		failure: app.ForgeFailure{OperationID: fmt.Sprintf("cli-%d", time.Now().UnixMilli()), Kind: kind, Model: model, Variant: variant, Recipe: recipe, Device: device}}
}

// observer prints the real phases and progress to stderr and remembers where
// the operation is, so a failure can name its phase.
func (r *forgeRun) observer() *setup.Observer {
	base := cliObserver(os.Stderr)
	return &setup.Observer{
		OnPhase:    func(p setup.Phase) { r.phase, r.step = string(p), ""; base.OnPhase(p) },
		OnProgress: func(p setup.Progress) { r.step = string(p.Step); base.OnProgress(p) },
	}
}

// log is the operation's log: the terminal and the home's setup log, so the
// diagnostic can quote the tail of this very run.
func (r *forgeRun) log() (io.Writer, func()) {
	f := r.failure
	w, closeLog := app.OpenSetupLog(r.h.Root, f.Kind, f.Device, f.Model, f.Variant)
	return io.MultiWriter(os.Stderr, w), closeLog
}

// fail records the diagnostic of a failed operation and returns err itself:
// the diagnostic and any trouble recording it are reported beside the error,
// never instead of it.
func (r *forgeRun) fail(err error) error {
	if err == nil {
		return nil
	}
	f := r.failure
	f.Phase, f.Step, f.Started, f.Finished = r.phase, r.step, r.started, time.Now()
	wrapped := app.WithForgeDiagnostic(r.h.Root, f, err)
	var de *app.DiagnosticError
	if errors.As(wrapped, &de) {
		switch {
		case de.ID != "":
			fmt.Fprintf(os.Stderr, "diagnostic recorded: %s (inspect: hachidori forge diagnostics show %s)\n", de.ID, de.ID)
		case de.CollectErr != nil:
			fmt.Fprintf(os.Stderr, "no diagnostic could be recorded: %v\n", de.CollectErr)
		}
	}
	return err
}

func signalContext() (context.Context, context.CancelFunc) {
	return signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
}

func forgePreflight(args []string) error {
	if len(args) == 0 {
		fmt.Fprint(os.Stderr, forgeCmdUsage)
		return errors.New("forge preflight needs a kind: materialize, optimize, probe or certify")
	}
	kind := args[0]
	fs := flag.NewFlagSet("forge preflight "+kind, flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	model := fs.String("model", setup.ClefFlash, "catalog model ID (materialize, optimize)")
	recipe := fs.String("recipe", "", "canonical recipe name (optimize; default: the model's first recipe)")
	variant := fs.String("variant", "", "variant ID (probe, certify)")
	device := fs.String("device", "", "device the operation runs on: cuda or cpu (required for materialize, probe and certify; optional for optimize, where it checks the serving device ahead of time). There is no default and no fallback")
	quick := fs.Bool("quick", false, "do not hash the artifacts: their digests are then reported UNKNOWN, not PASS")
	refDType := fs.String("reference-dtype", "", "certify: dtype of the high-precision reference (float32 or bfloat16; default $HACHIDORI_CLEF_DTYPE or the release's own)")
	asJSON := fs.Bool("json", false, "print the machine-readable report")
	fs.Parse(args[1:])
	switch kind {
	case setup.PreflightMaterialize, setup.PreflightOptimize, setup.PreflightProbe, setup.PreflightCertify:
	default:
		return fmt.Errorf("unknown preflight kind %q (materialize, optimize, probe, certify)", kind)
	}
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	p := app.PreflightParams{Kind: kind, Model: *model, Recipe: *recipe, Variant: *variant, Device: *device, Quick: *quick, ReferenceDType: *refDType}
	if kind == setup.PreflightOptimize && p.Recipe == "" {
		p.Recipe = defaultRecipe(p.Model)
	}
	if kind == setup.PreflightProbe || kind == setup.PreflightCertify {
		if p.Variant == "" {
			return fmt.Errorf("forge preflight %s needs -variant", kind)
		}
		p.Model = ""
	}
	ctx, stop := signalContext()
	defer stop()
	rep, err := app.RunPreflight(ctx, h.Root, p, cliObserver(os.Stderr))
	if err != nil {
		return err
	}
	if *asJSON {
		if err := printJSON(rep); err != nil {
			return err
		}
	} else {
		printPreflight(os.Stdout, rep)
	}
	if rep.Blocked() {
		return fmt.Errorf("preflight %s: %d blocker(s); the operation must not start", kind, rep.Counts.Blocker)
	}
	return nil
}

// printPreflight renders the report for a terminal. It only restates the
// typed findings.
func printPreflight(w io.Writer, r setup.PreflightReport) {
	fmt.Fprintf(w, "preflight %s: %s", r.Kind, strings.ToUpper(r.Outcome))
	for _, v := range []string{r.Model, r.Variant, r.Recipe, r.Device} {
		if v != "" {
			fmt.Fprintf(w, " %s", v)
		}
	}
	fmt.Fprintf(w, "\n  %d pass, %d warning, %d blocker, %d unknown\n", r.Counts.Pass, r.Counts.Warning, r.Counts.Blocker, r.Counts.Unknown)
	for _, f := range setup.SortedFindings(r.Findings) {
		fmt.Fprintf(w, "  %-8s %-24s %s\n", strings.ToUpper(string(f.Status)), f.ID, f.Summary)
	}
	for _, n := range r.NotMeasured {
		fmt.Fprintf(w, "  not measured: %s\n", n)
	}
	if r.Outcome != setup.OutcomeReady {
		fmt.Fprintln(w, "  a preflight that is not READY is not a promise that the operation fits or succeeds")
	}
}

// forgeProbe is the persisted-variant smoke: it loads the variant's published
// artifact in an isolated worker, asks one typed decision and tears the worker
// down. It is not a certification and activates nothing.
func forgeProbe(args []string) error {
	fs := flag.NewFlagSet("forge probe", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	device := fs.String("device", "cuda", "device the probe worker runs on: cuda or cpu; never substituted")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: hachidori forge probe [flags] <variant-id>")
		fs.PrintDefaults()
	}
	fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
		return errors.New("usage: hachidori forge probe [flags] <variant-id>")
	}
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	run := newForgeRun(h, app.OpProbe, "", fs.Arg(0), "", *device)
	log, closeLog := run.log()
	defer closeLog()
	fmt.Fprintln(os.Stderr, "== accepted")
	rec, err := app.Probe(ctx, h, app.ProbeParams{Variant: fs.Arg(0), Device: *device}, app.ProbeDeps{}, log, run.observer())
	if err != nil {
		fmt.Fprintln(os.Stderr, "== failed")
		run.failure.Probe = &rec
		return run.fail(err)
	}
	fmt.Fprintln(os.Stderr, "== completed")
	if err := printJSON(rec); err != nil {
		return err
	}
	fmt.Fprintln(os.Stderr, "probe passed: this variant loaded and answered one valid typed decision. It is not a certification; nothing was activated or changed")
	return nil
}

// forgeExecute runs a Forge execution session (app.RunExecution) on an exact
// source or variant and prints the stable reference of the resident run it
// recorded under the home. It is not a certification and activates nothing.
// It does not know the residents of a running Hachidori: the desktop's
// Controller owns quiescing those.
func forgeExecute(args []string) error {
	fs := flag.NewFlagSet("forge execute", flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	device := fs.String("device", "", "device the execution worker runs on: cuda or cpu (required; never substituted)")
	variant := fs.String("variant", "", "execute this persisted variant (default: the pinned source model)")
	model := fs.String("model", setup.ClefFlash, "catalog model ID of the source to execute")
	dtype := fs.String("dtype", "", "source reference dtype: float32 or bfloat16 (required for a source whose provider has a dtype control)")
	var defPaths pathList
	fs.Var(&defPaths, "questions", "Question Definition file or directory resolving question_refs (repeatable)")
	warmup := fs.Int("warmup", 1, "warmup requests excluded from latency")
	passes := fs.Int("passes", 1, "passes over the dataset for latency (the decisions of pass 1 are recorded)")
	high := fs.Float64("high-confidence", eval.DefaultHighConfidence, "confidence threshold for high-confidence errors")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: hachidori forge execute [flags] <dataset.jsonl>")
		fs.PrintDefaults()
	}
	fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
		return errors.New("usage: hachidori forge execute [flags] <dataset.jsonl>")
	}
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	var defs *question.Set
	if len(defPaths) > 0 {
		if defs, err = question.Load(defPaths...); err != nil {
			return err
		}
	}
	cases, sum, labelled, err := eval.LoadAny(fs.Arg(0), defs)
	if err != nil {
		return err
	}
	t := app.ExecutionTarget{Kind: eval.ForgeTargetSource, Model: *model, Device: *device, DType: *dtype}
	if *variant != "" {
		t = app.ExecutionTarget{Kind: eval.ForgeTargetVariant, Variant: *variant, Device: *device}
	}
	ctx, stop := signalContext()
	defer stop()
	log, closeLog := app.OpenSetupLog(h.Root, app.OpExecute, t.Device, t.Model, t.String())
	defer closeLog()
	res, err := app.RunExecution(ctx, h, app.ExecuteParams{Target: t, Input: app.ExecutionInput{Dataset: fs.Arg(0), DatasetSHA256: sum, Cases: cases,
		Labelled: labelled, Warmup: *warmup, Passes: *passes, HighConfidence: *high}}, app.ExecutionDeps{}, io.MultiWriter(os.Stderr, log))
	if err != nil {
		return err
	}
	fmt.Printf("evidence %s\n", res.EvidenceID)
	tg := res.Target
	fmt.Printf("recorded %s %s: %d observations (labelled=%v), %d request errors, device %s dtype %s\n", tg.Kind, tg.Model, res.Observations, labelled, res.ErrorCount, tg.Device, tg.DType)
	if res.ErrorCount > 0 {
		return fmt.Errorf("%d request errors; the run is recorded, and certification will refuse it unless the policy allows them", res.ErrorCount)
	}
	return nil
}

func forgeDiagnostics(args []string) error {
	if len(args) == 0 {
		fmt.Fprint(os.Stderr, forgeCmdUsage)
		return errors.New("forge diagnostics needs list, show or export")
	}
	sub := args[0]
	fs := flag.NewFlagSet("forge diagnostics "+sub, flag.ExitOnError)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	kind := fs.String("kind", "", "only diagnostics of this operation (materialize, setup, optimize, probe, certify)")
	model := fs.String("model", "", "only diagnostics of this catalog model")
	variant := fs.String("variant", "", "only diagnostics of this variant")
	out := fs.String("out", "", "export: write to this file, or into this directory (required)")
	fs.Parse(args[1:])
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	filter := diagnostics.ForgeFilter{Kind: *kind, Model: *model, Variant: *variant}
	switch sub {
	case "list":
		l := diagnostics.ListForge(h.Root, filter)
		if l == nil {
			l = []diagnostics.ForgeSummary{}
		}
		return printJSON(l)
	case "show", "export":
		d, err := selectDiagnostic(h, fs.Arg(0), filter)
		if err != nil {
			return err
		}
		if sub == "show" {
			b, err := diagnostics.FormatForge(d)
			if err != nil {
				return err
			}
			_, err = os.Stdout.Write(b)
			return err
		}
		if *out == "" {
			return errors.New("forge diagnostics export needs -out FILE or DIR")
		}
		p, err := diagnostics.ExportForge(d, *out)
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "diagnostic %s written to %s (local only; it holds identities and bounded redacted evidence, no weights, datasets or credentials)\n", d.ID, p)
		return nil
	}
	fmt.Fprint(os.Stderr, forgeCmdUsage)
	return fmt.Errorf("unknown diagnostics subcommand %q", sub)
}

// selectDiagnostic is the diagnostic named by id, or the latest one matching
// the filter when no id is given.
func selectDiagnostic(h home.Home, id string, filter diagnostics.ForgeFilter) (diagnostics.ForgeDiagnostic, error) {
	if id != "" {
		return diagnostics.LoadForge(h.Root, id)
	}
	d, ok := diagnostics.LatestForge(h.Root, filter)
	if !ok {
		return d, errors.New("no Forge diagnostic is recorded in this home for that selection")
	}
	return d, nil
}

// defaultRecipe is the first canonical recipe of a model, or "" when it has none.
func defaultRecipe(model string) string {
	if names := optimize.RecipeNames(model); len(names) > 0 {
		return names[0]
	}
	return ""
}
