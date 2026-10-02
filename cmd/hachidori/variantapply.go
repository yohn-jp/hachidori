package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// variantApply is `hachidori variant apply`: the CLI face of
// app.Controller.ApplyCertifiedVariant, the one transaction that activates an
// accepted variant, rebinds the serving runtime, waits for READY, proves the
// exact variant executes on the requested device and answers one typed
// decision. It composes a controller of its own for the process: nothing here
// restarts a separate `serve` or desktop process, and the worker it proved is
// stopped when the command ends. What it leaves behind is the verified
// activation record, which the next start serves.
func variantApply(args []string) error {
	return runVariantApply(os.Stdout, os.Stderr, args, nil)
}

// runVariantApply is variantApply with the runtime binding replaceable (nil is
// the production binding of the activation record).
func runVariantApply(stdout, stderr io.Writer, args []string, open app.OpenFunc) error {
	fs := flag.NewFlagSet("variant apply", flag.ContinueOnError)
	fs.SetOutput(stderr)
	homeFlag := fs.String("home", "", "HACHIDORI_HOME (default: $HACHIDORI_HOME)")
	device := fs.String("device", "", "device the variant must serve on: cuda or cpu (required; never substituted)")
	model := fs.String("model", "", "catalog model ID of the variant's source (default: the variant's own source)")
	materialize := fs.Bool("materialize", false, "allow the setup authority to materialize a missing serving runtime or source model first (network)")
	asJSON := fs.Bool("json", false, "print the machine-readable result")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: hachidori variant apply [flags] <variant-id>")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		fs.Usage()
		return errors.New("usage: hachidori variant apply [flags] <variant-id>")
	}
	if *device != "cuda" && *device != "cpu" {
		return fmt.Errorf("variant apply needs -device cuda or -device cpu, got %q (there is no default and no fallback)", *device)
	}
	h, err := home.Resolve(*homeFlag)
	if err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()

	if open == nil {
		if err := os.MkdirAll(h.Path("logs"), 0o755); err != nil {
			return err
		}
		logf, err := os.OpenFile(filepath.Join(h.Path("logs"), "worker.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return err
		}
		defer logf.Close()
		// No resident beside the active model: the saved desktop selection is
		// not this process's to start.
		open = app.ConfiguredRuntime(ctx, logf, worker.DefaultPolicy, func() ([]string, error) { return nil, nil }, nil, nil)
	}
	ctl := app.New(app.Config{Home: h.Root, Open: open})
	defer func() {
		shut, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if err := ctl.Close(shut); err != nil {
			fmt.Fprintln(stderr, "hachidori:", err)
		}
	}()

	phases, cancelSub := ctl.Subscribe()
	defer cancelSub()
	done, printed := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(printed)
		last := ""
		for {
			select {
			case <-done:
				return
			case <-phases:
				if op := ctl.Snapshot().Operation; op != nil && op.Phase != "" && op.Phase != last {
					last = op.Phase
					fmt.Fprintf(stderr, "== %s\n", op.Phase)
				}
			}
		}
	}()

	res, err := ctl.ApplyCertifiedVariant(ctx, app.ApplyParams{Variant: fs.Arg(0), Device: *device, Model: *model, Materialize: *materialize})
	close(done)
	<-printed // the phase printer is finished before anything else is written
	if err != nil {
		var ae *app.ApplyError
		if errors.As(err, &ae) {
			switch {
			case ae.Rollback != nil:
				fmt.Fprintln(stderr, "apply failed and the rollback failed too: the previous serving target was NOT verified restored")
			case ae.RolledBack:
				fmt.Fprintln(stderr, "apply failed; the previous activation was restored and verified")
			case !ae.Mutated:
				fmt.Fprintln(stderr, "apply refused; nothing was changed")
			}
		}
		if id := app.DiagnosticID(err); id != "" {
			fmt.Fprintf(stderr, "diagnostic recorded: %s (inspect: hachidori forge diagnostics show %s)\n", id, id)
		}
		return err
	}
	if *asJSON {
		b, err := json.MarshalIndent(res, "", "  ")
		if err != nil {
			return err
		}
		fmt.Fprintln(stdout, string(b))
		return nil
	}
	fmt.Fprintf(stdout, "applied %s of %s on %s (%s, %s): %d quantized modules, certification %s\n",
		res.Variant, res.Model, res.ReportedDevice, res.Scheme, res.DType, res.QuantizedModules, res.Certification)
	fmt.Fprintf(stdout, "typed decision: %s = %s (confidence %.3f)\n", res.Smoke.Question, res.Smoke.Choice, res.Smoke.Confidence)
	fmt.Fprintln(stdout, "the activation record now names the variant; this command's worker stops when it exits, so start (or restart) `hachidori serve` or the desktop to serve it")
	return nil
}
