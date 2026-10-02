package app

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/diagnostics"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// ForgeFailure describes a failed expensive Forge operation for its
// diagnostic. Every field is an identity or a measured fact the operation
// already had; the diagnostic reads nothing else about the caller.
type ForgeFailure struct {
	OperationID string
	Kind        string // OpMaterialize | OpOptimize | OpProbe | OpCertify
	Phase       string // the phase it failed in
	Step        string
	Started     time.Time
	Finished    time.Time
	Model       string
	Variant     string
	Recipe      string
	Device      string
	// Probe is the probe record of a failed probe.
	Probe *ProbeRecord
	// Certify are the explicit inputs of a failed certification.
	Certify *CertifyParams
}

// IsForgeOperation reports whether a failed operation of this kind and model
// leaves a Forge diagnostic: materialization of a System One source,
// optimization, probe and certification.
func IsForgeOperation(kind, model string) bool {
	switch kind {
	case OpOptimize, OpProbe, OpCertify:
		return true
	case OpMaterialize, OpSetup, OpRepair:
		if m, err := setup.LookupModel(model); err == nil {
			return setup.SupportsVariants(m)
		}
	}
	return false
}

// DiagnosticError is an operation's own error together with the diagnostic
// that was recorded for it. Its text and its chain are exactly the original
// error's: the diagnostic and any problem collecting it are attached evidence,
// never a replacement.
type DiagnosticError struct {
	Err error
	// ID is the stable identity of the recorded diagnostic ("" if none could
	// be written).
	ID string
	// CollectErr is why the diagnostic is missing or incomplete.
	CollectErr error
}

func (e *DiagnosticError) Error() string { return e.Err.Error() }
func (e *DiagnosticError) Unwrap() error { return e.Err }

// WithForgeDiagnostic records the Forge diagnostic of a failed operation and
// returns the error to report: the original error, wrapped so that the
// diagnostic identity travels with it. A nil error stays nil. Collecting or
// writing the diagnostic can fail in any way, including a panic; the original
// error is returned either way.
func WithForgeDiagnostic(root string, f ForgeFailure, err error) error {
	if err == nil {
		return nil
	}
	var de *DiagnosticError
	if errors.As(err, &de) {
		return err // already recorded
	}
	id, cerr := RecordForgeFailure(root, f, err)
	return &DiagnosticError{Err: err, ID: id, CollectErr: cerr}
}

// DiagnosticID is the diagnostic recorded for err, if any.
func DiagnosticID(err error) string {
	var de *DiagnosticError
	if errors.As(err, &de) {
		return de.ID
	}
	return ""
}

// RecordForgeFailure builds and stores the bounded, redacted diagnostic of a
// failed Forge operation under the home. It returns the diagnostic's identity,
// or "" and the reason it could not be written. It never panics and never
// returns the operation's own error.
func RecordForgeFailure(root string, f ForgeFailure, cause error) (id string, err error) {
	defer func() {
		if r := recover(); r != nil {
			id, err = "", fmt.Errorf("collecting the diagnostic panicked: %v", r)
		}
	}()
	if root == "" {
		return "", errors.New("no home is selected")
	}
	in, secondary := collectForge(root, f, cause)
	in.Secondary = append(in.Secondary, secondary...)
	now := time.Now()
	d := diagnostics.BuildForge(in, root, now)
	if _, err := diagnostics.SaveForge(root, d); err != nil {
		return "", fmt.Errorf("writing the diagnostic: %w", err)
	}
	return d.ID, nil
}

// collectForge gathers the allowlisted facts from the authorities. A part that
// cannot be gathered is reported as secondary evidence and the rest is kept.
func collectForge(root string, f ForgeFailure, cause error) (in diagnostics.ForgeInput, secondary []string) {
	h := home.Home{Root: root}
	try := func(what string, fn func() error) {
		defer func() {
			if r := recover(); r != nil {
				secondary = append(secondary, fmt.Sprintf("%s: panicked: %v", what, r))
			}
		}()
		if err := fn(); err != nil {
			secondary = append(secondary, what+": "+err.Error())
		}
	}
	in.Err, in.Phase = cause, f.Phase
	in.Operation = diagnostics.ForgeOperation{ID: f.OperationID, Kind: f.Kind, Phase: f.Phase, Step: f.Step}
	if !f.Started.IsZero() {
		in.Operation.Started = f.Started.UTC().Format(time.RFC3339)
		if !f.Finished.IsZero() {
			in.Operation.Finished = f.Finished.UTC().Format(time.RFC3339)
			in.Operation.DurationMS = f.Finished.Sub(f.Started).Milliseconds()
		}
	}
	in.Runtime = diagnostics.ForgeRuntime{Device: f.Device, OS: runtime.GOOS, Arch: runtime.GOARCH}
	if bi, ok := debug.ReadBuildInfo(); ok {
		in.Runtime.Hachidori = bi.Main.Version
	}

	// Identity: the pinned source, the variant and the runtime.
	var model home.ModelManifest
	var haveModel bool
	try("source identity", func() error {
		id := f.Model
		if f.Variant != "" {
			m, v, err := setup.FindVariant(h, f.Variant)
			if err == nil {
				model, haveModel = m, true
				in.Identity.Variant, in.Identity.VariantManifestSHA256, in.Identity.VariantBuildID = v.ID, v.ManifestSHA256(), v.BuildID
				in.Optimization = optimizationOf(v.Recipe, v.Optimizer.Engine, v.Optimizer.Version)
				in.Runtime.Quantization, in.Runtime.DType = v.Weights.Scheme, v.Weights.DType
				in.Runtime.CompressionBackend, in.Runtime.CompressionVersion = v.Optimizer.Engine, v.Optimizer.Version
				in.Runtime.Transformers, in.Runtime.CompressedTensors = v.Optimizer.Versions["transformers"], v.Optimizer.Versions["compressed-tensors"]
				in.Runtime.Torch = v.Optimizer.Versions["torch"]
				in.Identity.Variant = v.ID
			} else {
				in.Identity.Variant = f.Variant
				secondary = append(secondary, "variant: "+err.Error())
			}
		}
		if !haveModel {
			m, err := setup.LookupModel(id)
			if err != nil {
				return err
			}
			model, haveModel = m, true
		}
		src := home.SourceOf(model)
		in.Identity.Model, in.Identity.Provider, in.Identity.SourceRepo = model.ID, model.Provider, model.Repo
		in.Identity.SourceRevision, in.Identity.SourceFilesSHA256 = model.Revision, src.FilesSHA256
		return nil
	})
	if f.Variant == "" && f.Recipe != "" && haveModel {
		try("recipe", func() error {
			r, err := optimize.LookupRecipe(model.ID, f.Recipe)
			if err != nil {
				return err
			}
			in.Optimization = optimizationOf(r, r.Engine, setup.OptimizerEngineVersion)
			in.Runtime.Quantization = r.Scheme
			in.Runtime.CompressionBackend, in.Runtime.CompressionVersion = r.Engine, setup.OptimizerEngineVersion
			return nil
		})
	}

	// Runtime: the versions the runtime manifest verified at materialization.
	try("runtime", func() error {
		var spec home.RuntimeSpec
		var err error
		switch {
		case f.Kind == OpOptimize:
			spec, err = setup.DesiredOptimizer()
		case f.Device != "":
			spec, err = setup.Desired(f.Device)
		default:
			return nil
		}
		if err != nil {
			return err
		}
		in.Identity.Runtime = spec.ID()
		var rm home.RuntimeManifest
		if err := home.ReadJSON(h.Path("runtime", spec.ID(), "manifest.json"), &rm); err != nil {
			return fmt.Errorf("runtime %s manifest: %w", spec.ID(), err)
		}
		in.Runtime.Python = rm.PythonVersion
		for _, d := range rm.Installed {
			name, ver, _ := strings.Cut(d, "==")
			switch strings.ToLower(strings.ReplaceAll(name, "_", "-")) {
			case "torch":
				in.Runtime.Torch = ver
			case "transformers":
				in.Runtime.Transformers = ver
			case "compressed-tensors":
				in.Runtime.CompressedTensors = ver
			case "llmcompressor":
				in.Runtime.CompressionBackend, in.Runtime.CompressionVersion = "llmcompressor", ver
			}
		}
		return nil
	})

	// Resources: what the host reports now, and what the operation measured.
	try("host resources", func() error {
		host := setup.DefaultHost()
		if m := host.Mem(); m.TotalKnown {
			in.Resources.RAMTotal = m.Total
			if m.AvailableKnown {
				in.Resources.RAMAvailable = m.Available
			}
		}
		if free, ok := host.FreeDiskAt(root); ok {
			in.Resources.DiskFree = free
		}
		return nil
	})
	if f.Kind == OpMaterialize || f.Kind == OpSetup || f.Kind == OpRepair {
		try("partial download", func() error {
			for _, e := range setup.Inspect(h, false).Models {
				if e.ID == f.Model || (f.Model == "" && e.ID == setup.DefaultModel) {
					in.Resources.PartialBytes = e.PartialBytes
				}
			}
			return nil
		})
	}
	if p := f.Probe; p != nil {
		x := p.Execution
		in.Runtime.Device, in.Runtime.DType, in.Runtime.DeviceName = firstNonEmpty(x.Device, f.Device), x.DType, x.DeviceName
		in.Runtime.Torch, in.Runtime.TorchCUDA, in.Runtime.Python = firstNonEmpty(x.TorchVersion, in.Runtime.Torch), x.TorchCUDA, firstNonEmpty(x.PythonVersion, in.Runtime.Python)
		in.Runtime.ProviderVersion, in.Runtime.Quantization = x.ProviderVersion, firstNonEmpty(x.Quantization, in.Runtime.Quantization)
		r := p.Resources
		in.Resources.VRAMTotal, in.Resources.VRAMFree, in.Resources.VRAMObserved = r.VRAMTotal, r.VRAMFree, r.VRAMAllocated
		in.Resources.WorkerRSS, in.Resources.WeightBytes, in.Resources.WorkerStopped = r.HostRSSBytes, r.WeightFileBytes, p.TornDown
		in.Resources.LoadMS, in.Resources.WarmupMS, in.Resources.StartupMS, in.Resources.RequestMS = p.Timing.LoadMS, p.Timing.WarmupMS, p.Timing.StartupMS, p.Timing.RequestMS
		in.Resources.WorkerPID, in.Resources.WorkerClass = x.PID, p.WorkerClass
		in.StderrTail = append(in.StderrTail, p.StderrTail...)
	}

	// Certification: identities and digests only.
	if c := f.Certify; c != nil {
		in.Certification = &diagnostics.ForgeCertification{}
		try("certification policy", func() error {
			pol := eval.DefaultPolicy()
			if c.Policy != "" {
				var err error
				if pol, err = eval.LoadPolicy(c.Policy); err != nil {
					return err
				}
			}
			in.Certification.PolicyID, in.Certification.PolicySHA256 = pol.ID, pol.SHA256()
			return nil
		})
		try("reference run identity", func() error {
			r, err := eval.LoadResidentRun(c.Reference)
			if err != nil {
				return err
			}
			in.Certification.DatasetSHA256, in.Certification.ReferenceIdentitySHA256 = r.DatasetSHA256, r.Run.Identity.Digest
			return nil
		})
		try("candidate run identity", func() error {
			r, err := eval.LoadResidentRun(c.Candidate)
			if err != nil {
				return err
			}
			in.Certification.CandidateIdentitySHA256 = r.Run.Identity.Digest
			return nil
		})
	}

	// Preflight: the latest readiness report for this target.
	try("preflight", func() error {
		kind := map[string]string{OpOptimize: setup.PreflightOptimize, OpProbe: setup.PreflightProbe, OpCertify: setup.PreflightCertify,
			OpMaterialize: setup.PreflightMaterialize}[f.Kind]
		if kind == "" {
			return nil
		}
		if r, ok := LatestPreflight(h, kind, in.Identity.Model, f.Variant); ok {
			p := &diagnostics.ForgePreflight{Kind: r.Kind, Outcome: r.Outcome, At: r.CreatedAt}
			for _, fd := range r.Findings {
				if fd.Status != setup.FindingPass {
					p.Findings = append(p.Findings, diagnostics.ForgeFinding{ID: fd.ID, Status: string(fd.Status), Summary: fd.Summary})
				}
			}
			in.Preflight = p
		}
		return nil
	})

	// Evidence: the worker's stderr tail from the error chain, and the tail of
	// this operation's own log section.
	var wf *worker.Failure
	if errors.As(cause, &wf) {
		in.StderrTail = append(in.StderrTail, wf.Stderr...)
		in.Resources.WorkerClass = wf.Class
	}
	try("log tail", func() error {
		lines, err := logTail(SetupLogPath(root), f.Kind, diagnostics.MaxForgeLogLines)
		in.LogTail = lines
		return err
	})
	return in, secondary
}

func firstNonEmpty(vs ...string) string {
	for _, v := range vs {
		if v != "" {
			return v
		}
	}
	return ""
}

func optimizationOf(r home.Recipe, engine, version string) *diagnostics.ForgeOptimization {
	o := &diagnostics.ForgeOptimization{Recipe: r.Name, RecipeSHA256: r.SHA256(), Backend: engine, BackendVersion: version, Scheme: r.Scheme, Algorithm: r.Algorithm}
	for _, p := range r.Preserved {
		o.Preserved = append(o.Preserved, p.Pattern)
	}
	return o
}

// logTail returns the last lines of the setup log's most recent section of
// this kind of operation (sections begin with "== <time> <kind> ..."), at most
// max lines. It reads only the end of the file, never all of it. A missing log
// is not an error: there was nothing to tail.
func logTail(path, kind string, max int) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	const window = 256 << 10
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	off := int64(0)
	if fi.Size() > window {
		off = fi.Size() - window
	}
	if _, err := f.Seek(off, io.SeekStart); err != nil {
		return nil, err
	}
	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		l := sc.Text()
		if strings.HasPrefix(l, "== ") && strings.Contains(l, " "+kind+" ") {
			lines = lines[:0] // a newer section of this kind starts here
		}
		lines = append(lines, l)
	}
	if err := sc.Err(); err != nil {
		return lines, err
	}
	if len(lines) > max {
		lines = lines[len(lines)-max:]
	}
	return lines, nil
}
