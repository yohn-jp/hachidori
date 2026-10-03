// Package dashboard serves the host-local Semantic Experiment Workstation:
// Runtime (status and worker lifecycle actions), the Question Workbench, the
// Experiment Runner, Evidence (the Error Explorer), Models (runtime, model and
// residency administration), Forge (the System One variant lifecycle: build,
// preflight/probe, self-contained certification, explicit apply), Diagnostics
// (doctor, worker failures, the SSH reverse-tunnel launcher) and Settings
// (desktop preferences, language, updates, development connections).
//
// It keeps no runtime state of its own. Status is the /v1/status document
// (server.StatusBody), lifecycle actions go through worker.Lifecycle, doctor is
// the same doctor.Run as the CLI, the tunnel is a tunnel.Manager, and the
// workbench and experiment runner are callers of the existing inference API
// at APIAddr (the runner through internal/eval).
package dashboard

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/diagnostics"
	"github.com/yohn-jp/hachidori/internal/eval"
	"github.com/yohn-jp/hachidori/internal/history"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/i18n"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/settings"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tunnel"
	"github.com/yohn-jp/hachidori/internal/ui"
)

// DefaultListen is the default dashboard address (loopback only).
const DefaultListen = "127.0.0.1:7844"

// Lifecycle is the runtime start/stop authority (worker.Lifecycle).
type Lifecycle interface {
	Start() bool
	Stop()
	Restart()
	Running() bool
}

// PathPicker is the optional desktop capability used by workstation file
// workflows. Browser hosted dashboards leave it nil and retain typed paths.
// A dismissed dialog returns ErrPickCancelled.
type PathPicker interface {
	PickOpen(context.Context, string) (string, error)
	PickSave(context.Context, string) (string, error)
	PickFolder(context.Context, string) (string, error)
}

// Config wires the dashboard to the existing host authorities.
type Config struct {
	APIAddr   string               // the loopback inference API address; the workbench calls it
	Status    func() server.Status // the /v1/status document
	Lifecycle Lifecycle
	Doctor    func(out io.Writer) bool // doctor.Run bound to HACHIDORI_HOME
	Tunnel    *tunnel.Manager
	PrefsPath string // non-secret tunnel preferences; empty disables persistence
	// HistoryDir is the experiment history root beneath HACHIDORI_HOME
	// (state/history). Empty disables saving and listing experiment history;
	// experiments then stay memory-only apart from explicit exports.
	HistoryDir string
	// Desktop, when set, adds the desktop preferences panel (start at
	// sign-in, start minimized). It is nil for serve/dashboard.
	Desktop Desktop
	// Settings, when set, adds the Settings workspace's saved runtime
	// defaults (the typed settings authority, internal/settings). Saving
	// them never touches the running runtime.
	Settings Settings
	// Models, when set, adds the Models and Forge workspaces. It is the
	// application's typed maintenance authority (app.Controller over
	// internal/setup); the dashboard never inspects or deletes directories
	// itself. It is nil for serve/dashboard.
	Models Models
	// Residency, when set (with Models), adds the resident-model selection
	// to the Models workspace. It is the settings authority's desired
	// additional-resident set; saving it only stores it.
	Residency Residency
	// Connections, when set, adds the Development Connections profiles to
	// the Settings workspace and makes the Diagnostics tunnel form save
	// through the same profiles. Profiles are persisted by the settings
	// authority; the live tunnel stays exclusively in Tunnel.
	Connections Connections
	// Updates, when set, adds the Settings > Updates surface: the manual,
	// verified executable update (internal/update). Nothing in the dashboard
	// calls it except an explicit operator action; rendering any page,
	// opening Updates and changing the channel use no network.
	Updates Updates
	// WebView2 is the installed WebView2 Runtime version when the desktop
	// shell hosts the dashboard; it is only a fact for the diagnostic bundle.
	WebView2 string
	// PathPicker is present only in the native desktop composition.
	PathPicker PathPicker
	// Variants, when set, adds the System One Forge actions (optimize,
	// preflight, probe, certify, apply, and the advanced experimental
	// activation) to the Forge workspace. The variants themselves, their
	// lifecycle, certification state and the operations' progress are
	// projections of the same inventory and operation state Models reports;
	// Variants only forwards the operator's actions.
	Variants VariantActions
	// Tuning, when set (with Models), adds the Tuning workspace: backend
	// semantic-region analysis, versioned preservation profiles and the
	// handoff of one exact saved profile to the Forge build. It is the
	// typed tuning authority (internal/tuning through the application);
	// the dashboard edits and persists intent through it and never runs an
	// optimizer itself.
	Tuning Tuning
	// Token is the per-process form token. A composition that replaces its
	// dashboard during the process (the desktop, on every runtime rebind)
	// passes one from NewToken so a page loaded before the replacement keeps
	// working; empty means New draws one.
	Token string
}

// NewToken draws a form token.
func NewToken() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b)
}

// VariantActions are the System One forge actions of the maintenance
// authority. Like Models, each returns once the action is accepted and runs in
// the background; its phases, progress and outcome are read from
// Models.State.
type VariantActions interface {
	// ActivateVariant activates the variant of the model and nothing else:
	// the low-level activation, never part of the normal workflow.
	// experimental requests the operator-only launch of a variant without a
	// certification record.
	ActivateVariant(device, model, variant string, experimental bool) error
	Optimize(model, recipe string) error
	// BuildAndEvaluate runs one composed operation from source/profile and
	// semantic evaluation inputs. Its internal phases are reported by State.
	BuildAndEvaluate(ForgeBuildEvaluateRequest) error
	// CertifyVariant starts the self-contained certification of one persisted
	// variant from semantic inputs alone (app.Controller.CertifyVariant): the
	// backend produces and binds both runs itself, and a verdict never
	// activates anything.
	CertifyVariant(CertifyRequest) error
	// Apply is the one backend transaction that makes an accepted variant the
	// serving artifact on an explicit device and proves it
	// (app.Controller.ApplyCertifiedVariant): activation, rebind, READY,
	// provenance, one typed decision, and rollback on failure.
	Apply(device, model, variant string, materialize bool) error
	// Preflight inspects what can be known before the expensive operation
	// kind (materialize, optimize, probe, certify) and records the report. It
	// changes nothing. Probe loads one persisted variant on an explicit device
	// in an isolated worker and asks one typed decision; it is not a
	// certification and touches no resident or activation. Diagnostic returns
	// one stored failure diagnostic by its identity.
	Preflight(kind, model, recipe, variant, device string) error
	Probe(variant, device string) error
	Diagnostic(id string) ([]byte, error)
}

// CertifyRequest is the semantic input of a Forge certification: the variant,
// the evaluation corpus, the Question Definitions, an optional policy and the
// execution choices. It has no reference or candidate run: the backend
// produces both.
type CertifyRequest struct {
	Variant         string
	Device          string // the candidate's device
	ReferenceDevice string // empty: the backend's canonical default
	ReferenceDType  string // empty: the backend's canonical default
	Dataset         string
	Questions       []string
	Policy          string
	Materialize     bool
}

// Models is the explicit model/runtime maintenance authority, and the
// application's own runtime lifecycle. Every method is an operator action;
// none restarts the worker except Start and Restart. Each maintenance action
// returns once it is accepted and runs in the background: its phases,
// progress and outcome are read from State.
type Models interface {
	State() ModelsState
	// StartDesiredState applies one complete serving and resident intent as a
	// single backend operation. The dashboard only forwards the submitted
	// identities and choices; the controller owns reconciliation and rollback.
	StartDesiredState(DesiredStateRequest) error
	Verify(kind, id string) error
	Materialize(device, model string) error
	Repair(device, model string) error
	Activate(device, model string) error // activation only; the worker keeps running as it is
	Remove(kind, id string) error
	// The application's lifecycle. Unlike the worker lifecycle bound at
	// startup, it always acts on the active runtime and model: a start or
	// restart after an activation binds the activation, not the runtime
	// this dashboard was created for.
	Start() error
	Stop() error
	Restart() error
}

// ModelsState is the maintenance authority's view at one instant.
type ModelsState struct {
	Inventory       setup.Inventory
	Err             string                // the inventory could not be read
	Busy            *ModelOp              // an action in flight
	Last            *ModelOp              // the most recently finished action
	Checks          map[string]ModelCheck // last explicit verification per "<kind> <id>"
	RestartRequired bool                  // the running worker predates the current activation or resident selection
	// ResidencyChanged: the desired resident selection differs from the
	// residents the running runtime was started with.
	ResidencyChanged bool
	// Forge is the recorded Forge readiness: preflights, probes, diagnostics.
	Forge ForgeState
}

// Residency reads and stores the desired additional resident models (catalog
// IDs; the active model is always the default resident and is not part of
// it). It is next-start intent only: neither method touches a worker.
type Residency interface {
	Residents() ([]string, error)
	SetResidents([]string) error
}

// ResidentRow is one catalog model in the resident selection.
type ResidentRow struct {
	ID           string
	Default      bool // the active model: always resident, the default route
	Selected     bool // desired as an additional resident at next start
	Bound        bool // a member of the running resident set now
	Materialized bool
}

// ResidencyView is the resident-selection view model. Rows restate the
// desired selection, the inventory and the running set; nothing is decided
// here.
type ResidencyView struct {
	Rows []ResidentRow
	Err  string
}

// ModelCheck is the outcome of the last explicit verification of one artifact.
type ModelCheck struct {
	OK   bool
	Msg  string
	Time time.Time
}

// ModelOp describes one maintenance action: where it is (Phase of Plan), what
// it is doing (Step, Detail) and how far, when that is measurable (Done/Total
// bytes; Total is zero for an indeterminate step), and how it ended.
type ModelOp struct {
	Kind, Device, Model, Target string
	// Desired-state device facts are reported by the Controller. DeviceMode
	// distinguishes an Auto resolution from an operator-pinned override; the
	// actual value is the worker's reported device after execution.
	DeviceMode, RequestedDevice, ResolvedDevice, ActualDevice string
	Plan, Phases                                              []string
	Residents                                                 []string
	Phase                                                     string
	Step, Detail                                              string
	Done, Total                                               int64
	Item, Items                                               int
	Started, Finished                                         time.Time
	Failure                                                   string // empty when it succeeded
	FailurePhase, FailureStep                                 string
	// Resumed is how many of Done a download already held when it began
	// (an interrupted partial being continued); Diagnostic is the identity of
	// the failure diagnostic recorded for a failed Forge operation.
	Resumed    int64
	Diagnostic string
	Log        string // the setup log holding the action's output
}

// Determinate reports whether the step has a measurable total.
func (o ModelOp) Determinate() bool { return o.Total > 0 }

// Percent is the completed share of a determinate step (0 otherwise).
func (o ModelOp) Percent() float64 { return ratio64(o.Done, o.Total) }

// ModelsView is the Models & Runtimes view model.
type ModelsView struct {
	ModelsState
	Devices  []string
	ModelIDs []string
	Runtimes []RuntimeRow
	Models   []ModelRow
	// ActiveModel and ActiveDevice are the activation record: what the next
	// start or restart serves. RunningModel and RunningDevice are what the
	// resident worker was actually started with (from the status authority);
	// they are empty while no worker runs. The two differ exactly while a
	// restart is required.
	ActiveModel, ActiveDevice   string
	RunningModel, RunningDevice string
	Residency                   *ResidencyView // nil unless the resident selection is configured
	// RunningVariant is the variant the resident worker executes now (from
	// the status authority), "" for the source artifact or no worker.
	RunningVariant string
	// Variants are the derived System One variants with their certification
	// state, restating the inventory. VariantControls is set when the
	// variant actions are configured; OptimizeModels are the materialized
	// catalog models that can be optimized and Recipes their recipe names.
	Variants        []VariantRow
	VariantControls bool
	OptimizeModels  []OptimizeTarget
}

// VariantRow is one variant of the inventory with the last explicit
// verification and what it is doing now.
type VariantRow struct {
	setup.VariantEntry
	Check   string
	Running bool // the resident worker executes this variant now
	// Pending: it is the active variant but the running worker started from
	// something else; it applies on restart.
	Pending bool
	// CanActivate: activation is allowed without the experimental
	// escape: an accepted certification record exists.
	CanActivate bool
	// CanExperiment: no record exists, so the explicit operator-only
	// experimental/uncertified launch is offered (never for a rejected one).
	CanExperiment bool
	// Probe is the latest probe of this variant, nil if it was never probed.
	// ProbeStale: it was recorded for another manifest of this ID, so it says
	// nothing about the variant as it is now.
	Probe      *ProbeRow
	ProbeStale bool
}

// OptimizeTarget is a materialized catalog model that can be optimized.
type OptimizeTarget struct {
	ID      string
	Recipes []string
}

// RuntimeRow and ModelRow add the last explicit verification to an entry.
type RuntimeRow struct {
	setup.RuntimeEntry
	Check string
}

type ModelRow struct {
	setup.ModelEntry
	Check   string
	Running bool // the resident worker is serving this model now
}

// Connections stores the named, non-secret Development Connection profiles.
// The dashboard connects them only through tunnel.Manager.
type Connections interface {
	Connections() ([]settings.Connection, error)
	SaveConnection(settings.Connection) error
	RemoveConnection(name string) error
}

// ConnectionItem is one saved profile with its relation to the live tunnel.
type ConnectionItem struct {
	settings.Connection
	Endpoint       string // the HACHIDORI_ENDPOINT value for the development host
	Resolved       tunnel.Spec
	BindEdit       string
	RemotePortEdit int
	LocalPortEdit  int
	Current        bool // the manager's current/last spec is this profile
	Running        bool // ... and its ssh child is alive
}

// ConnectionsView is the Development Connections view model.
type ConnectionsView struct {
	Items []ConnectionItem
	New   tunnel.Spec // defaults for a new profile
	Err   string
}

// Settings reads and stores the saved runtime defaults and the operator UI
// locale selection. The dashboard only renders and forwards the form;
// validation and persistence live behind it.
type Settings interface {
	Defaults() (settings.Defaults, error)
	SetDefaults(settings.Defaults) error
	Locale() (string, error) // "" when no explicit selection was made
	SetLocale(string) error
}

// SettingsView is the Settings workspace's runtime-defaults and language
// view model.
type SettingsView struct {
	Device  string
	Model   string
	Models  []string
	Err     string
	Locale  string // the explicit selection; "" follows the host
	Locales []i18n.Locale
}

// Desktop reads and applies the per-user desktop preferences. The dashboard
// only renders and forwards the form; the mechanism lives behind it.
type Desktop interface {
	Prefs() (startAtSignIn, startMinimized bool, err error)
	Set(startAtSignIn, startMinimized bool) error
}

// DesktopView is the desktop panel's view model.
type DesktopView struct {
	StartAtSignIn  bool
	StartMinimized bool
	Err            string
}

// Dashboard is the HTTP surface. Create it with New.
type Dashboard struct {
	cfg   Config
	token string
	mux   *http.ServeMux

	mu     sync.Mutex
	last   *Action
	doctor DoctorRun

	exp     experiments
	errs    explorer
	hist    *history.Store // nil when Config.HistoryDir is empty or unusable
	histErr string
	// recent is the in-process catalog of resources the operator used, by
	// kind (rememberResource); guarded by mu.
	recent map[string][]string
}

// Action is the visible outcome of the last state-changing request.
type Action struct {
	Time    time.Time
	Name    string
	OK      bool
	Message string
}

// DoctorRun is the state of the most recent doctor run.
type DoctorRun struct {
	Running  bool
	Started  time.Time
	Finished time.Time
	OK       bool
	Output   string
}

// Prefs is everything the dashboard persists: the last tunnel form values.
// It has no field for keys, passwords or any other secret.
type Prefs struct {
	Tunnel tunnel.Spec `json:"tunnel"`
}

//go:embed page.html workbench.html experiments.html errors.html updates.html models.html forge.html tuning.html
var pageFS embed.FS

// pageBase parses the workstation templates once; "t" is the catalog lookup,
// bound per locale in pages.
var pageBase = template.Must(template.New("page.html").Funcs(template.FuncMap{
	"t":             i18n.English.T,
	"get":           get,
	"resourceInput": resourceInput,
	"disclosure":    func(id, label string) DisclosureProjection { return DisclosureProjection{ID: id, Label: label} },
	"mib":           mib,
	"ms":            func(v float64) string { return strconv.FormatFloat(v, 'f', 1, 64) + " ms" },
	"when":          when,
	"alerts":        alerts,
	"tone":          stateTone,
	"gpuMem":        gpuMem,
	"memPressure":   memPressureOf,
	"runtimeAction": runtimeActionOf,
	"ratio":         ratio,
	"errTotal":      errTotal,
	"uptime":        uptime,
	"doctorOut":     doctorOut,
	"pct":           func(p float64) string { return strconv.FormatFloat(100*p, 'f', 1, 64) + "%" },
	"prob":          func(p float64) string { return strconv.FormatFloat(p, 'f', 4, 64) },
	"distOf":        distOf,
	"perQ":          perQuestion,
	"f4":            func(v float64) string { return strconv.FormatFloat(v, 'f', 4, 64) },
	"sd":            func(v float64) string { return signed(v, "") },
	"sms":           func(v float64) string { return signed(v, "ms") },
	"short":         func(s string) string { return s[:min(len(s), 12)] },
	"systemCSS":     ui.CSS,
	"add":           func(a, b int) int { return a + b },
	// long-running work (ops.go)
	"opStages":        opStages,
	"phasePosition":   phasePosition,
	"phaseLabel":      phaseLabel,
	"opPhase":         opPhaseLabel,
	"stepLabel":       stepLabel,
	"workerStages":    workerStages,
	"workerPhaseWord": workerPhaseWord,
	"bytesIn":         bytesIn,
	"since":           since,
	"took":            took,
	"failureOf":       failureOf,
	"objective":       objectiveLabel,
}).ParseFS(pageFS, "page.html", "workbench.html", "experiments.html", "errors.html", "updates.html", "models.html", "forge.html", "tuning.html"))

// pages are the workstation templates for each supported locale. Rendering
// goes through the catalog, never through rewriting rendered HTML.
var pages = func() map[i18n.Locale]*template.Template {
	m := map[i18n.Locale]*template.Template{}
	for _, l := range i18n.Supported {
		m[l] = template.Must(pageBase.Clone()).Funcs(template.FuncMap{"t": l.T})
	}
	return m
}()

// Chrome is what the shared workstation shell needs on every page: the page
// title, the active workspace, the inference API address and the compact
// runtime status restated from the /v1/status document.
type Chrome struct {
	Title       string
	Nav         string // runtime | models | forge | tuning | workbench | experiments | evidence | diagnostics | settings
	Lang        i18n.Locale
	APIAddr     string
	Live        bool // the workspace shows the live-refresh indicator
	HasSettings bool // the Settings workspace is available
	HasModels   bool // the Models and Forge workspaces are available
	HasTuning   bool // the Tuning workspace is available
	Rt          shellStatus
}

// hasTuning: the Tuning workspace needs the tuning authority and the Models
// inventory it lists sources from.
func (c Config) hasTuning() bool { return c.Tuning != nil && c.Models != nil }

func (c Config) hasSettings() bool {
	return c.Settings != nil || c.Desktop != nil || c.Connections != nil || c.Updates != nil
}

// ErrPickCancelled is a PathPicker's answer when the operator dismisses the
// dialog; it is intent, not a failure.
var ErrPickCancelled = errors.New("path selection cancelled")

func pickWasCancelled(err error) bool { return errors.Is(err, ErrPickCancelled) }

// New builds the dashboard.
func New(cfg Config) *Dashboard {
	token := cfg.Token
	if token == "" {
		token = NewToken()
	}
	d := &Dashboard{cfg: cfg, token: token, mux: http.NewServeMux()}
	if cfg.HistoryDir != "" {
		var err error
		if d.hist, err = history.Open(cfg.HistoryDir); err != nil {
			d.histErr = err.Error()
		}
	}
	d.mux.HandleFunc("GET /{$}", d.render("page", "Runtime", "runtime"))
	d.mux.HandleFunc("GET /diagnostics", d.render("diagnostics", "Diagnostics", "diagnostics"))
	d.mux.HandleFunc("GET /live", d.render("live", "Runtime", "runtime"))
	d.mux.HandleFunc("GET /api/status", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, cfg.Status())
	})
	d.mux.HandleFunc("GET /api/tunnel", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, cfg.Tunnel.Status())
	})
	d.mux.HandleFunc("POST /runtime/{op}", d.runtimeOp)
	d.mux.HandleFunc("POST /doctor", d.runDoctor)
	d.mux.HandleFunc("POST /diagnostics/export", d.exportDiagnostics)
	d.mux.HandleFunc("GET /workbench", d.workbenchPage)
	d.mux.HandleFunc("POST /workbench", d.workbenchPost)
	d.mux.HandleFunc("GET /experiments", d.experimentsPage)
	d.mux.HandleFunc("GET /experiments/live", d.experimentsLive)
	d.mux.HandleFunc("POST /experiments/preflight", d.experimentsPreflight)
	d.mux.HandleFunc("POST /experiments/run", d.experimentsRun)
	d.mux.HandleFunc("POST /experiments/export", d.experimentsExport)
	d.mux.HandleFunc("POST /experiments/save", d.experimentsSave)
	d.mux.HandleFunc("POST /experiments/pick", d.experimentsPick)
	d.mux.HandleFunc("POST /history/open", d.historyOpen)
	d.mux.HandleFunc("POST /history/delete", d.historyDelete)
	d.mux.HandleFunc("POST /history/compare", d.historyCompare)
	d.mux.HandleFunc("GET /errors", d.errorsPage)
	d.mux.HandleFunc("POST /errors/open", d.errorsOpen)
	d.mux.HandleFunc("POST /errors/use-experiment", d.errorsUseExperiment)
	d.mux.HandleFunc("POST /errors/export", d.errorsExport)
	d.mux.HandleFunc("POST /errors/pick", d.errorsPick)
	if cfg.hasSettings() {
		d.mux.HandleFunc("GET /settings", d.settingsPage)
	}
	if cfg.Models != nil {
		// The Models and Forge workspaces own their pages and forms. The
		// /settings/... POST and diagnostic routes predate them and stay as
		// compatibility aliases of the same handlers; their outcome is shown
		// in the workspace that now holds the action.
		d.mux.HandleFunc("GET /models", d.modelsPage)
		d.mux.HandleFunc("GET /forge", d.forgePage)
		d.mux.HandleFunc("POST /models/{op}", d.modelsOp)
		d.mux.HandleFunc("POST /forge/{op}", d.variantsOp)
		d.mux.HandleFunc("GET /forge/diagnostics/{id}", d.forgeDiagnostic)
		d.mux.HandleFunc("POST /settings/models/{op}", d.modelsOp)
		d.mux.HandleFunc("POST /settings/variants/{op}", d.variantsOp)
		d.mux.HandleFunc("GET /settings/forge/diagnostics/{id}", d.forgeDiagnostic)
	}
	if cfg.hasTuning() {
		d.mux.HandleFunc("GET /tuning", d.tuningPage)
		d.mux.HandleFunc("POST /tuning/save", d.tuningSave)
		d.mux.HandleFunc("POST /tuning/budget", d.tuningBudget)
		d.mux.HandleFunc("POST /tuning/build", d.tuningBuild)
	}
	if cfg.Models != nil && cfg.Residency != nil {
		d.mux.HandleFunc("POST /models/residents", d.settingsResidents)
		d.mux.HandleFunc("POST /settings/residents", d.settingsResidents)
	}
	if cfg.Updates != nil {
		d.mux.HandleFunc("GET /settings/updates", d.updatesPage)
		d.mux.HandleFunc("POST /settings/updates/channel", d.updatesChannel)
		d.mux.HandleFunc("POST /settings/updates/check", d.updatesCheck)
		d.mux.HandleFunc("POST /settings/updates/download", d.updatesDownload)
		d.mux.HandleFunc("POST /settings/updates/install", d.updatesInstall)
	}
	if cfg.Connections != nil {
		d.mux.HandleFunc("POST /settings/connections/save", d.connectionSave)
		d.mux.HandleFunc("POST /settings/connections/remove", d.connectionRemove)
		d.mux.HandleFunc("POST /settings/connections/connect", d.connectionConnect)
		d.mux.HandleFunc("POST /settings/connections/reconnect", d.connectionReconnect)
	}
	if cfg.Settings != nil {
		d.mux.HandleFunc("POST /settings/defaults", d.settingsDefaults)
		d.mux.HandleFunc("POST /settings/locale", d.settingsLocale)
	}
	if cfg.Desktop != nil {
		d.mux.HandleFunc("POST /desktop/prefs", d.desktopPrefs)
	}
	d.mux.HandleFunc("POST /tunnel/connect", d.connect)
	d.mux.HandleFunc("POST /tunnel/disconnect", d.disconnect)
	return d
}

// Close releases what the dashboard owns: it aborts a running experiment and
// terminates the managed ssh child (if any), waiting for both, so dashboard
// shutdown never leaves them running.
func (d *Dashboard) Close() {
	d.StopExperiment()
	d.cfg.Tunnel.Disconnect()
}

// ServeHTTP enforces the host-local boundary before routing: the Host header
// must name a loopback address (defeats DNS rebinding) and every POST must be
// same-origin and carry the per-process form token (defeats cross-site
// requests from other pages open in the operator's browser).
func (d *Dashboard) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !server.LoopbackHost(r.Host) {
		http.Error(w, "dashboard is host-local: Host must be a loopback address", http.StatusForbidden)
		return
	}
	w.Header().Set("X-Frame-Options", "DENY")
	w.Header().Set("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'; form-action 'self'")
	w.Header().Set("Cache-Control", "no-store")
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		if o := r.Header.Get("Origin"); o != "" && o != "http://"+r.Host {
			http.Error(w, "cross-origin request refused", http.StatusForbidden)
			return
		}
		if s := r.Header.Get("Sec-Fetch-Site"); s != "" && s != "same-origin" && s != "none" {
			http.Error(w, "cross-site request refused", http.StatusForbidden)
			return
		}
		limit := int64(4 << 10)
		if r.URL.Path == "/workbench" {
			limit = workbenchMaxBody
		}
		r.Body = http.MaxBytesReader(w, r.Body, limit)
		if err := r.ParseForm(); err != nil {
			var tooLarge *http.MaxBytesError
			if errors.As(err, &tooLarge) {
				http.Error(w, fmt.Sprintf("request body exceeds %d bytes", limit), http.StatusRequestEntityTooLarge)
				return
			}
		}
		if subtle.ConstantTimeCompare([]byte(r.PostFormValue("token")), []byte(d.token)) != 1 {
			http.Error(w, "missing or stale form token; reload the dashboard", http.StatusForbidden)
			return
		}
	}
	d.mux.ServeHTTP(w, r)
}

type view struct {
	Chrome
	Token   string
	Running bool
	S       server.Status
	Last    *Action
	Doctor  DoctorRun
	Tunnel  tunnel.Status
	Form    tunnel.Spec
	Desktop *DesktopView     // nil unless the desktop shell is hosting the dashboard
	Set     *SettingsView    // nil unless the settings authority is configured
	Models  *ModelsView      // nil unless the model/runtime manager is configured
	Forge   *ForgeView       // the Forge workspace only
	Art     *ArtifactsView   // the Models and Forge workspaces only
	Tuning  *TuningView      // the Tuning workspace only
	Next    *nextStart       // nil unless the model/runtime manager is configured
	Conns   *ConnectionsView // nil unless Development Connections are configured
	Upd     *UpdatesView     // nil unless the update subsystem is configured
	// HasUpdates: the Settings workspace links to Updates.
	HasUpdates bool
	// FormName is the profile the Diagnostics tunnel form saves to; empty
	// when profiles are not configured.
	FormName string
}

// statusView is the part of the runtime view every page's shell needs.
func (d *Dashboard) statusView(title, nav string) view {
	d.mu.Lock()
	last, doc := d.last, d.doctor
	d.mu.Unlock()
	v := view{Chrome: Chrome{Title: title, Nav: nav, Lang: d.locale(), APIAddr: d.cfg.APIAddr, HasSettings: d.cfg.hasSettings(), HasModels: d.cfg.Models != nil, HasTuning: d.cfg.hasTuning()},
		Token: d.token, Running: d.cfg.Lifecycle.Running(), S: d.cfg.Status(), Last: last, Doctor: doc, Tunnel: d.cfg.Tunnel.Status()}
	v.Rt = shellOf(v)
	return v
}

// chrome is the shell for the workspaces that do not render runtime detail.
func (d *Dashboard) chrome(title, nav string) Chrome { return d.statusView(title, nav).Chrome }

func (d *Dashboard) view(title, nav string) view {
	v := d.statusView(title, nav)
	v.Live, v.Form = true, d.formDefaults()
	if d.cfg.Connections != nil {
		v.FormName = d.formName(v.Form)
	}
	if d.cfg.Models != nil {
		v.Next = nextOf(d.cfg.Models.State(), v.S.Runtime)
	}
	if d.cfg.Desktop != nil {
		dv := &DesktopView{}
		var err error
		if dv.StartAtSignIn, dv.StartMinimized, err = d.cfg.Desktop.Prefs(); err != nil {
			dv.Err = err.Error()
		}
		v.Desktop = dv
	}
	return v
}

// formDefaults: the running/last spec, else the first saved profile, else the
// legacy saved form values (read-only when profiles are configured), else
// defaults.
func (d *Dashboard) formDefaults() tunnel.Spec {
	if st := d.cfg.Tunnel.Status(); st.Spec != nil {
		return *st.Spec
	}
	if d.cfg.Connections != nil {
		if cs, err := d.cfg.Connections.Connections(); err == nil && len(cs) > 0 {
			return cs[0].Spec()
		}
	}
	f := tunnel.Spec{RemoteBind: tunnel.DefaultRemoteBind, RemotePort: tunnel.DefaultPort, LocalPort: tunnel.DefaultPort}
	if _, p, err := net.SplitHostPort(d.cfg.APIAddr); err == nil {
		if n, err := strconv.Atoi(p); err == nil {
			f.LocalPort, f.RemotePort = n, n
		}
	}
	if d.cfg.PrefsPath != "" {
		var p Prefs
		if home.ReadJSON(d.cfg.PrefsPath, &p) == nil && p.Tunnel.Validate() == nil {
			f = p.Tunnel
		}
	}
	return f
}

// formName is the profile name the Diagnostics form saves to: the profile
// whose spec the form shows, else "default".
func (d *Dashboard) formName(f tunnel.Spec) string {
	if cs, err := d.cfg.Connections.Connections(); err == nil {
		for _, c := range cs {
			if c.Spec() == f {
				return c.Name
			}
		}
	}
	return defaultConnection
}

const defaultConnection = "default"

func (d *Dashboard) render(name, title, nav string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) { d.renderView(w, name, d.view(title, nav)) }
}

// locale is the operator UI locale: the explicit selection saved by the
// settings authority, else the host locale, else English.
func (d *Dashboard) locale() i18n.Locale {
	saved := ""
	if d.cfg.Settings != nil {
		saved, _ = d.cfg.Settings.Locale()
	}
	return i18n.Resolve(saved, i18n.HostLocales()...)
}

func (d *Dashboard) renderView(w http.ResponseWriter, name string, v any) {
	var buf bytes.Buffer
	if err := pages[d.locale()].ExecuteTemplate(&buf, name, v); err != nil {
		http.Error(w, "render: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = buf.WriteTo(w)
}

// remember records the outcome of an explicit operator action; the next
// workspace render shows it once.
func (d *Dashboard) remember(name string, err error, okMsg string) {
	a := &Action{Time: time.Now(), Name: name, OK: err == nil, Message: okMsg}
	if err != nil {
		a.Message = err.Error()
	}
	d.mu.Lock()
	d.last = a
	d.mu.Unlock()
}

func (d *Dashboard) done(w http.ResponseWriter, r *http.Request, name string, err error, okMsg string) {
	d.remember(name, err, okMsg)
	dest := returnTo(r.URL.Path)
	switch r.PostFormValue("return") {
	case "settings":
		dest = "/settings"
	case "models":
		dest = "/models"
	case "forge":
		dest = "/forge"
	case "tuning":
		dest = "/tuning"
	}
	http.Redirect(w, r, dest, http.StatusSeeOther)
}

// returnTo is the workspace an action's outcome is shown in: lifecycle
// actions return to Runtime, model/runtime administration to Models, the
// System One variant actions to Forge, doctor and tunnel setup to Diagnostics.
func returnTo(path string) string {
	switch {
	case strings.HasPrefix(path, "/runtime/"):
		return "/"
	case strings.HasPrefix(path, "/models/") || strings.HasPrefix(path, "/settings/models/") || path == "/settings/residents":
		return "/models"
	case strings.HasPrefix(path, "/forge/") || strings.HasPrefix(path, "/settings/variants/"):
		return "/forge"
	case strings.HasPrefix(path, "/tuning/"):
		return "/tuning"
	}
	if strings.HasPrefix(path, "/settings/updates/") {
		return "/settings/updates"
	}
	if strings.HasPrefix(path, "/settings/") {
		return "/settings"
	}
	return "/diagnostics"
}

func (d *Dashboard) runtimeOp(w http.ResponseWriter, r *http.Request) {
	lc := d.cfg.Lifecycle
	switch op := r.PathValue("op"); op {
	case "start":
		if m := d.cfg.Models; m != nil {
			d.done(w, r, "start runtime", m.Start(), "worker starting; READY once the model is loaded and warmed up")
		} else if lc.Start() {
			d.done(w, r, "start runtime", nil, "worker starting; READY once the model is loaded and warmed up")
		} else {
			d.done(w, r, "start runtime", fmt.Errorf("runtime is already running"), "")
		}
	case "stop":
		if m := d.cfg.Models; m != nil {
			d.done(w, r, "stop runtime", m.Stop(), "worker stopped; the API stays bound and reports not ready")
		} else {
			lc.Stop()
			d.done(w, r, "stop runtime", nil, "worker stopped; the API stays bound and reports not ready")
		}
	case "restart":
		if m := d.cfg.Models; m != nil {
			// With the application hosted, every lifecycle action is the
			// application's: only it binds the current activation, so a
			// restart after an activation (or after a failure) starts the
			// activated model, never the runtime this dashboard was
			// created for.
			d.done(w, r, "restart runtime", m.Restart(), "runtime restarting on the active runtime/model")
			return
		}
		lc.Restart()
		d.done(w, r, "restart runtime", nil, "worker restarting")
	default:
		http.NotFound(w, r)
	}
}

// desktopPrefs applies the desktop preferences form. Unchecked boxes are
// absent from the form, so the posted state is the complete desired state.
func (d *Dashboard) desktopPrefs(w http.ResponseWriter, r *http.Request) {
	err := d.cfg.Desktop.Set(r.PostFormValue("start_at_sign_in") == "1", r.PostFormValue("start_minimized") == "1")
	d.done(w, r, "desktop preferences", err, "saved")
}

// settingsPage renders the Settings workspace: the desktop preferences (through
// the existing desktop authority) and the saved runtime defaults.
func (d *Dashboard) settingsPage(w http.ResponseWriter, r *http.Request) {
	v := d.view("Settings", "settings")
	v.Live = false
	v.HasUpdates = d.cfg.Updates != nil
	if d.cfg.Settings != nil {
		sv := &SettingsView{Models: settings.Models(), Locales: i18n.Supported}
		def, err := d.cfg.Settings.Defaults()
		if err != nil {
			sv.Err = err.Error()
		}
		sv.Locale, _ = d.cfg.Settings.Locale()
		sv.Device, sv.Model = def.Device, def.Model
		v.Set = sv
	}
	if d.cfg.Connections != nil {
		v.Conns = d.connectionsView(v.Tunnel)
	}
	d.renderView(w, "settings", v)
}

func (d *Dashboard) modelsView(v view) *ModelsView {
	st := d.cfg.Models.State()
	mv := &ModelsView{ModelsState: st, Devices: setup.Devices}
	if a := st.Inventory.Active; a != nil {
		mv.ActiveDevice = a.Device
	}
	if v.Running {
		mv.RunningModel, mv.RunningDevice = v.S.Runtime.ModelID, v.S.Runtime.Device
	}
	check := func(kind, id string) string {
		c, ok := st.Checks[kind+" "+id]
		switch {
		case !ok:
			return ""
		case c.OK:
			return "verified " + when(c.Time)
		}
		return "failed " + when(c.Time) + ": " + c.Msg
	}
	for _, r := range st.Inventory.Runtimes {
		mv.Runtimes = append(mv.Runtimes, RuntimeRow{r, check(setup.KindRuntime, r.ID)})
	}
	for _, m := range st.Inventory.Models {
		mv.ModelIDs = append(mv.ModelIDs, m.ID)
		if m.Active {
			mv.ActiveModel = m.ID
		}
		mv.Models = append(mv.Models, ModelRow{m, check(setup.KindModel, m.ID), m.ID == mv.RunningModel})
	}
	if v.Running {
		if rv := v.S.Runtime.Variant; rv != nil {
			mv.RunningVariant = rv.ID
		}
	}
	mv.VariantControls = d.cfg.Variants != nil
	for _, e := range st.Inventory.Variants {
		row := VariantRow{VariantEntry: e, Check: check(setup.KindVariant, e.ID), Running: v.Running && e.ID == mv.RunningVariant}
		row.Pending = e.Active && st.RestartRequired && !row.Running
		row.CanActivate = e.Problem == "" && e.SourceMaterialized && e.Certification == eval.StateAccepted
		row.CanExperiment = e.Problem == "" && e.SourceMaterialized && e.Certification == eval.StateUncertified
		for i := range st.Forge.Probes {
			if p := &st.Forge.Probes[i]; p.Variant == e.ID {
				row.Probe, row.ProbeStale = p, p.ManifestSHA256 != e.ManifestSHA256
				break // newest first
			}
		}
		mv.Variants = append(mv.Variants, row)
	}
	if mv.VariantControls {
		for _, m := range st.Inventory.Models {
			if m.Materialized && setup.SupportsVariants(home.ModelManifest{Provider: m.Provider}) {
				mv.OptimizeModels = append(mv.OptimizeModels, OptimizeTarget{ID: m.ID, Recipes: optimize.RecipeNames(m.ID)})
			}
		}
	}
	if d.cfg.Residency != nil {
		mv.Residency = d.residencyView(v, mv)
	}
	return mv
}

// variantsOp forwards one explicit System One variant action to the
// maintenance authority. Normal Build & evaluate submits semantic intent in
// one composed call; the other cases preserve existing action aliases.
func (d *Dashboard) variantsOp(w http.ResponseWriter, r *http.Request) {
	va := d.cfg.Variants
	if va == nil {
		http.NotFound(w, r)
		return
	}
	f := func(k string) string { return strings.TrimSpace(r.PostFormValue(k)) }
	switch op := r.PathValue("op"); op {
	case "pick":
		d.forgePick(w, r)
	case "build-evaluate":
		req, err := forgeBuildEvaluateRequest(r)
		if err == nil {
			err = va.BuildAndEvaluate(req)
		}
		if err == nil {
			d.rememberResource(resDataset, req.Dataset)
			d.rememberResource(resQuestions, strings.Join(req.Questions, "\n"))
			d.rememberResource(resPolicy, req.Policy)
		}
		d.done(w, r, "build and evaluate "+f("source"), err,
			"started; the backend builds the selected profile, evaluates that candidate and records its evidence. Nothing is applied")
	case "activate":
		if d.forgeBlocked(f("variant")) {
			d.done(w, r, "activate variant "+f("variant"), errOutsideEnvelope, "")
			return
		}
		d.done(w, r, "activate variant "+f("variant"), va.ActivateVariant(f("device"), f("model"), f("variant"), f("experimental") == "1"),
			"started; once it finishes, a running worker keeps what it started with until you restart it")
	case "optimize":
		d.done(w, r, "optimize "+f("model"), va.Optimize(f("model"), f("recipe")),
			"started; the source and the active runtime are not touched, and nothing is selectable until the build is verified and published")
	case "certify":
		req, err := certifyRequest(r)
		if err == nil {
			err = va.CertifyVariant(req)
		}
		d.done(w, r, "certify "+f("variant"), err,
			"started; the backend runs the reference and the variant itself, and the evidence is recorded whatever the verdict. Nothing is activated")
	case "apply":
		if d.forgeBlocked(f("variant")) {
			d.done(w, r, "apply "+f("variant"), errOutsideEnvelope, "")
			return
		}
		d.done(w, r, "apply "+f("variant"), va.Apply(f("device"), f("model"), f("variant"), f("materialize") == "1"),
			"started; one transaction activates the certified variant, rebinds the runtime and proves it serves, and restores the previous target if anything fails")
	case "preflight":
		d.done(w, r, "preflight "+f("kind"), va.Preflight(f("kind"), f("model"), f("recipe"), f("variant"), f("device")),
			"started; it only inspects and records a report, and starts, downloads and changes nothing")
	case "probe":
		d.done(w, r, "probe "+f("variant"), va.Probe(f("variant"), f("device")),
			"started; the variant loads in its own worker and is stopped afterwards. It is not a certification and no resident, activation or route changes")
	default:
		http.NotFound(w, r)
	}
}

// certifyRequest reads the certification form. Like the Experiments runner, the
// dashboard forwards only explicit absolute local paths (absPath): it never
// resolves one against its own working directory.
func certifyRequest(r *http.Request) (CertifyRequest, error) {
	f := func(k string) string { return strings.TrimSpace(r.PostFormValue(k)) }
	req := CertifyRequest{Variant: f("variant"), Device: f("device"), ReferenceDevice: f("reference_device"), ReferenceDType: f("reference_dtype"),
		Materialize: f("materialize") == "1"}
	var err error
	if req.Dataset, err = absPath(f("dataset"), "evaluation dataset"); err != nil {
		return req, err
	}
	for _, q := range lines(r.PostFormValue("questions")) {
		p, err := absPath(q, "Question Definition")
		if err != nil {
			return req, err
		}
		req.Questions = append(req.Questions, p)
	}
	if pol := f("policy"); pol != "" {
		if req.Policy, err = absPath(pol, "policy"); err != nil {
			return req, err
		}
	}
	return req, nil
}

// lines splits a textarea into its non-empty trimmed lines.
func lines(s string) []string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if l = strings.TrimSpace(l); l != "" {
			out = append(out, l)
		}
	}
	return out
}

// residencyView restates the desired selection beside the inventory and the
// running set (the status document's residents, or the one running model).
func (d *Dashboard) residencyView(v view, mv *ModelsView) *ResidencyView {
	rv := &ResidencyView{}
	want, err := d.cfg.Residency.Residents()
	if err != nil {
		rv.Err = err.Error()
	}
	bound := map[string]bool{}
	for _, r := range v.S.Residents {
		bound[r.Model] = true
	}
	if len(bound) == 0 && mv.RunningModel != "" {
		bound[mv.RunningModel] = true
	}
	for _, m := range mv.Models {
		rv.Rows = append(rv.Rows, ResidentRow{ID: m.ID, Default: m.Active, Selected: !m.Active && slices.Contains(want, m.ID),
			Bound: bound[m.ID], Materialized: m.Materialized})
	}
	return rv
}

// settingsResidents stores the desired resident selection. It only stores:
// nothing is materialized, activated, started, stopped or restarted; the
// selection applies at the next start of the runtime.
func (d *Dashboard) settingsResidents(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		d.done(w, r, "resident models", err, "")
		return
	}
	var ids []string
	for _, id := range r.PostForm["resident"] {
		ids = append(ids, strings.TrimSpace(id))
	}
	d.done(w, r, "resident models", d.cfg.Residency.SetResidents(ids), "saved; applies on the next start. Running workers are unchanged until you restart the runtime")
}

// modelsOp forwards one explicit Models & Runtimes action to the maintenance
// authority. The form names only catalog identities; the authority refuses
// anything else, the active artifacts and any path outside HACHIDORI_HOME.
func (d *Dashboard) modelsOp(w http.ResponseWriter, r *http.Request) {
	m := d.cfg.Models
	device, model := strings.TrimSpace(r.PostFormValue("device")), strings.TrimSpace(r.PostFormValue("model"))
	kind, id := strings.TrimSpace(r.PostFormValue("kind")), strings.TrimSpace(r.PostFormValue("id"))
	switch op := r.PathValue("op"); op {
	case "desired-state":
		intent, err := desiredStateRequest(r, m.State().Inventory)
		if err == nil {
			err = m.StartDesiredState(intent)
		}
		d.done(w, r, "desired state", err, "started; the Controller is reconciling the requested execution target and resident set")
	case "verify":
		d.done(w, r, "verify "+kind+" "+id, m.Verify(kind, id), "started; the result is shown beside the artifact")
	case "materialize":
		d.done(w, r, "materialize", m.Materialize(device, model), "started; it is not activated until you activate it")
	case "repair":
		d.done(w, r, "repair", m.Repair(device, model), "started; the active state is unchanged unless the rebuild succeeds")
	case "activate":
		d.done(w, r, "activate", m.Activate(device, model), "started; once it finishes, a running worker keeps its current runtime until you restart it")
	case "remove":
		d.done(w, r, "remove "+kind+" "+id, m.Remove(kind, id), "started; the result is shown below")
	case "restart":
		d.done(w, r, "restart runtime", m.Restart(), "runtime restarting on the active runtime/model")
	default:
		http.NotFound(w, r)
	}
}

func (d *Dashboard) connectionsView(st tunnel.Status) *ConnectionsView {
	newProfile := d.formDefaults()
	newProfile.RemoteBind = tunnel.DefaultRemoteBind
	if endpoint, err := tunnel.LocalEndpointFromAddr(d.cfg.APIAddr); err == nil {
		newProfile.RemotePort, newProfile.LocalPort = endpoint.Port, endpoint.Port
	}
	cv := &ConnectionsView{New: newProfile}
	cs, err := d.cfg.Connections.Connections()
	if err != nil {
		cv.Err = err.Error()
	}
	for _, c := range cs {
		cur := st.Spec != nil && *st.Spec == c.Spec()
		resolved := c.Spec()
		bindEdit, remotePortEdit, localPortEdit := c.RemoteBind, c.RemotePort, c.LocalPort
		if c.RemoteBindMode == settings.ConnectionAuto {
			bindEdit = resolved.RemoteBind
		}
		if c.RemotePortMode == settings.ConnectionAuto {
			remotePortEdit = resolved.RemotePort
		}
		if c.LocalPortMode == settings.ConnectionAuto {
			localPortEdit = resolved.LocalPort
		}
		cv.Items = append(cv.Items, ConnectionItem{Connection: c, Endpoint: c.Endpoint(), Resolved: resolved,
			BindEdit: bindEdit, RemotePortEdit: remotePortEdit, LocalPortEdit: localPortEdit,
			Current: cur, Running: cur && st.State == tunnel.StateRunning})
	}
	return cv
}

// settingsDefaults stores the runtime defaults. It only stores: the running
// worker, the active model and setup are not consulted or changed.
func (d *Dashboard) settingsDefaults(w http.ResponseWriter, r *http.Request) {
	err := d.cfg.Settings.SetDefaults(settings.Defaults{
		Device: strings.TrimSpace(r.PostFormValue("device")),
		Model:  strings.TrimSpace(r.PostFormValue("model")),
	})
	d.done(w, r, "runtime defaults", err, "saved; the running runtime is unchanged")
}

// settingsLocale stores the operator UI locale selection. It changes only
// presentation; the next render uses it.
func (d *Dashboard) settingsLocale(w http.ResponseWriter, r *http.Request) {
	d.done(w, r, "language", d.cfg.Settings.SetLocale(strings.TrimSpace(r.PostFormValue("locale"))), "saved")
}

func (d *Dashboard) runDoctor(w http.ResponseWriter, r *http.Request) {
	d.mu.Lock()
	if d.doctor.Running {
		d.mu.Unlock()
		d.done(w, r, "doctor", fmt.Errorf("doctor is already running"), "")
		return
	}
	d.doctor = DoctorRun{Running: true, Started: time.Now()}
	d.mu.Unlock()
	go func() {
		var out bytes.Buffer
		ok := d.cfg.Doctor(&out)
		d.mu.Lock()
		d.doctor = DoctorRun{Started: d.doctor.Started, Finished: time.Now(), OK: ok, Output: out.String()}
		d.mu.Unlock()
	}()
	d.done(w, r, "doctor", nil, "doctor started (it runs its own temporary worker)")
}

// exportDiagnostics writes one local diagnostic bundle beneath
// HACHIDORI_HOME/state/diagnostics. It runs only on this explicit operator
// action, reads nothing but the status document and the bounded worker log
// tail (internal/diagnostics), and never uploads or serves the archive.
func (d *Dashboard) exportDiagnostics(w http.ResponseWriter, r *http.Request) {
	st := d.cfg.Status()
	root := st.Runtime.Home
	if root == "" {
		d.done(w, r, "export diagnostics", fmt.Errorf("no runtime home is active"), "")
		return
	}
	exe, _ := os.Executable()
	path, err := diagnostics.Export(diagnostics.Source{Status: st, Home: root, WebView2: d.cfg.WebView2, Executable: exe},
		filepath.Join(root, "state", "diagnostics"))
	d.done(w, r, "export diagnostics", err, "local diagnostic bundle written: "+path+" (nothing was uploaded)")
}

// parseSpec reads the tunnel form fields.
func parseSpec(r *http.Request) (tunnel.Spec, error) {
	s := tunnel.Spec{Destination: strings.TrimSpace(r.PostFormValue("destination")),
		RemoteBind: strings.TrimSpace(r.PostFormValue("remote_bind"))}
	var err error
	if s.RemotePort, err = strconv.Atoi(strings.TrimSpace(r.PostFormValue("remote_port"))); err != nil {
		return s, fmt.Errorf("remote port: not a number")
	}
	if s.LocalPort, err = strconv.Atoi(strings.TrimSpace(r.PostFormValue("local_port"))); err != nil {
		return s, fmt.Errorf("local port: not a number")
	}
	return s, nil
}

func parseConnection(r *http.Request) (settings.Connection, error) {
	c := settings.Connection{
		Name:        strings.TrimSpace(r.PostFormValue("name")),
		Destination: strings.TrimSpace(r.PostFormValue("destination")),
	}
	mode := strings.TrimSpace(r.PostFormValue("remote_bind_mode"))
	switch mode {
	case "", string(settings.ConnectionPinned): // Empty mode preserves older explicit forms.
		c.RemoteBindMode = settings.ConnectionPinned
		c.RemoteBind = strings.TrimSpace(r.PostFormValue("remote_bind"))
	case string(settings.ConnectionAuto):
		c.RemoteBindMode = settings.ConnectionAuto
	default:
		return c, fmt.Errorf("remote bind mode: expected auto or pinned")
	}
	var err error
	if c.RemotePort, c.RemotePortMode, err = parseConnectionPort(r, "remote port", "remote_port_mode", "remote_port"); err != nil {
		return c, err
	}
	if c.LocalPort, c.LocalPortMode, err = parseConnectionPort(r, "local port", "local_port_mode", "local_port"); err != nil {
		return c, err
	}
	return c, nil
}

func parseConnectionPort(r *http.Request, field, modeField, valueField string) (int, settings.TransportMode, error) {
	mode := strings.TrimSpace(r.PostFormValue(modeField))
	switch mode {
	case string(settings.ConnectionAuto):
		return 0, settings.ConnectionAuto, nil
	case "", string(settings.ConnectionPinned): // Empty mode preserves older explicit forms.
		port, err := strconv.Atoi(strings.TrimSpace(r.PostFormValue(valueField)))
		if err != nil {
			return 0, "", fmt.Errorf("%s: not a number", field)
		}
		return port, settings.ConnectionPinned, nil
	default:
		return 0, "", fmt.Errorf("%s mode: expected auto or pinned", field)
	}
}

func namedConnection(name string, s tunnel.Spec) settings.Connection {
	return settings.Connection{Name: strings.TrimSpace(name), Destination: s.Destination,
		RemoteBind: s.RemoteBind, RemotePort: s.RemotePort, LocalPort: s.LocalPort}
}

// connect is the Diagnostics tunnel form. With profiles configured it connects
// through tunnel.Manager and saves the form as a profile: the same profile
// store and the same manager as the Settings workspace, not a second
// transport path.
func (d *Dashboard) connect(w http.ResponseWriter, r *http.Request) {
	s, err := parseSpec(r)
	if err != nil {
		d.done(w, r, "connect tunnel", err, "")
		return
	}
	var prof settings.Connection
	if d.cfg.Connections != nil {
		prof = namedConnection(r.PostFormValue("name"), s)
		if prof.Name == "" {
			prof.Name = defaultConnection
		}
		if err := prof.Validate(); err != nil {
			d.done(w, r, "connect tunnel", err, "")
			return
		}
	}
	started, err := d.cfg.Tunnel.Connect(s)
	switch {
	case err != nil:
		d.done(w, r, "connect tunnel", err, "")
		return
	case !started:
		d.done(w, r, "connect tunnel", nil, "tunnel already running for this destination; nothing started")
		return
	}
	switch {
	case d.cfg.Connections != nil:
		if err := d.cfg.Connections.SaveConnection(prof); err != nil {
			d.done(w, r, "connect tunnel", fmt.Errorf("ssh started, but saving connection %q failed: %w", prof.Name, err), "")
			return
		}
	case d.cfg.PrefsPath != "":
		if err := home.WriteJSON(d.cfg.PrefsPath, Prefs{Tunnel: s}); err != nil {
			d.done(w, r, "connect tunnel", fmt.Errorf("ssh started, but saving preferences failed: %w", err), "")
			return
		}
	}
	d.done(w, r, "connect tunnel", nil, "ssh started; caller endpoint "+s.CallerEndpoint())
}

// profile finds the saved profile named in the request.
func (d *Dashboard) profile(r *http.Request) (settings.Connection, error) {
	name := strings.TrimSpace(r.PostFormValue("name"))
	cs, err := d.cfg.Connections.Connections()
	if err != nil {
		return settings.Connection{}, err
	}
	for _, c := range cs {
		if c.Name == name {
			return c, nil
		}
	}
	return settings.Connection{}, fmt.Errorf("connection %q does not exist", name)
}

func (d *Dashboard) connectionSave(w http.ResponseWriter, r *http.Request) {
	c, err := parseConnection(r)
	if err == nil {
		err = d.cfg.Connections.SaveConnection(c)
	}
	d.done(w, r, "save connection", err, "saved; nothing was connected")
}

func (d *Dashboard) connectionRemove(w http.ResponseWriter, r *http.Request) {
	c, err := d.profile(r)
	if err == nil {
		if st := d.cfg.Tunnel.Status(); st.State == tunnel.StateRunning && st.Spec != nil && *st.Spec == c.Spec() {
			err = fmt.Errorf("connection %q is connected; disconnect it first", c.Name)
		} else {
			err = d.cfg.Connections.RemoveConnection(c.Name)
		}
	}
	d.done(w, r, "remove connection", err, "removed")
}

func (d *Dashboard) connectionConnect(w http.ResponseWriter, r *http.Request) {
	c, err := d.profile(r)
	if err != nil {
		d.done(w, r, "connect tunnel", err, "")
		return
	}
	started, err := d.cfg.Tunnel.Connect(c.Spec())
	switch {
	case err != nil:
		d.done(w, r, "connect tunnel", err, "")
	case !started:
		d.done(w, r, "connect tunnel", nil, "connection "+c.Name+" is already running; nothing started")
	default:
		d.done(w, r, "connect tunnel", nil, "ssh started; HACHIDORI_ENDPOINT="+c.Endpoint())
	}
}

// connectionReconnect replaces whatever tunnel the manager runs with the
// profile: Disconnect, then Connect, both through tunnel.Manager.
func (d *Dashboard) connectionReconnect(w http.ResponseWriter, r *http.Request) {
	c, err := d.profile(r)
	if err != nil {
		d.done(w, r, "reconnect tunnel", err, "")
		return
	}
	d.cfg.Tunnel.Disconnect()
	if _, err := d.cfg.Tunnel.Connect(c.Spec()); err != nil {
		d.done(w, r, "reconnect tunnel", err, "")
		return
	}
	d.done(w, r, "reconnect tunnel", nil, "ssh restarted; HACHIDORI_ENDPOINT="+c.Endpoint())
}

func (d *Dashboard) disconnect(w http.ResponseWriter, r *http.Request) {
	if d.cfg.Tunnel.Status().State != tunnel.StateRunning {
		d.done(w, r, "disconnect tunnel", fmt.Errorf("no tunnel is running"), "")
		return
	}
	d.cfg.Tunnel.Disconnect()
	d.done(w, r, "disconnect tunnel", nil, "ssh terminated")
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

// get renders one value of a free-form worker map ("-" when absent).
func get(m map[string]any, key string) string {
	v, ok := m[key]
	if !ok || v == nil {
		return "-"
	}
	return fmt.Sprint(v)
}

func mib(m map[string]any, key string) string {
	v, ok := num(m, key)
	if !ok {
		return "-"
	}
	return strconv.FormatFloat(v/(1<<20), 'f', 0, 64) + " MiB"
}

func when(t time.Time) string {
	if t.IsZero() {
		return "-"
	}
	return t.Local().Format("2006-01-02 15:04:05")
}
