// Package dashboard serves the host-local Semantic Experiment Workstation:
// Runtime (status and worker lifecycle actions), the Question Workbench, the
// Experiment Runner, Evidence (the Error Explorer) and Diagnostics (doctor,
// worker failures, the SSH reverse-tunnel launcher, desktop preferences).
//
// It keeps no runtime state of its own. Status is the /v1/status document
// (server.StatusBody), lifecycle actions go through worker.Lifecycle, doctor is
// the same doctor.Run as the CLI, the tunnel is a tunnel.Manager, and the
// workbench and experiment runner are callers of the existing inference API
// at APIAddr (the runner through internal/eval).
package dashboard

import (
	"bytes"
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
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/diagnostics"
	"github.com/yohn-jp/hachidori/internal/history"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/settings"
	"github.com/yohn-jp/hachidori/internal/tunnel"
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
	// WebView2 is the installed WebView2 Runtime version when the desktop
	// shell hosts the dashboard; it is only a fact for the diagnostic bundle.
	WebView2 string
}

// Settings reads and stores the saved runtime defaults. The dashboard only
// renders and forwards the form; validation and persistence live behind it.
type Settings interface {
	Defaults() (settings.Defaults, error)
	SetDefaults(settings.Defaults) error
}

// SettingsView is the Settings workspace's runtime-defaults view model.
type SettingsView struct {
	Device string
	Model  string
	Models []string
	Err    string
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

//go:embed page.html workbench.html experiments.html errors.html
var pageFS embed.FS

var page = template.Must(template.New("page.html").Funcs(template.FuncMap{
	"get":       get,
	"mib":       mib,
	"ms":        func(v float64) string { return strconv.FormatFloat(v, 'f', 1, 64) + " ms" },
	"when":      when,
	"alerts":    alerts,
	"gpuMem":    gpuMem,
	"ratio":     ratio,
	"errTotal":  errTotal,
	"uptime":    uptime,
	"doctorOut": doctorOut,
	"pct":       func(p float64) string { return strconv.FormatFloat(100*p, 'f', 1, 64) + "%" },
	"prob":      func(p float64) string { return strconv.FormatFloat(p, 'f', 4, 64) },
	"distOf":    distOf,
	"perQ":      perQuestion,
	"f4":        func(v float64) string { return strconv.FormatFloat(v, 'f', 4, 64) },
	"sd":        func(v float64) string { return signed(v, "") },
	"sms":       func(v float64) string { return signed(v, "ms") },
	"short":     func(s string) string { return s[:min(len(s), 12)] },
}).ParseFS(pageFS, "page.html", "workbench.html", "experiments.html", "errors.html"))

// Chrome is what the shared workstation shell needs on every page: the page
// title, the active workspace, the inference API address and the compact
// runtime status restated from the /v1/status document.
type Chrome struct {
	Title       string
	Nav         string // runtime | workbench | experiments | evidence | diagnostics | settings
	APIAddr     string
	Live        bool // the workspace shows the live-refresh indicator
	HasSettings bool // the Settings workspace is available
	Rt          shellStatus
}

// New builds the dashboard.
func New(cfg Config) *Dashboard {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	d := &Dashboard{cfg: cfg, token: hex.EncodeToString(b), mux: http.NewServeMux()}
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
	d.mux.HandleFunc("POST /history/open", d.historyOpen)
	d.mux.HandleFunc("POST /history/delete", d.historyDelete)
	d.mux.HandleFunc("POST /history/compare", d.historyCompare)
	d.mux.HandleFunc("GET /errors", d.errorsPage)
	d.mux.HandleFunc("POST /errors/open", d.errorsOpen)
	d.mux.HandleFunc("POST /errors/use-experiment", d.errorsUseExperiment)
	d.mux.HandleFunc("POST /errors/export", d.errorsExport)
	if cfg.Settings != nil || cfg.Desktop != nil {
		d.mux.HandleFunc("GET /settings", d.settingsPage)
	}
	if cfg.Settings != nil {
		d.mux.HandleFunc("POST /settings/defaults", d.settingsDefaults)
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
	if !loopbackHost(r.Host) {
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

func loopbackHost(hostport string) bool {
	host := hostport
	if h, _, err := net.SplitHostPort(hostport); err == nil {
		host = h
	}
	host = strings.TrimSuffix(strings.TrimPrefix(host, "["), "]")
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
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
	Desktop *DesktopView  // nil unless the desktop shell is hosting the dashboard
	Set     *SettingsView // nil unless the settings authority is configured
}

// statusView is the part of the runtime view every page's shell needs.
func (d *Dashboard) statusView(title, nav string) view {
	d.mu.Lock()
	last, doc := d.last, d.doctor
	d.mu.Unlock()
	v := view{Chrome: Chrome{Title: title, Nav: nav, APIAddr: d.cfg.APIAddr, HasSettings: d.cfg.Settings != nil || d.cfg.Desktop != nil},
		Token: d.token, Running: d.cfg.Lifecycle.Running(), S: d.cfg.Status(), Last: last, Doctor: doc, Tunnel: d.cfg.Tunnel.Status()}
	v.Rt = shellOf(v)
	return v
}

// chrome is the shell for the workspaces that do not render runtime detail.
func (d *Dashboard) chrome(title, nav string) Chrome { return d.statusView(title, nav).Chrome }

func (d *Dashboard) view(title, nav string) view {
	v := d.statusView(title, nav)
	v.Live, v.Form = true, d.formDefaults()
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

// formDefaults: the running/last spec, else saved prefs, else defaults.
func (d *Dashboard) formDefaults() tunnel.Spec {
	if st := d.cfg.Tunnel.Status(); st.Spec != nil {
		return *st.Spec
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

func (d *Dashboard) render(name, title, nav string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) { d.renderView(w, name, d.view(title, nav)) }
}

func (d *Dashboard) renderView(w http.ResponseWriter, name string, v any) {
	var buf bytes.Buffer
	if err := page.ExecuteTemplate(&buf, name, v); err != nil {
		http.Error(w, "render: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = buf.WriteTo(w)
}

func (d *Dashboard) done(w http.ResponseWriter, r *http.Request, name string, err error, okMsg string) {
	a := &Action{Time: time.Now(), Name: name, OK: err == nil, Message: okMsg}
	if err != nil {
		a.Message = err.Error()
	}
	d.mu.Lock()
	d.last = a
	d.mu.Unlock()
	dest := returnTo(r.URL.Path)
	if r.PostFormValue("return") == "settings" {
		dest = "/settings"
	}
	http.Redirect(w, r, dest, http.StatusSeeOther)
}

// returnTo is the workspace an action's outcome is shown in: lifecycle
// actions return to Runtime, doctor, tunnel and desktop setup to Diagnostics.
func returnTo(path string) string {
	if strings.HasPrefix(path, "/runtime/") {
		return "/"
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
		if lc.Start() {
			d.done(w, r, "start runtime", nil, "worker starting; READY once the model is loaded and warmed up")
		} else {
			d.done(w, r, "start runtime", fmt.Errorf("runtime is already running"), "")
		}
	case "stop":
		lc.Stop()
		d.done(w, r, "stop runtime", nil, "worker stopped; the API stays bound and reports not ready")
	case "restart":
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
	if d.cfg.Settings != nil {
		sv := &SettingsView{Models: settings.Models()}
		def, err := d.cfg.Settings.Defaults()
		if err != nil {
			sv.Err = err.Error()
		}
		sv.Device, sv.Model = def.Device, def.Model
		v.Set = sv
	}
	d.renderView(w, "settings", v)
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

func (d *Dashboard) connect(w http.ResponseWriter, r *http.Request) {
	s := tunnel.Spec{Destination: strings.TrimSpace(r.PostFormValue("destination")),
		RemoteBind: strings.TrimSpace(r.PostFormValue("remote_bind"))}
	var err error
	if s.RemotePort, err = strconv.Atoi(r.PostFormValue("remote_port")); err != nil {
		d.done(w, r, "connect tunnel", fmt.Errorf("remote port: not a number"), "")
		return
	}
	if s.LocalPort, err = strconv.Atoi(r.PostFormValue("local_port")); err != nil {
		d.done(w, r, "connect tunnel", fmt.Errorf("local port: not a number"), "")
		return
	}
	started, err := d.cfg.Tunnel.Connect(s)
	switch {
	case err != nil:
		d.done(w, r, "connect tunnel", err, "")
	case !started:
		d.done(w, r, "connect tunnel", nil, "tunnel already running for this destination; nothing started")
	default:
		if d.cfg.PrefsPath != "" {
			if err := home.WriteJSON(d.cfg.PrefsPath, Prefs{Tunnel: s}); err != nil {
				d.done(w, r, "connect tunnel", fmt.Errorf("ssh started, but saving preferences failed: %w", err), "")
				return
			}
		}
		d.done(w, r, "connect tunnel", nil, "ssh started; caller endpoint "+s.CallerEndpoint())
	}
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
