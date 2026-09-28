// Package dashboard serves the host-local operator page: runtime/GPU status,
// worker lifecycle actions, doctor, and the SSH reverse-tunnel launcher.
//
// It keeps no runtime state of its own. Status is the /v1/status document
// (server.StatusBody), lifecycle actions go through worker.Lifecycle, doctor is
// the same doctor.Run as the CLI, and the tunnel is a tunnel.Manager.
package dashboard

import (
	"bytes"
	"crypto/rand"
	"crypto/subtle"
	"embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
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
	APIAddr   string               // the loopback inference API address, for display
	Status    func() server.Status // the /v1/status document
	Lifecycle Lifecycle
	Doctor    func(out io.Writer) bool // doctor.Run bound to HACHIDORI_HOME
	Tunnel    *tunnel.Manager
	PrefsPath string // non-secret tunnel preferences; empty disables persistence
}

// Dashboard is the HTTP surface. Create it with New.
type Dashboard struct {
	cfg   Config
	token string
	mux   *http.ServeMux

	mu     sync.Mutex
	last   *Action
	doctor DoctorRun
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

//go:embed page.html
var pageFS embed.FS

var page = template.Must(template.New("page.html").Funcs(template.FuncMap{
	"get":  get,
	"mib":  mib,
	"ms":   func(v float64) string { return strconv.FormatFloat(v, 'f', 1, 64) + " ms" },
	"when": when,
}).ParseFS(pageFS, "page.html"))

// New builds the dashboard.
func New(cfg Config) *Dashboard {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	d := &Dashboard{cfg: cfg, token: hex.EncodeToString(b), mux: http.NewServeMux()}
	d.mux.HandleFunc("GET /{$}", d.render("page"))
	d.mux.HandleFunc("GET /live", d.render("live"))
	d.mux.HandleFunc("GET /api/status", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, cfg.Status())
	})
	d.mux.HandleFunc("GET /api/tunnel", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, cfg.Tunnel.Status())
	})
	d.mux.HandleFunc("POST /runtime/{op}", d.runtimeOp)
	d.mux.HandleFunc("POST /doctor", d.runDoctor)
	d.mux.HandleFunc("POST /tunnel/connect", d.connect)
	d.mux.HandleFunc("POST /tunnel/disconnect", d.disconnect)
	return d
}

// Close releases what the dashboard owns: it terminates the managed ssh child
// (if any) and waits for it, so dashboard shutdown never leaves it running.
func (d *Dashboard) Close() { d.cfg.Tunnel.Disconnect() }

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
		r.Body = http.MaxBytesReader(w, r.Body, 4<<10)
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
	Token   string
	APIAddr string
	Running bool
	S       server.Status
	Last    *Action
	Doctor  DoctorRun
	Tunnel  tunnel.Status
	Form    tunnel.Spec
}

func (d *Dashboard) view() view {
	d.mu.Lock()
	last, doc := d.last, d.doctor
	d.mu.Unlock()
	v := view{Token: d.token, APIAddr: d.cfg.APIAddr, Running: d.cfg.Lifecycle.Running(), S: d.cfg.Status(),
		Last: last, Doctor: doc, Tunnel: d.cfg.Tunnel.Status(), Form: d.formDefaults()}
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

func (d *Dashboard) render(name string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var buf bytes.Buffer
		if err := page.ExecuteTemplate(&buf, name, d.view()); err != nil {
			http.Error(w, "render: "+err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = buf.WriteTo(w)
	}
}

func (d *Dashboard) done(w http.ResponseWriter, r *http.Request, name string, err error, okMsg string) {
	a := &Action{Time: time.Now(), Name: name, OK: err == nil, Message: okMsg}
	if err != nil {
		a.Message = err.Error()
	}
	d.mu.Lock()
	d.last = a
	d.mu.Unlock()
	http.Redirect(w, r, "/", http.StatusSeeOther)
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
	v, ok := m[key].(float64)
	if !ok {
		if n, ok := m[key].(int); ok {
			v = float64(n)
		} else {
			return "-"
		}
	}
	return strconv.FormatFloat(v/(1<<20), 'f', 0, 64) + " MiB"
}

func when(t time.Time) string {
	if t.IsZero() {
		return "-"
	}
	return t.Local().Format("2006-01-02 15:04:05")
}
