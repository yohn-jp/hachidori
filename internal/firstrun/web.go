package firstrun

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/subtle"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"html/template"
	"net"
	"net/http"
	"strings"

	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/i18n"
)

//go:embed page.html
var pageSrc string

// pages holds the first-run page once per supported locale; "t" is the
// locale's catalog lookup.
var pages = func() map[i18n.Locale]*template.Template {
	m := map[i18n.Locale]*template.Template{}
	for _, l := range i18n.Supported {
		m[l] = template.Must(template.New("page").Funcs(template.FuncMap{"t": l.T}).Parse(pageSrc))
	}
	return m
}()

// Handler is the loopback HTTP surface of the desktop window. Under /wizard/
// it serves the first-run/recovery screen and its actions; every other path
// belongs to the normal desktop home (the dashboard) once the flow is Done,
// and shows the first-run screen until then. The page is embedded, loads
// nothing remote, and its actions are same-origin POSTs carrying a
// per-process token, like the dashboard's.
type Handler struct {
	flow  *Flow
	home  func() http.Handler // the desktop home; nil result: not bound yet
	token string
	mux   *http.ServeMux

	// Locale resolves the operator UI locale per request; nil is English.
	// The desktop resolves it through the typed settings authority, which
	// exists before any Hachidori home is selected.
	Locale func() i18n.Locale
}

// NewHandler builds the handler. home returns the dashboard handler bound to
// the current runtime, or nil when none is bound.
func NewHandler(f *Flow, home func() http.Handler) *Handler {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	h := &Handler{flow: f, home: home, token: hex.EncodeToString(b), mux: http.NewServeMux()}
	h.mux.HandleFunc("GET /wizard/state", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, http.StatusOK, f.View()) })
	h.mux.HandleFunc("POST /wizard/browse", h.browse)
	h.mux.HandleFunc("POST /wizard/install", h.action(func(r *http.Request) error { return f.Install(r.PostFormValue("device")) }))
	h.mux.HandleFunc("POST /wizard/use-existing", h.action(func(*http.Request) error { return f.UseExisting() }))
	h.mux.HandleFunc("POST /wizard/retry", h.action(func(*http.Request) error { return f.Retry() }))
	h.mux.HandleFunc("POST /wizard/change", h.action(func(*http.Request) error { return f.Change() }))
	h.mux.HandleFunc("GET /{$}", h.render)
	return h
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !loopbackHost(r.Host) {
		http.Error(w, "host-local: Host must be a loopback address", http.StatusForbidden)
		return
	}
	if !strings.HasPrefix(r.URL.Path, "/wizard/") {
		if h.flow.Done() {
			if d := h.home(); d != nil {
				d.ServeHTTP(w, r)
				return
			}
		}
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
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
		if subtle.ConstantTimeCompare([]byte(r.PostFormValue("token")), []byte(h.token)) != 1 {
			http.Error(w, "missing or stale token; reload", http.StatusForbidden)
			return
		}
	}
	h.mux.ServeHTTP(w, r)
}

func (h *Handler) render(w http.ResponseWriter, r *http.Request) {
	loc := i18n.English
	if h.Locale != nil && pages[h.Locale()] != nil {
		loc = h.Locale()
	}
	var buf bytes.Buffer
	data := struct {
		Token string
		Lang  i18n.Locale
		L     map[string]string
	}{h.token, loc, loc.Table(i18n.FirstRunScript)}
	if err := pages[loc].Execute(&buf, data); err != nil {
		http.Error(w, "render: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = buf.WriteTo(w)
}

func (h *Handler) browse(w http.ResponseWriter, r *http.Request) {
	_, err := h.flow.Select(r.Context())
	switch {
	case errors.Is(err, desktop.ErrPickCancelled):
		writeJSON(w, http.StatusOK, h.flow.View())
	case err != nil:
		writeError(w, err)
	default:
		writeJSON(w, http.StatusOK, h.flow.View())
	}
}

func (h *Handler) action(do func(*http.Request) error) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if err := do(r); err != nil {
			writeError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, h.flow.View())
	}
}

func writeError(w http.ResponseWriter, err error) {
	code := http.StatusConflict
	if errors.Is(err, context.Canceled) {
		code = http.StatusRequestTimeout
	}
	writeJSON(w, code, map[string]string{"error": err.Error()})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
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
