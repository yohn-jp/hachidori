package firstrun

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/worker"
)

func serve(h http.Handler, method, target string, form url.Values, hdr map[string]string) *httptest.ResponseRecorder {
	var body *strings.Reader
	if form != nil {
		body = strings.NewReader(form.Encode())
	} else {
		body = strings.NewReader("")
	}
	r := httptest.NewRequest(method, target, body)
	r.Host = "127.0.0.1:7844"
	if form != nil {
		r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}
	for k, v := range hdr {
		r.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func tokenOf(t *testing.T, h http.Handler) string {
	t.Helper()
	w := serve(h, "GET", "/", nil, nil)
	m := regexp.MustCompile(`const TOKEN = "([0-9a-f]+)"`).FindStringSubmatch(w.Body.String())
	if w.Code != 200 || m == nil {
		t.Fatalf("page %d: token not found", w.Code)
	}
	return m[1]
}

func TestHandlerServesWizardAndProtectsActions(t *testing.T) {
	f := newFixture(t, Plan{Mode: ModeFirstRun})
	var dash http.Handler
	h := NewHandler(f.flow, func() http.Handler { return dash })

	w := serve(h, "GET", "/", nil, nil)
	if w.Code != 200 || !strings.Contains(w.Body.String(), "Where should Hachidori keep its models and runtime?") {
		t.Fatalf("first-run page: %d", w.Code)
	}
	if csp := w.Header().Get("Content-Security-Policy"); !strings.Contains(csp, "default-src 'self'") {
		t.Fatalf("CSP %q", csp)
	}
	// Nothing remote: the page references no external origin.
	if strings.Contains(w.Body.String(), "http://") || strings.Contains(w.Body.String(), "https://") {
		t.Fatal("first-run page references a remote URL")
	}
	// The page never asks for internal paths.
	for _, banned := range []string{"Python", "uv ", "cache", "Hugging"} {
		if strings.Contains(w.Body.String(), banned) {
			t.Errorf("page mentions %q", banned)
		}
	}
	if w = serve(h, "GET", "/other", nil, nil); w.Code != 404 {
		t.Fatalf("unknown path %d", w.Code)
	}

	// Host, token and origin checks.
	r := httptest.NewRequest("GET", "/wizard/state", nil)
	r.Host = "evil.example:7844"
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	if rec.Code != 403 {
		t.Fatalf("non-loopback host %d", rec.Code)
	}
	tok := tokenOf(t, h)
	if w = serve(h, "POST", "/wizard/browse", url.Values{}, nil); w.Code != 403 {
		t.Fatalf("no token %d", w.Code)
	}
	if w = serve(h, "POST", "/wizard/browse", url.Values{"token": {tok}}, map[string]string{"Origin": "http://evil.example"}); w.Code != 403 {
		t.Fatalf("cross origin %d", w.Code)
	}
	if w = serve(h, "POST", "/wizard/browse", url.Values{"token": {tok}}, map[string]string{"Sec-Fetch-Site": "cross-site"}); w.Code != 403 {
		t.Fatalf("cross site %d", w.Code)
	}

	// Browse -> picker -> validated selection in the returned view.
	dir := f.dir("Models")
	f.picker.set(dir)
	w = serve(h, "POST", "/wizard/browse", url.Values{"token": {tok}}, map[string]string{"Origin": "http://127.0.0.1:7844"})
	var v View
	if err := json.Unmarshal(w.Body.Bytes(), &v); err != nil || w.Code != 200 || v.Selection == nil || v.Selection.Home != dir {
		t.Fatalf("browse %d %s", w.Code, w.Body)
	}
	// A picker cancel is not an error.
	f.picker.mu.Lock()
	f.picker.err = desktop.ErrPickCancelled
	f.picker.mu.Unlock()
	if w = serve(h, "POST", "/wizard/browse", url.Values{"token": {tok}}, nil); w.Code != 200 {
		t.Fatalf("cancel %d", w.Code)
	}
	// A rejected action reports its reason.
	w = serve(h, "POST", "/wizard/install", url.Values{"token": {tok}, "device": {"tpu"}}, nil)
	if w.Code != 409 || !strings.Contains(w.Body.String(), "device") {
		t.Fatalf("bad device %d %s", w.Code, w.Body)
	}
	if f.setups.Load() != 0 {
		t.Fatal("setup ran")
	}
	// The client can only install what the server-side picker selected.
	w = serve(h, "POST", "/wizard/install", url.Values{"token": {tok}, "device": {"cpu"}, "path": {"C:\\Windows"}}, nil)
	if w.Code != 200 {
		t.Fatalf("install %d %s", w.Code, w.Body)
	}
	f.waitStage(StageStarting)
	f.waitRemembered(dir)

	// Until READY every non-wizard path is still the wizard; after READY the
	// desktop home takes over.
	dash = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("dashboard")) })
	if w = serve(h, "GET", "/", nil, nil); strings.Contains(w.Body.String(), "dashboard") {
		t.Fatal("dashboard served before READY")
	}
	f.rt.set(worker.StateReady, "ready")
	if w = serve(h, "GET", "/", nil, nil); w.Body.String() != "dashboard" {
		t.Fatalf("after READY: %s", w.Body)
	}
	if w = serve(h, "GET", "/live", nil, nil); w.Body.String() != "dashboard" {
		t.Fatalf("dashboard path after READY: %s", w.Body)
	}
	if w = serve(h, "GET", "/wizard/state", nil, nil); w.Code != 200 || !strings.Contains(w.Body.String(), `"stage":"ready"`) {
		t.Fatalf("state after READY %d %s", w.Code, w.Body)
	}
}
