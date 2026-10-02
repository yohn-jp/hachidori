package main

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/firstrun"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/worker"
)

const (
	layaBase = setup.DefaultModel
	nanoID   = setup.OpenDeciderNano
)

// residentRuntime is a bound runtime with a known member list (default first)
// and the status document a resident set would report; no process exists.
type residentRuntime struct {
	models []string

	mu      sync.Mutex
	running bool
	starts  int
	stops   int
}

func (r *residentRuntime) Start() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.starts++
	r.running = true
	return true
}
func (r *residentRuntime) Restart() { r.Start() }
func (r *residentRuntime) Stop() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.stops++
	r.running = false
}
func (r *residentRuntime) Running() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.running
}
func (r *residentRuntime) Models() []string { return slices.Clone(r.models) }
func (r *residentRuntime) counts() (starts, stops int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.starts, r.stops
}

func (r *residentRuntime) Status() server.Status {
	w := worker.Snapshot{State: worker.StateReady, Phase: "ready", Ready: true, PID: 100, Starts: 1, Info: worker.Info{"device": "cuda"}}
	st := server.Status{Schema: "hachidori.v1", Runtime: server.Runtime{ModelID: r.models[0], Device: "cuda"}, Worker: w}
	if len(r.models) > 1 { // a single worker reports no resident table
		for i, m := range r.models {
			rw := w
			rw.PID = 100 + i
			st.Residents = append(st.Residents, server.ResidentStatus{Model: m, Provider: "p-" + m, Default: i == 0, Running: r.Running(),
				Status: server.Status{Runtime: server.Runtime{ModelID: m, Device: "cuda"}, Worker: rw}})
		}
	}
	return st
}

// residentDesktop is the desktop composition over the real dashboard binding,
// controller and settings store, with the runtime itself replaced. It records
// the selection each open read and the runtime each open produced.
type residentDesktop struct {
	app    *desktopApp
	mu     sync.Mutex
	read   [][]string
	opened []*residentRuntime
}

func newResidentDesktop(t *testing.T, prefsPath string) *residentDesktop {
	t.Helper()
	root := t.TempDir()
	p := &fakePlatform{version: "130.0"}
	a, _, _ := testApp(p, func() (home.Discovery, error) {
		return home.Discovery{Home: home.Home{Root: root}, Source: home.SourceEnv}, nil
	})
	a.PrefsPath = prefsPath
	a.Env = firstrun.Env{Load: func(string) (home.Active, error) { return home.Active{Device: "cuda"}, nil }}
	d := &residentDesktop{app: a}
	a.OpenRuntime = func(requested func() ([]string, error), bind bindFunc) app.OpenFunc {
		return func(string) (app.Runtime, error) {
			want, err := requested()
			if err != nil {
				return nil, err
			}
			rt := &residentRuntime{models: []string{layaBase}}
			for _, id := range want {
				if id != layaBase && !slices.Contains(rt.models, id) {
					rt.models = append(rt.models, id)
				}
			}
			d.mu.Lock()
			d.read = append(d.read, slices.Clone(want))
			d.opened = append(d.opened, rt)
			d.mu.Unlock()
			bind(rt.Status, rt, hostWorker{}, rt.Status().Runtime, time.Now())
			return rt, nil
		}
	}
	return d
}

func (d *residentDesktop) opens() ([][]string, []*residentRuntime) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return slices.Clone(d.read), slices.Clone(d.opened)
}

// run launches the desktop; in is called from inside the native window with
// the dashboard origin.
func (d *residentDesktop) run(t *testing.T, in func(origin string)) {
	t.Helper()
	d.app.Platform.(*fakePlatform).open = func(ctx context.Context, w desktop.Window) error {
		in(strings.TrimSuffix(w.URL, "/"))
		return nil
	}
	if err := d.app.run(); err != nil {
		t.Fatal(err)
	}
}

var tokenRe = regexp.MustCompile(`name="token" value="([^"]+)"`)

func page(t *testing.T, origin, path string) string {
	t.Helper()
	code, body := get(t, origin+path)
	if code != http.StatusOK {
		t.Fatalf("GET %s: %d", path, code)
	}
	return body
}

func post(t *testing.T, origin, path string, form url.Values) {
	t.Helper()
	m := tokenRe.FindStringSubmatch(page(t, origin, "/"))
	if m == nil {
		t.Fatal("no dashboard token on the dashboard")
	}
	form.Set("token", m[1])
	req, _ := http.NewRequest("POST", origin+path, strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Origin", origin)
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusSeeOther {
		t.Fatalf("POST %s: %d", path, resp.StatusCode)
	}
}

func rowOf(t *testing.T, body, model string) string {
	t.Helper()
	m := regexp.MustCompile(`(?s)<tr data-model="` + regexp.QuoteMeta(model) + `">.*?</tr>`).FindString(body)
	if m == "" {
		t.Fatalf("no resident-selection row for %s", model)
	}
	return m
}

// Issue #130: the resident selection is saved through the desktop's settings
// store beside desktop.json and survives a relaunch; the desktop's no-argument
// composition opens the persisted selection with no CLI argument.
func TestDesktopResidentSelectionPersistsAndIsRestoredOnLaunch(t *testing.T) {
	dir := t.TempDir()
	prefs := filepath.Join(dir, "desktop.json")
	legacy := `{"schema":"hachidori.desktop/1","start_minimized":true,"close_notice_shown":true}`
	if err := os.WriteFile(prefs, []byte(legacy), 0o644); err != nil {
		t.Fatal(err)
	}

	// First session: nothing saved, one resident; the operator selects the pair.
	first := newResidentDesktop(t, prefs)
	first.run(t, func(origin string) {
		reads, rts := first.opens()
		if len(reads) != 1 || len(reads[0]) != 0 || !slices.Equal(rts[0].models, []string{layaBase}) {
			t.Errorf("default launch: reads %v members %v", reads, rts[0].models)
		}
		post(t, origin, "/models/residents", url.Values{"resident": {nanoID}})
	})
	if got, err := settingsStore(prefs, nil).Residents(); err != nil || !slices.Equal(got, []string{nanoID}) {
		t.Fatalf("saved selection: %v %v", got, err)
	}
	if b, _ := os.ReadFile(prefs); string(b) != legacy {
		t.Fatalf("saving the selection rewrote desktop.json: %s", b)
	}

	// Relaunch: no arguments, a new composition over the same prefs path.
	second := newResidentDesktop(t, prefs)
	second.run(t, func(origin string) {
		reads, rts := second.opens()
		if len(reads) != 1 || !slices.Equal(reads[0], []string{nanoID}) || !slices.Equal(rts[0].models, []string{layaBase, nanoID}) {
			t.Errorf("relaunch did not restore the selection: reads %v members %v", reads, rts[0].models)
		}
		if starts, _ := rts[0].counts(); starts != 1 {
			t.Errorf("the restored set was started %d times", starts)
		}
		body := page(t, origin, "/models")
		if strings.Contains(body, `id="restart-required"`) || !strings.Contains(rowOf(t, body, nanoID), "resident") || !strings.Contains(rowOf(t, body, nanoID), " checked") {
			t.Errorf("restored selection is shown as pending or unselected:\n%s", rowOf(t, body, nanoID))
		}
		if !strings.Contains(page(t, origin, "/"), `id="resident-`+nanoID+`"`) {
			t.Error("Runtime does not project the second resident")
		}
	})
}

// Changing the selection under a running desktop is next-start intent: nothing
// is opened, started or stopped until the explicit Restart, which opens exactly
// the saved set and retires the old one.
func TestDesktopResidencyChangeRequiresExplicitRestart(t *testing.T) {
	prefs := filepath.Join(t.TempDir(), "desktop.json")
	if err := settingsStore(prefs, nil).SetResidents([]string{nanoID}); err != nil {
		t.Fatal(err)
	}
	d := newResidentDesktop(t, prefs)
	d.run(t, func(origin string) {
		_, rts := d.opens()
		old := rts[0]

		post(t, origin, "/models/residents", url.Values{}) // deselect everything
		body := page(t, origin, "/models")
		if !strings.Contains(body, `id="restart-required"`) || !strings.Contains(body, "The resident selection changed") ||
			!strings.Contains(rowOf(t, body, nanoID), "resident · removed on restart") {
			t.Errorf("a selection change is not shown as restart-required:\n%s", body)
		}
		if reads, rts := d.opens(); len(reads) != 1 || len(rts) != 1 {
			t.Errorf("the change opened a runtime: %v", reads)
		}
		if starts, stops := old.counts(); starts != 1 || stops != 0 || !old.Running() {
			t.Errorf("the change touched the running runtime: starts %d stops %d", starts, stops)
		}

		post(t, origin, "/models/restart", url.Values{})
		reads, rts := d.opens()
		if len(reads) != 2 || len(reads[1]) != 0 || !slices.Equal(rts[1].models, []string{layaBase}) {
			t.Fatalf("restart did not apply the selection: reads %v", reads)
		}
		if _, stops := old.counts(); stops != 1 || old.Running() {
			t.Errorf("the old set was not retired: stops %d", stops)
		}
		if starts, _ := rts[1].counts(); starts != 1 || !rts[1].Running() {
			t.Errorf("the new set was not started once: %d", starts)
		}
		if body := page(t, origin, "/models"); strings.Contains(body, `id="restart-required"`) {
			t.Error("restart-required remains after the restart applied the selection")
		}
	})
	if got, _ := settingsStore(prefs, nil).Residents(); len(got) != 0 {
		t.Fatalf("persisted selection %v", got)
	}
}

// An unreadable selection stops the start with its cause; it never starts a
// runtime with fewer residents than the operator asked for.
func TestDesktopUnreadableResidentSelectionFailsTheOpen(t *testing.T) {
	prefs := filepath.Join(t.TempDir(), "desktop.json")
	if err := os.WriteFile(settingsPathOf(prefs), []byte(`{"schema":"hachidori.settings/1","resident_models":["someone/else"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	d := newResidentDesktop(t, prefs)
	d.run(t, func(string) {})
	if reads, rts := d.opens(); len(reads) != 0 || len(rts) != 0 {
		t.Fatalf("a runtime opened over an unreadable selection: %v", reads)
	}
}

func settingsPathOf(prefs string) string { return desktop.SettingsPath(prefs) }

// The form token is per process: a page the window loaded before a runtime
// rebind (a Restart that applies a residency change replaces the dashboard)
// still posts with its token instead of being refused as stale.
func TestDesktopFormTokenSurvivesARuntimeRebind(t *testing.T) {
	prefs := filepath.Join(t.TempDir(), "desktop.json")
	if err := settingsStore(prefs, nil).SetResidents([]string{nanoID}); err != nil {
		t.Fatal(err)
	}
	d := newResidentDesktop(t, prefs)
	d.run(t, func(origin string) {
		m := tokenRe.FindStringSubmatch(page(t, origin, "/settings"))
		if m == nil {
			t.Fatal("no dashboard token on Settings")
		}
		loaded := m[1]
		post(t, origin, "/models/residents", url.Values{})
		post(t, origin, "/models/restart", url.Values{})
		if reads, _ := d.opens(); len(reads) != 2 {
			t.Fatalf("the restart did not rebind the runtime: %v", reads)
		}
		form := url.Values{"token": {loaded}, "model": {nanoID}}
		req, _ := http.NewRequest("POST", origin+"/models/residents", strings.NewReader(form.Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		req.Header.Set("Origin", origin)
		client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
		resp, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusSeeOther {
			t.Fatalf("a post with the token of the page loaded before the rebind: %d %s", resp.StatusCode, strings.TrimSpace(string(body)))
		}
	})
}

// Forge's Apply reaches the persistent desktop controller as one action
// (ApplyCertifiedVariant): a refusal is the backend's validation, shown in the
// Forge workspace beside its diagnostic, and nothing else (activation, stop,
// restart) is called from the browser side.
func TestDesktopForgeApplyIsOneControllerAction(t *testing.T) {
	d := newResidentDesktop(t, filepath.Join(t.TempDir(), "desktop.json"))
	d.run(t, func(origin string) {
		_, rts := d.opens()
		rt := rts[0]
		post(t, origin, "/forge/apply", url.Values{"variant": {"clef-flash--none--000000000000"}, "model": {setup.ClefFlash}, "device": {"cpu"}, "return": {"forge"}})
		deadline := time.Now().Add(10 * time.Second)
		var body string
		for time.Now().Before(deadline) {
			body = page(t, origin, "/forge")
			if strings.Contains(body, `id="models-last"`) {
				break
			}
			time.Sleep(20 * time.Millisecond)
		}
		for _, want := range []string{`id="models-last"`, "apply", "Failed in phase", "<strong>Validating</strong>", "clef-flash--none--000000000000", `id="forge-diagnostic-last"`} {
			if !strings.Contains(body, want) {
				t.Errorf("the failed apply is not shown in Forge: lacks %q", want)
			}
		}
		if starts, stops := rt.counts(); starts != 1 || stops != 0 || !rt.Running() {
			t.Errorf("a refused apply touched the runtime: starts %d stops %d", starts, stops)
		}
		if reads, _ := d.opens(); len(reads) != 1 {
			t.Errorf("a refused apply rebound the runtime: %d opens", len(reads))
		}
		// An invalid request is refused at admission, before any operation exists.
		post(t, origin, "/forge/apply", url.Values{"variant": {"clef-flash--none--000000000000"}, "device": {"sideways"}})
		if body := page(t, origin, "/forge"); !strings.Contains(body, "the serving device must be cpu or cuda") {
			t.Errorf("the admission refusal is not shown:\n%s", body)
		}
	})
}
