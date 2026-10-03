package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/i18n"
	"github.com/yohn-jp/hachidori/internal/optimize"
	"github.com/yohn-jp/hachidori/internal/settings"
	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tuning"
	"github.com/yohn-jp/hachidori/internal/tunnel"
)

// The settings authority composes the existing desktop preferences (an existing
// desktop.json keeps working) and stores runtime defaults beside it without
// rewriting desktop.json.
func TestSettingsStoreComposesDesktopPrefs(t *testing.T) {
	dir := t.TempDir()
	prefs := filepath.Join(dir, "desktop.json")
	legacy := `{"schema":"hachidori.desktop/1","start_minimized":true,"close_notice_shown":true}`
	if err := os.WriteFile(prefs, []byte(legacy), 0o644); err != nil {
		t.Fatal(err)
	}
	st := settingsStore(prefs, &desktop.Manager{Path: prefs})
	if _, min, err := st.Prefs(); err != nil || !min {
		t.Fatalf("start minimized %v %v", min, err)
	}
	if err := st.SetDefaults(settings.Defaults{Device: "cpu", Model: setup.DefaultModel}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "settings.json")); err != nil {
		t.Fatalf("settings.json not beside desktop.json: %v", err)
	}
	if b, _ := os.ReadFile(prefs); string(b) != legacy {
		t.Fatalf("saving defaults rewrote desktop.json: %s", b)
	}
}

// The desktop hosts Development Connection profiles in the same settings
// store, beside desktop.json, and they survive a restart of the composition.
func TestSettingsStoreHostsDevelopmentConnections(t *testing.T) {
	dir := t.TempDir()
	prefs := filepath.Join(dir, "desktop.json")
	var _ dashboard.Connections = settingsStore(prefs, nil)
	c := settings.Connection{Name: "nixos-dev", Destination: "dev@nixos", RemoteBind: "127.0.0.1", RemoteBindMode: settings.ConnectionPinned, RemotePort: 7843, RemotePortMode: settings.ConnectionPinned, LocalPort: 7843, LocalPortMode: settings.ConnectionPinned}
	if err := settingsStore(prefs, nil).SaveConnection(c); err != nil {
		t.Fatal(err)
	}
	got, err := settingsStore(prefs, nil).Connections()
	if err != nil || len(got) != 1 || got[0].Name != c.Name || got[0].Destination != c.Destination || got[0].RemoteBind != c.RemoteBind || got[0].RemoteBindMode != c.RemoteBindMode || got[0].RemotePort != c.RemotePort || got[0].RemotePortMode != c.RemotePortMode || got[0].LocalPort != c.LocalPort || got[0].LocalPortMode != c.LocalPortMode {
		t.Fatalf("after restart: %+v %v", got, err)
	}
	if _, err := os.Stat(prefs); !os.IsNotExist(err) {
		t.Fatalf("saving a profile wrote desktop.json: %v", err)
	}
}

func TestSettingsStoreResolvesAutoFromBoundDesktopAPIEndpoint(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	endpoint, err := tunnel.LocalEndpointFromAddr(ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}

	store := settingsStore(filepath.Join(t.TempDir(), "desktop.json"), nil)
	if err := store.SetLocalEndpoint(endpoint); err != nil {
		t.Fatal(err)
	}
	auto := settings.Connection{
		Name: "auto-dev", Destination: "devhost",
		RemoteBindMode: settings.ConnectionAuto, RemotePortMode: settings.ConnectionAuto, LocalPortMode: settings.ConnectionAuto,
	}
	if err := store.SaveConnection(auto); err != nil {
		t.Fatal(err)
	}
	profiles, err := store.Connections()
	if err != nil || len(profiles) != 1 {
		t.Fatalf("saved connection: %+v %v", profiles, err)
	}
	got := profiles[0].Spec()
	want := tunnel.Spec{Destination: "devhost", RemoteBind: "127.0.0.1", RemotePort: endpoint.Port, LocalPort: endpoint.Port}
	if got != want {
		t.Fatalf("managed API endpoint %q resolved to %+v, want %+v", ln.Addr(), got, want)
	}
	if profiles[0].RemotePort != 0 || profiles[0].LocalPort != 0 || profiles[0].RemotePortMode != settings.ConnectionAuto || profiles[0].LocalPortMode != settings.ConnectionAuto {
		t.Fatalf("resolved values replaced stored intent: %+v", profiles[0])
	}
}

// Proof 11 and 12: explicit commands stay CLI commands and the no-argument
// desktop is reachable only through the platform hook.
func TestRunDispatch(t *testing.T) {
	var called atomic.Int32
	noArg := func() error { called.Add(1); return nil }

	if code := run(nil, noArg); code != 0 || called.Load() != 1 {
		t.Fatalf("no args with a desktop hook: code %d, hook calls %d", code, called.Load())
	}
	called.Store(0)
	if code := run(nil, func() error { called.Add(1); return errors.New("boom") }); code != 1 || called.Load() != 1 {
		t.Fatalf("desktop failure: code %d", code)
	}
	called.Store(0)
	if code := run(nil, nil); code != 2 {
		t.Fatalf("no args and no hook must print usage (2), got %d", code)
	}
	for _, args := range [][]string{
		{"bogus"},
		{"status", "-endpoint", "http://127.0.0.1:1"}, // a real CLI command that fails fast
	} {
		if code := run(args, noArg); code == 0 {
			t.Fatalf("%v succeeded", args)
		}
	}
	if called.Load() != 0 {
		t.Fatalf("the desktop hook ran for explicit CLI commands (%d)", called.Load())
	}
}

type fakePlatform struct {
	release atomic.Int32
	open    func(ctx context.Context, w desktop.Window) error
	version string
	opened  atomic.Int32
}

func (p *fakePlatform) RuntimeVersion() (string, error) { return p.version, nil }
func (p *fakePlatform) AcquireInstance() (func(), error) {
	return func() { p.release.Add(1) }, nil
}
func (p *fakePlatform) Activate() error            { return nil }
func (p *fakePlatform) ReportError(string, string) {}
func (p *fakePlatform) Open(ctx context.Context, w desktop.Window) error {
	p.opened.Add(1)
	return p.open(ctx, w)
}

type neverPicker struct{ calls atomic.Int32 }

func (p *neverPicker) PickFolder(context.Context, string) (string, error) {
	p.calls.Add(1)
	return "", desktop.ErrPickCancelled
}

func get(t *testing.T, url string) (int, string) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

func testApp(p desktop.Platform, d func() (home.Discovery, error)) (*desktopApp, *atomic.Int32, *atomic.Int32) {
	var setups, remembers atomic.Int32
	return &desktopApp{
		Platform: p, Picker: &neverPicker{}, Discover: d,
		Remember: func(string) (home.Home, error) { remembers.Add(1); return home.Home{}, errors.New("unexpected") },
		APIAddr:  "127.0.0.1:0", DashAddr: "127.0.0.1:0",
		Setup:  func(string, string, string, io.Writer, *setup.Observer) error { setups.Add(1); return nil },
		Stderr: &bytes.Buffer{},
	}, &setups, &remembers
}

// Proof 1: no bootstrap -> a native window with the first-run wizard, before
// any runtime, home, setup or bootstrap write exists.
func TestNoBootstrapOpensFirstRunWindow(t *testing.T) {
	var dataDir, url string
	p := &fakePlatform{version: "130.0"}
	p.open = func(ctx context.Context, w desktop.Window) error {
		dataDir, url = w.DataDir, w.URL
		if !w.Policy.AllowNavigation(w.URL) || w.Policy.AllowNavigation("https://example.com/") {
			t.Error("window policy must allow only the local origin")
		}
		if _, err := os.Stat(w.DataDir); err != nil {
			os.MkdirAll(w.DataDir, 0o755) // the real window creates its profile
		}
		code, body := get(t, w.URL)
		if code != 200 || !strings.Contains(body, "Where should Hachidori keep its models and runtime?") {
			t.Errorf("first-run page %d", code)
		}
		code, body = get(t, w.URL+"wizard/state")
		if code != 200 || !strings.Contains(body, `"mode":"first_run"`) || !strings.Contains(body, `"stage":"select"`) || !strings.Contains(body, `"state":"unconfigured"`) {
			t.Errorf("state %d %s", code, body)
		}
		return nil
	}
	a, setups, remembers := testApp(p, func() (home.Discovery, error) { return home.Discovery{Source: home.SourceUnconfigured}, nil })
	if err := a.run(); err != nil {
		t.Fatal(err)
	}
	if p.opened.Load() != 1 || p.release.Load() != 1 || setups.Load() != 0 || remembers.Load() != 0 {
		t.Fatalf("opened %d released %d setups %d remembers %d", p.opened.Load(), p.release.Load(), setups.Load(), remembers.Load())
	}
	if !strings.HasPrefix(url, "http://127.0.0.1:") {
		t.Fatalf("url %q", url)
	}
	// With no home yet, the window profile is throwaway and removed on exit.
	if _, err := os.Stat(dataDir); !os.IsNotExist(err) {
		t.Fatalf("temporary window profile %s not removed: %v", dataDir, err)
	}
	if !strings.HasPrefix(dataDir, os.TempDir()) {
		t.Fatalf("data dir %q", dataDir)
	}
}

// Proof 3: a stored home that vanished shows recovery, never a fresh install.
func TestMissingStoredHomeShowsRecovery(t *testing.T) {
	loc := home.Locator{Path: filepath.Join(t.TempDir(), "bootstrap.json")}
	gone := filepath.Join(t.TempDir(), "gone")
	if err := os.Mkdir(gone, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := loc.Save(gone); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(gone); err != nil {
		t.Fatal(err)
	}
	p := &fakePlatform{version: "130.0"}
	p.open = func(ctx context.Context, w desktop.Window) error {
		_, body := get(t, w.URL+"wizard/state")
		if !strings.Contains(body, `"mode":"recovery_missing"`) || !strings.Contains(body, `"stage":"select"`) || !strings.Contains(body, "missing or unavailable") {
			t.Errorf("recovery state %s", body)
		}
		return nil
	}
	a, setups, remembers := testApp(p, func() (home.Discovery, error) {
		h, _, err := loc.Lookup()
		return home.Discovery{Home: h, Source: home.SourceLocator}, err
	})
	if err := a.run(); err != nil {
		t.Fatal(err)
	}
	if setups.Load() != 0 || remembers.Load() != 0 {
		t.Fatal("recovery installed or rewrote the locator on its own")
	}
	if _, err := os.Stat(gone); !os.IsNotExist(err) {
		t.Fatal("the missing home was recreated")
	}
	b, found, err := loc.Load()
	if err != nil || !found || b.Home != gone {
		t.Fatalf("locator %+v %v %v", b, found, err)
	}
}

// Prerequisite failures happen before anything starts.
func TestPreflightFailureStartsNothing(t *testing.T) {
	p := &fakePlatform{version: ""} // WebView2 missing
	p.open = func(context.Context, desktop.Window) error { t.Error("window opened"); return nil }
	a, _, _ := testApp(p, func() (home.Discovery, error) { t.Error("discovery ran"); return home.Discovery{}, nil })
	if err := a.run(); !errors.Is(err, desktop.ErrWebView2Missing) {
		t.Fatalf("err %v", err)
	}
}

var _ dashboard.Models = modelManager{}

func TestForgeResolutionProjectionPreservesAutoAndOverrides(t *testing.T) {
	resolution := forgeResolution(&app.ForgeBuildEvaluateResolution{
		Source: "clef-flash", Recipe: "clef-flash-w4a16-rtn-g128", Variant: "clef-flash--r--aaaaaaaaaaaa",
		CandidateDevice: app.ForgeResolvedValue{Mode: app.ForgeSelectionAuto, Value: "cuda"},
		ReferenceDevice: app.ForgeResolvedValue{Mode: app.ForgeSelectionOverride, Value: "cpu"},
		ReferenceDType:  app.ForgeResolvedValue{Mode: app.ForgeSelectionAuto, Value: "bfloat16"},
		CandidateDType:  "bfloat16",
	})
	if resolution == nil || resolution.Source != "clef-flash" || resolution.Variant != "clef-flash--r--aaaaaaaaaaaa" ||
		resolution.CandidateDevice != (dashboard.ForgeResolvedValue{Mode: "Auto", Value: "cuda"}) ||
		resolution.ReferenceDevice != (dashboard.ForgeResolvedValue{Mode: "Override", Value: "cpu"}) ||
		resolution.ReferenceDType != (dashboard.ForgeResolvedValue{Mode: "Auto", Value: "bfloat16"}) || resolution.CandidateDType != "bfloat16" {
		t.Fatalf("resolved plan projection %+v", resolution)
	}
}

// The dashboard's manager is the application controller over the real setup
// authority: the inventory lists only catalog identities, an activation
// whose artifacts are not materialized is refused without writing anything,
// and removal cannot escape HACHIDORI_HOME or name a non-catalog artifact.
func TestModelManagerOverRealSetupAuthority(t *testing.T) {
	root := t.TempDir()
	h := home.Home{Root: filepath.Join(root, "home")}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(root, "outside")
	if err := os.MkdirAll(outside, 0o755); err != nil {
		t.Fatal(err)
	}
	ctl := app.New(app.Config{Home: h.Root})
	m := modelManager{ctl: func() *app.Controller { return ctl }}

	st := m.State()
	if st.Err != "" || len(st.Inventory.Runtimes) != len(setup.Devices) || len(st.Inventory.Models) != len(setup.Models) || st.RestartRequired {
		t.Fatalf("state %+v", st)
	}
	for _, r := range st.Inventory.Runtimes {
		if r.Materialized || r.Active {
			t.Fatalf("empty home reports %+v", r)
		}
	}
	if err := m.Activate("cuda", setup.DefaultModel); err != nil {
		t.Fatalf("activation was not accepted: %v", err)
	}
	waitModelsIdle(t, m)
	if _, err := os.Stat(h.Path("state", "active-runtime.json")); !os.IsNotExist(err) {
		t.Fatal("failed activation wrote an activation record")
	}
	if got := m.State().Last; got == nil || got.Kind != app.OpActivate || got.Failure == "" {
		t.Fatalf("failure not reported: %+v", got)
	}
	for _, c := range [][2]string{{"runtime", "../../outside"}, {"runtime", outside}, {"model", "../outside"}, {"model", ""}, {"runtime", "cuda-notcatalog"}} {
		if err := m.Remove(c[0], c[1]); err != nil {
			t.Fatalf("Remove(%v) was not accepted: %v", c, err)
		}
		waitModelsIdle(t, m)
		if got := m.State().Last; got == nil || got.Kind != app.OpRemove || got.Failure == "" {
			t.Fatalf("Remove(%v) did not fail: %+v", c, got)
		}
	}
	if _, err := os.Stat(outside); err != nil {
		t.Fatal("directory outside the home removed")
	}
	// Without a home nothing is inspected.
	if st := (modelManager{ctl: func() *app.Controller { return app.New(app.Config{}) }}).State(); st.Err == "" {
		t.Fatal("no home is not reported")
	}
}

func TestModelManagerStartsAndProjectsDesiredStateOperation(t *testing.T) {
	root := t.TempDir()
	h := home.Home{Root: filepath.Join(root, "home")}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	prefs := settingsStore(filepath.Join(root, "desktop.json"), nil)
	ctl := app.New(app.Config{Home: h.Root, Residents: func() []string { ids, _ := prefs.Residents(); return ids }, SetResidents: prefs.SetResidents})
	m := modelManager{ctl: func() *app.Controller { return ctl }}
	intent := dashboard.DesiredStateRequest{Model: setup.DefaultModel, DeviceMode: "pinned", Device: "cpu", Residents: []string{"opendecider-nano"}}
	if err := m.StartDesiredState(intent); err != nil {
		t.Fatalf("desired-state operation was not accepted: %v", err)
	}
	waitModelsIdle(t, m)
	op := m.State().Last
	if op == nil || op.Kind != app.OpDesiredState || op.Model != setup.DefaultModel || op.DeviceMode != "pinned" || op.RequestedDevice != "cpu" || op.ResolvedDevice != "cpu" ||
		len(op.Residents) != 1 || op.Residents[0] != "opendecider-nano" {
		t.Fatalf("desired-state operation was not projected: %+v", op)
	}
}

// waitModelsIdle waits for the maintenance action in flight to finish: the
// actions are accepted at once and run in the background.
func waitModelsIdle(t *testing.T, m modelManager) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for m.State().Busy != nil {
		if time.Now().After(deadline) {
			t.Fatal("maintenance action did not finish")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

type cancelPathPicker struct{ neverPicker }

func (*cancelPathPicker) PickOpen(context.Context, string) (string, error) {
	return "", desktop.ErrPickCancelled
}
func (*cancelPathPicker) PickSave(context.Context, string) (string, error) {
	return "", errors.New("dialog broke")
}

func TestDashboardPathPickerIsDesktopFileCapabilityOnly(t *testing.T) {
	if p := dashboardPathPicker(&neverPicker{}); p != nil {
		t.Fatal("a folder-only picker became a dashboard path picker")
	}
	if p := dashboardPathPicker(desktop.NativePicker()); p != nil && runtime.GOOS != "windows" {
		t.Fatal("a platform without native file dialogs exposed a path picker")
	}
	p := dashboardPathPicker(&cancelPathPicker{})
	if p == nil {
		t.Fatal("a file-capable picker was not exposed")
	}
	if _, err := p.PickOpen(context.Background(), ""); !errors.Is(err, dashboard.ErrPickCancelled) {
		t.Fatalf("cancel = %v, want dashboard.ErrPickCancelled", err)
	}
	if _, err := p.PickFolder(context.Background(), ""); !errors.Is(err, dashboard.ErrPickCancelled) {
		t.Fatalf("folder cancel = %v", err)
	}
	if _, err := p.PickSave(context.Background(), ""); err == nil || errors.Is(err, dashboard.ErrPickCancelled) {
		t.Fatalf("failure = %v, want the picker error", err)
	}
}

// The operator UI locale is stored by the same settings authority beside
// desktop.json, exists before any home is selected (first run resolves it
// there too), and survives a restart of the composition.
func TestSettingsStorePersistsOperatorLocale(t *testing.T) {
	t.Setenv("LC_ALL", "")
	t.Setenv("LC_MESSAGES", "")
	t.Setenv("LANG", "C.UTF-8")
	prefs := filepath.Join(t.TempDir(), "desktop.json")
	var _ dashboard.Settings = settingsStore(prefs, nil)
	if got := settingsStore(prefs, nil).ResolvedLocale(); got != i18n.English {
		t.Fatalf("unset locale resolves to %q", got)
	}
	if err := settingsStore(prefs, nil).SetLocale("ja"); err != nil {
		t.Fatal(err)
	}
	if got := settingsStore(prefs, nil).ResolvedLocale(); got != i18n.Japanese {
		t.Fatalf("after restart: %q", got)
	}
}

// tuningFixture materializes a synthetic catalog Clef model: digest-pinned
// config and safetensors index, one shard with a real safetensors header and
// the recipe's carried files. Names carry the Linear modules the analyzer must
// find and the norm, convolution, embedding and patch tensors it must not.
type fixtureTensor struct {
	name  string
	shape []int
}

var tuningFixtureTensors = []fixtureTensor{
	{"lm_head.weight", []int{4, 8}},
	{"model.language_model.embed_tokens.weight", []int{4, 8}},
	{"model.language_model.layers.0.input_layernorm.weight", []int{8}},
	{"model.language_model.layers.0.linear_attn.conv1d.weight", []int{6, 1, 4}},
	{"model.language_model.layers.0.linear_attn.norm.weight", []int{8}},
	{"model.language_model.layers.0.linear_attn.in_proj_a.weight", []int{2, 8}},
	{"model.language_model.layers.0.linear_attn.in_proj_b.weight", []int{2, 8}},
	{"model.language_model.layers.0.linear_attn.in_proj_qkv.weight", []int{24, 8}},
	{"model.language_model.layers.0.mlp.gate_proj.weight", []int{16, 8}},
	{"model.language_model.layers.1.self_attn.q_proj.weight", []int{8, 8}},
	{"model.language_model.layers.1.self_attn.q_norm.weight", []int{8}},
	{"model.visual.patch_embed.proj.weight", []int{8, 3, 2, 2, 2}},
	{"model.visual.pos_embed.weight", []int{4, 8}},
	{"model.visual.blocks.0.norm1.weight", []int{8}},
	{"model.visual.blocks.0.attn.qkv.weight", []int{24, 8}},
	{"model.visual.merger.linear_fc1.weight", []int{8, 8}},
}

func (f fixtureTensor) bytes() int64 {
	n := int64(2) // bfloat16
	for _, d := range f.shape {
		n *= int64(d)
	}
	return n
}

func writeSafetensors(t *testing.T, path string, tensors []fixtureTensor) {
	t.Helper()
	header := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	var off int64
	for _, ts := range tensors {
		header[ts.name] = map[string]any{"dtype": "BF16", "shape": ts.shape, "data_offsets": []int64{off, off + ts.bytes()}}
		off += ts.bytes()
	}
	raw, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	var n [8]byte
	binary.LittleEndian.PutUint64(n[:], uint64(len(raw)))
	if err := os.WriteFile(path, append(append(n[:], raw...), make([]byte, off)...), 0o644); err != nil {
		t.Fatal(err)
	}
}

func tuningFixture(t *testing.T) (tuningStore, home.Home, home.ModelManifest, string) {
	t.Helper()
	h := home.Home{Root: t.TempDir()}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	m := home.ModelManifest{ID: setup.ClefFlash, Provider: home.ProviderClef, Repo: "test/clef", Revision: strings.Repeat("ef", 20), Files: map[string]string{}}
	dir := h.Path("models", filepath.FromSlash(setup.ModelDirName(m)))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	recipe, err := optimize.LookupRecipe(setup.ClefFlash, optimize.RecipeClefFlashW4A16)
	if err != nil {
		t.Fatal(err)
	}
	weightMap := map[string]string{}
	for _, ts := range tuningFixtureTensors {
		weightMap[ts.name] = "model-00001-of-00001.safetensors"
	}
	index, _ := json.Marshal(map[string]any{"weight_map": weightMap})
	config, _ := json.Marshal(map[string]any{
		"model_type": "qwen3_5", "architectures": []string{"Qwen3_5ForConditionalGeneration"},
		"text_config": map[string]any{"model_type": "qwen3_5_text", "layer_types": []string{"linear_attention", "full_attention"}},
	})
	files := map[string][]byte{"config.json": config, "generation_config.json": []byte("{}"), "model.safetensors.index.json": index}
	for _, f := range recipe.Carry {
		files[f] = []byte("carried " + f)
	}
	for rel, b := range files {
		if err := os.WriteFile(filepath.Join(dir, rel), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	writeSafetensors(t, filepath.Join(dir, "model-00001-of-00001.safetensors"), tuningFixtureTensors)
	for _, rel := range []string{"model-00001-of-00001.safetensors"} {
		files[rel] = nil
	}
	for rel := range files {
		d, err := setup.FileSHA256(filepath.Join(dir, rel))
		if err != nil {
			t.Fatal(err)
		}
		m.Files[rel] = d
	}
	if err := home.WriteJSON(filepath.Join(dir, "hachidori-model.json"), m); err != nil {
		t.Fatal(err)
	}
	old := setup.Models
	setup.Models = []home.ModelManifest{m}
	t.Cleanup(func() { setup.Models = old })
	ctl := app.New(app.Config{Home: h.Root})
	return tuningStore{ctl: func() *app.Controller { return ctl }}, h, m, dir
}

func pinnedProfile(t *testing.T, a tuning.Analysis, objective string, regions ...string) tuning.Profile {
	t.Helper()
	p, err := tuning.NewDefaultProfile(a, objective)
	if err != nil {
		t.Fatal(err)
	}
	for _, region := range regions {
		ids, err := tuning.Selection{Region: region, From: -1, To: -1}.Select(a)
		if err != nil {
			t.Fatal(err)
		}
		if p, err = tuning.SetPolicy(p, a, ids, home.PolicySourcePrecision); err != nil {
			t.Fatal(err)
		}
	}
	return p
}

func TestTuningStoreAnalyzesTheExactPinnedSourceAndStoresProfiles(t *testing.T) {
	store, h, _, dir := tuningFixture(t)
	var _ dashboard.Tuning = store
	a, err := store.Analysis(setup.ClefFlash)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Regions) != 8 {
		t.Fatalf("regions: %+v", a.Regions)
	}
	modules := map[string][]string{}
	for _, r := range a.Regions {
		modules[r.ID] = r.Modules
	}
	if got := strings.Join(modules[tuning.RegionLinearAttention], ","); got != "model.language_model.layers.0.linear_attn.in_proj_qkv" {
		t.Errorf("linear-attention members %q", got)
	}
	if got := strings.Join(modules[tuning.RegionVision], ","); got != "model.visual.blocks.0.attn.qkv,model.visual.merger.linear_fc1" {
		t.Errorf("vision members %q", got)
	}
	all := strings.Join(func() (s []string) {
		for _, ms := range modules {
			s = append(s, ms...)
		}
		return
	}(), "\n")
	for _, not := range []string{"conv1d", "linear_attn.norm", "embed_tokens", "patch_embed", "pos_embed", "q_norm", "layernorm", "norm1"} {
		if strings.Contains(all, not) {
			t.Errorf("non-Linear tensor %q was analyzed as a Linear module", not)
		}
	}
	again, err := store.Analysis(setup.ClefFlash)
	if err != nil || string(again.Canonical()) != string(a.Canonical()) {
		t.Fatalf("analysis is not deterministic: %v", err)
	}

	// versioned profiles persist under HACHIDORI_HOME, newest first
	first, second := pinnedProfile(t, a, "balanced"), pinnedProfile(t, a, "maximum-fidelity", tuning.RegionFeedForward)
	for i, p := range []tuning.Profile{first, second} {
		if err := store.SaveProfile(p, a); err != nil {
			t.Fatal(err)
		}
		at := time.Now().Add(time.Duration(i-5) * time.Minute)
		if err := os.Chtimes(h.Path("state", "tuning", "profiles", p.ID()+".json"), at, at); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(h.Path("state", "tuning", "profiles", strings.Repeat("0", 64)+".json"), []byte("{broken"), 0o644); err != nil {
		t.Fatal(err)
	}
	list, err := store.Profiles(setup.ClefFlash)
	if err != nil || len(list) != 2 || list[0].ID() != second.ID() || list[1].ID() != first.ID() {
		t.Fatalf("profiles %v %v", err, list)
	}
	if other, _ := store.Profiles("another-model"); len(other) != 0 {
		t.Errorf("profiles of another source: %v", other)
	}
	got, ga, err := store.LoadProfile(second.ID())
	if err != nil || got.ID() != second.ID() || ga.SHA256() != a.SHA256() {
		t.Fatalf("load: %v", err)
	}

	// a source file that is not the pinned one is refused, never analyzed
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"other"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Analysis(setup.ClefFlash); err == nil || !strings.Contains(err.Error(), "pinned digest") {
		t.Errorf("tampered config: %v", err)
	}
	if err := os.Remove(filepath.Join(dir, "hachidori-model.json")); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Analysis(setup.ClefFlash); err == nil || !strings.Contains(err.Error(), "not materialized") {
		t.Errorf("unmaterialized source: %v", err)
	}
}

func TestTuningStoreImpactIsEstimatedUntilACandidateIsMeasured(t *testing.T) {
	store, h, m, dir := tuningFixture(t)
	a, err := store.Analysis(setup.ClefFlash)
	if err != nil {
		t.Fatal(err)
	}
	impact := func(p tuning.Profile) dashboard.TuningImpact {
		t.Helper()
		c, err := tuning.Compile(p, a)
		if err != nil {
			t.Fatal(err)
		}
		imp, err := store.Impact(p, a, c)
		if err != nil {
			t.Fatal(err)
		}
		return imp
	}
	notChecked := func(name string, v dashboard.ImpactValue) {
		t.Helper()
		if v.State != dashboard.NotChecked || v.Value != "" || v.Basis == "" {
			t.Errorf("%s is %+v, want NOT_CHECKED with a reason", name, v)
		}
	}

	// expected estimate, computed independently from the fixture
	var total, quantizable int64
	for _, ts := range tuningFixtureTensors {
		total += ts.bytes()
		if strings.Contains(ts.name, "in_proj_qkv") || strings.Contains(ts.name, "mlp.") || strings.Contains(ts.name, "self_attn.q_proj") {
			quantizable += ts.bytes()
		}
	}
	var carried int64
	recipe, _ := optimize.LookupRecipe(setup.ClefFlash, optimize.RecipeClefFlashW4A16)
	for _, f := range recipe.Carry {
		info, err := os.Stat(filepath.Join(dir, f))
		if err != nil {
			t.Fatal(err)
		}
		carried += info.Size()
	}
	want := func(quantized int64) string {
		return "about " + gib(int64(float64(total)-float64(quantized)*(1-w4a16BytesPerBF16Byte)+float64(carried)))
	}
	auto := pinnedProfile(t, a, "balanced")
	imp := impact(auto)
	if imp.Size.State != dashboard.Estimated || imp.Size.Value != want(quantizable) || !strings.Contains(imp.Size.Basis, "Not a measurement") {
		t.Errorf("Auto size %+v, want ESTIMATED %s", imp.Size, want(quantizable))
	}
	// Without a candidate the memory figure is the estimated weights: a lower
	// bound, never a measurement.
	if imp.Memory.State != dashboard.Estimated || !strings.Contains(imp.Memory.Value, "weights only") || !strings.Contains(imp.Memory.Basis, "lower bound") ||
		!imp.MemoryUsage.LowerBound || imp.MemoryUsage.Bytes == 0 {
		t.Errorf("Auto memory %+v / %+v, want an ESTIMATED lower bound", imp.Memory, imp.MemoryUsage)
	}
	// Carried files are artifact size, never device-resident memory.
	if carried == 0 || int64(imp.MemoryUsage.Bytes) > int64(float64(total)-float64(quantizable)*(1-w4a16BytesPerBF16Byte))+1 {
		t.Errorf("memory lower bound %d includes carried files (%d bytes)", imp.MemoryUsage.Bytes, carried)
	}
	notChecked("latency", imp.Latency)
	notChecked("fidelity", imp.Fidelity)
	pinned := pinnedProfile(t, a, "maximum-fidelity", tuning.RegionFeedForward)
	if est := impact(pinned).Size; est.State != dashboard.Estimated || est.Value != want(quantizable-16*8*2) {
		t.Errorf("pinned size %+v, want ESTIMATED %s", est, want(quantizable-16*8*2))
	}
	if err := store.SaveProfile(pinned, a); err != nil {
		t.Fatal(err)
	}

	// a candidate built from the pinned profile is MEASURED, only for that profile
	compiled, _ := tuning.Compile(pinned, a)
	v := home.VariantManifest{
		Source: home.SourceOf(m), Provider: m.Provider,
		Optimizer: home.Optimizer{Engine: recipe.Engine, Version: "1"}, Recipe: compiled.Recipe,
		Tuning:   func() *home.TuningProvenance { p := tuning.Provenance(pinned, a, compiled); return &p }(),
		Weights:  home.WeightPrecision{Scheme: recipe.Scheme, Bits: 4, GroupSize: 128, Format: "compressed-tensors", DType: "bfloat16"},
		Files:    map[string]string{"model.safetensors": strings.Repeat("ab", 32), "config.json": strings.Repeat("cd", 32)},
		Creation: home.Creation{CreatedAt: "2026-10-03T00:00:00Z"},
	}
	for _, f := range compiled.Recipe.Carry {
		v.Files[f] = strings.Repeat("ef", 32)
	}
	v.Seal()
	vdir := h.VariantDir(m.ID, v.ID)
	if err := os.MkdirAll(vdir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(vdir, "model.safetensors"), make([]byte, 3<<20), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(vdir, "config.json"), make([]byte, 1<<20), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, f := range compiled.Recipe.Carry {
		if err := os.MkdirAll(filepath.Dir(filepath.Join(vdir, f)), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(vdir, f), make([]byte, 1<<20), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if !slices.Contains(compiled.Recipe.Carry, "joint_head.safetensors") {
		t.Fatalf("fixture recipe carries no weights-like file: %v", compiled.Recipe.Carry)
	}
	if err := home.WriteJSON(filepath.Join(vdir, home.VariantManifestFile), v); err != nil {
		t.Fatal(err)
	}
	imp = impact(pinned)
	if imp.Size.State != dashboard.Measured || imp.Size.Value != gib(int64(4+len(compiled.Recipe.Carry))<<20) || !strings.Contains(imp.Size.Basis, v.ID) {
		t.Errorf("measured size %+v", imp.Size)
	}
	if imp.Memory.State != dashboard.Estimated || imp.MemoryUsage != (tuning.Usage{Bytes: 3 << 20, LowerBound: true}) {
		t.Errorf("memory without a probe %+v / %+v, want the candidate's weights as an ESTIMATED lower bound", imp.Memory, imp.MemoryUsage)
	}
	notChecked("latency without a probe", imp.Latency)
	notChecked("fidelity without a certification", imp.Fidelity)
	if other := impact(auto).Size; other.State != dashboard.Estimated {
		t.Errorf("another profile's size is %+v: a candidate of one profile was attributed to another", other)
	}

	probe := app.ProbeRecord{Schema: app.ProbeSchema, Result: "passed", Variant: v.ID, Device: "cuda", VariantManifestSHA256: "stale", StartedAt: "2026-10-03T01:00:00Z",
		Timing: app.ProbeTiming{RequestMS: 12.5}, Resources: app.ProbeResources{VRAMAllocated: 3 << 30}}
	if err := app.SaveProbe(h, probe); err != nil {
		t.Fatal(err)
	}
	imp = impact(pinned)
	notChecked("latency from a probe of another manifest", imp.Latency)
	if imp.Memory.State == dashboard.Measured || !imp.MemoryUsage.LowerBound {
		t.Errorf("memory from a probe of another manifest is measured: %+v", imp.Memory)
	}
	probe.VariantManifestSHA256 = v.ManifestSHA256()
	if err := app.SaveProbe(h, probe); err != nil {
		t.Fatal(err)
	}
	imp = impact(pinned)
	if imp.Latency.State != dashboard.Measured || imp.Latency.Value != "12.5 ms per request" || !strings.Contains(imp.Latency.Basis, "cuda") ||
		imp.Memory.State != dashboard.Measured || imp.Memory.Value != "VRAM reserved 3.00 GiB" || imp.MemoryUsage != (tuning.Usage{Bytes: 3 << 30}) {
		t.Errorf("probe measurements: %+v / %+v", imp.Latency, imp.Memory)
	}
	notChecked("fidelity without a certification", imp.Fidelity)
}

// The desktop composition hosts Tuning beside Models and Forge: it is in the
// navigation, and over a home with no materialized source it says so rather
// than showing regions it did not analyze.
func TestDesktopHostsTheTuningWorkspace(t *testing.T) {
	d := newResidentDesktop(t, filepath.Join(t.TempDir(), "desktop.json"))
	d.run(t, func(base string) {
		body := page(t, base, "/tuning")
		if !strings.Contains(body, `<a href="/tuning" aria-current="page">Tuning</a>`) {
			t.Error("Tuning is not in the desktop navigation")
		}
		if !strings.Contains(body, `id="tuning-unavailable"`) || strings.Contains(body, `data-region=`) || strings.Contains(body, "/tuning/build") {
			t.Errorf("an unmaterialized source is analyzed or buildable:\n%s", body)
		}
		post(t, base, "/tuning/build", url.Values{"source": {setup.ClefFlash}, "objective": {"balanced"}})
		if body := page(t, base, "/tuning"); !strings.Contains(body, "build candidate "+setup.ClefFlash) || !strings.Contains(body, "FAILED") {
			t.Error("a build request without an analysis is not refused visibly")
		}
	})
}

// The production layout reader is intentionally pinned to the exact Clef-Flash
// checkpoint vocabulary. Keep representative names from every Linear family in
// the pinned weight map here so a naming drift fails closed instead of silently
// dropping a semantic region.
func TestClefLinearModuleMatchesPinnedCheckpointVocabulary(t *testing.T) {
	t.Parallel()
	linear := []string{
		"lm_head",
		"model.language_model.layers.0.linear_attn.in_proj_a",
		"model.language_model.layers.0.linear_attn.in_proj_b",
		"model.language_model.layers.0.linear_attn.in_proj_qkv",
		"model.language_model.layers.0.linear_attn.in_proj_z",
		"model.language_model.layers.0.linear_attn.out_proj",
		"model.language_model.layers.0.mlp.gate_proj",
		"model.language_model.layers.0.mlp.up_proj",
		"model.language_model.layers.0.mlp.down_proj",
		"model.language_model.layers.3.self_attn.q_proj",
		"model.language_model.layers.3.self_attn.k_proj",
		"model.language_model.layers.3.self_attn.v_proj",
		"model.language_model.layers.3.self_attn.o_proj",
		"model.visual.blocks.0.attn.qkv",
		"model.visual.blocks.0.attn.proj",
		"model.visual.blocks.0.mlp.linear_fc1",
		"model.visual.blocks.0.mlp.linear_fc2",
		"model.visual.merger.linear_fc1",
		"model.visual.merger.linear_fc2",
	}
	for _, name := range linear {
		if !clefLinearModule(name) {
			t.Errorf("pinned Linear module %q is not recognized", name)
		}
	}
	notLinear := []string{
		"model.language_model.embed_tokens",
		"model.language_model.layers.0.input_layernorm",
		"model.language_model.layers.0.post_attention_layernorm",
		"model.language_model.layers.0.linear_attn.conv1d",
		"model.language_model.layers.0.linear_attn.norm",
		"model.language_model.layers.3.self_attn.q_norm",
		"model.language_model.layers.3.self_attn.k_norm",
		"model.visual.blocks.0.norm1",
		"model.visual.blocks.0.norm2",
		"model.visual.merger.norm",
		"model.visual.patch_embed.proj",
		"model.visual.pos_embed",
	}
	for _, name := range notLinear {
		if clefLinearModule(name) {
			t.Errorf("non-Linear pinned module %q is classified as Linear", name)
		}
	}
}
