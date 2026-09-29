package home

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"
)

func env(m map[string]string) func(string) string {
	return func(k string) string { return m[k] }
}

// testLocator returns a locator whose parent directory does not exist yet.
func testLocator(t *testing.T) Locator {
	return Locator{Path: filepath.Join(t.TempDir(), "appdata", "Hachidori", "bootstrap.json")}
}

func writeLocator(t *testing.T, l Locator, data string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(l.Path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(l.Path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
}

func record(t *testing.T, schema, home string) string {
	b, err := json.Marshal(Bootstrap{Schema: schema, Home: home})
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// snapshot lists every path under root (relative), for "nothing else written".
func snapshot(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, p)
		out = append(out, rel)
		return nil
	})
	sort.Strings(out)
	return out
}

// 1. explicit home wins over env and locator.
func TestDiscoverExplicitWins(t *testing.T) {
	l := testLocator(t)
	stored := t.TempDir()
	if _, err := l.Save(stored); err != nil {
		t.Fatal(err)
	}
	explicit := t.TempDir()
	d, err := discover(explicit, env(map[string]string{"HACHIDORI_HOME": t.TempDir()}), &l)
	if err != nil || d.Source != SourceExplicit || d.Home.Root != explicit {
		t.Fatal(d, err)
	}
	// A broken locator is not consulted when a higher authority applies.
	writeLocator(t, l, "{not json")
	if d, err := discover(explicit, env(nil), &l); err != nil || d.Source != SourceExplicit {
		t.Fatal(d, err)
	}
}

// 2. env home wins over the stored locator when no explicit home is passed.
func TestDiscoverEnvBeatsLocator(t *testing.T) {
	l := testLocator(t)
	if _, err := l.Save(t.TempDir()); err != nil {
		t.Fatal(err)
	}
	envHome := t.TempDir()
	d, err := discover("", env(map[string]string{"HACHIDORI_HOME": envHome}), &l)
	if err != nil || d.Source != SourceEnv || d.Home.Root != envHome {
		t.Fatal(d, err)
	}
	writeLocator(t, l, `{"schema":"other/9"}`)
	if d, err := discover("", env(map[string]string{"HACHIDORI_HOME": envHome}), &l); err != nil || d.Source != SourceEnv {
		t.Fatal(d, err)
	}
}

// 3. a valid locator resolves deterministically, normalized.
func TestDiscoverLocatorDeterministic(t *testing.T) {
	l := testLocator(t)
	root := t.TempDir()
	h, err := l.Save(filepath.Join(root, "x", ".."))
	if err != nil || h.Root != root {
		t.Fatal(h, err)
	}
	for i := 0; i < 3; i++ {
		d, err := discover("", env(nil), &l)
		if err != nil || d.Source != SourceLocator || d.Home.Root != root {
			t.Fatal(d, err)
		}
	}
	var b Bootstrap
	if err := ReadJSON(l.Path, &b); err != nil || b != (Bootstrap{Schema: BootstrapSchema, Home: root}) {
		t.Fatal(b, err)
	}
}

// 4. an absent locator is a clean first run and creates nothing.
func TestDiscoverAbsentIsUnconfigured(t *testing.T) {
	l := testLocator(t)
	d, err := discover("", env(nil), &l)
	if err != nil || d.Source != SourceUnconfigured || d.Home != (Home{}) {
		t.Fatal(d, err)
	}
	if _, err := os.Stat(filepath.Dir(l.Path)); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("locator parent created by a read: %v", err)
	}
}

// 5. malformed JSON is distinguishable from absent state.
func TestLocatorMalformed(t *testing.T) {
	abs := t.TempDir()
	for name, data := range map[string]string{
		"syntax":        "{not json",
		"empty":         "",
		"trailing":      record(t, BootstrapSchema, abs) + "{}",
		"extra field":   `{"schema":"` + BootstrapSchema + `","home":` + mustJSON(abs) + `,"device":"cuda"}`,
		"missing home":  `{"schema":"` + BootstrapSchema + `"}`,
		"relative home": record(t, BootstrapSchema, "rel/home"),
		"unclean home":  record(t, BootstrapSchema, abs+string(filepath.Separator)+"."),
	} {
		t.Run(name, func(t *testing.T) {
			l := testLocator(t)
			writeLocator(t, l, data)
			_, found, err := l.Load()
			if !errors.Is(err, ErrBootstrapMalformed) || found {
				t.Fatalf("found=%v err=%v", found, err)
			}
			var be *BootstrapError
			if !errors.As(err, &be) || be.Path != l.Path {
				t.Fatalf("not a typed error: %#v", err)
			}
			d, err := discover("", env(nil), &l)
			if !errors.Is(err, ErrBootstrapMalformed) || d.Source == SourceUnconfigured {
				t.Fatal(d, err)
			}
		})
	}
}

func mustJSON(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

// 6. unknown schema is rejected.
func TestLocatorUnknownSchema(t *testing.T) {
	for _, schema := range []string{"", "hachidori.bootstrap/2", "hachidori.runtime/1"} {
		l := testLocator(t)
		writeLocator(t, l, record(t, schema, t.TempDir()))
		d, err := discover("", env(nil), &l)
		if !errors.Is(err, ErrBootstrapUnknownSchema) || errors.Is(err, ErrBootstrapMalformed) || d.Source == SourceUnconfigured {
			t.Fatalf("schema %q: %v %v", schema, d, err)
		}
	}
}

// 7. a missing stored home is distinct from never configured.
func TestLocatorStoredHomeMissing(t *testing.T) {
	l := testLocator(t)
	root := filepath.Join(t.TempDir(), "gone")
	if err := os.Mkdir(root, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := l.Save(root); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(root); err != nil {
		t.Fatal(err)
	}
	d, err := discover("", env(nil), &l)
	var be *BootstrapError
	if !errors.Is(err, ErrStoredHomeMissing) || !errors.As(err, &be) || be.Home != root || d.Source != SourceLocator {
		t.Fatal(d, err)
	}
	// A file where the home was is also not a home.
	if err := os.WriteFile(root, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := l.Lookup(); !errors.Is(err, ErrStoredHomeMissing) {
		t.Fatal(err)
	}
}

// 8. writes are atomic: a failed replace leaves the previous record intact
// with no temporary debris, and concurrent readers never see a partial record.
func TestLocatorAtomicWrite(t *testing.T) {
	l := testLocator(t)
	a, b := t.TempDir(), t.TempDir()
	if _, err := l.Save(a); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	stop := make(chan struct{})
	errs := make(chan error, 1)
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			h, found, err := l.Lookup()
			if err != nil || !found || (h.Root != a && h.Root != b) {
				select {
				case errs <- errors.Join(err, errors.New("partial or invalid record observed: "+h.Root)):
				default:
				}
				return
			}
		}
	}()
	for i := 0; i < 200; i++ {
		root := a
		if i%2 == 1 {
			root = b
		}
		if _, err := l.Save(root); err != nil {
			t.Fatal(err)
		}
	}
	close(stop)
	wg.Wait()
	select {
	case err := <-errs:
		t.Fatal(err)
	default:
	}
	if got := snapshot(t, filepath.Dir(l.Path)); len(got) != 2 || got[1] != "bootstrap.json" {
		t.Fatalf("temporary debris: %v", got)
	}

	// Failed rename: target replaced by a non-empty directory.
	l2 := testLocator(t)
	if err := os.MkdirAll(filepath.Join(l2.Path, "blocker"), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := l2.Save(a); err == nil {
		t.Fatal("save over a directory succeeded")
	}
	if got := snapshot(t, filepath.Dir(l2.Path)); len(got) != 3 {
		t.Fatalf("temporary debris after failed write: %v", got)
	}
}

// Selecting validates the target as a directory but requires no runtime.
func TestLocatorSaveRequiresDirectory(t *testing.T) {
	l := testLocator(t)
	if _, err := l.Save(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("nonexistent home selected")
	}
	if _, err := l.Save(""); err == nil {
		t.Fatal("empty home selected")
	}
	f := filepath.Join(t.TempDir(), "file")
	os.WriteFile(f, nil, 0o644)
	if _, err := l.Save(f); err == nil {
		t.Fatal("file selected as home")
	}
	if _, err := os.Stat(filepath.Dir(l.Path)); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("locator parent created by a rejected selection: %v", err)
	}
}

// 9. reset removes only the locator.
func TestLocatorForgetRemovesOnlyLocator(t *testing.T) {
	l := testLocator(t)
	root := t.TempDir()
	h := Home{Root: root}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(h.Path("state", "active-runtime.json"), []byte("{}"), 0o644)
	if _, err := l.Save(root); err != nil {
		t.Fatal(err)
	}
	sibling := filepath.Join(filepath.Dir(l.Path), "other.txt")
	os.WriteFile(sibling, []byte("x"), 0o644)
	before := snapshot(t, root)

	if err := l.Forget(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(l.Path); !errors.Is(err, fs.ErrNotExist) {
		t.Fatal("locator not removed")
	}
	if _, err := os.Stat(sibling); err != nil {
		t.Fatal("forget removed more than the locator")
	}
	if after := snapshot(t, root); len(after) != len(before) {
		t.Fatalf("home changed: %v -> %v", before, after)
	}
	if err := l.Forget(); err != nil {
		t.Fatal("forgetting an absent locator:", err)
	}
	if d, err := discover("", env(nil), &l); err != nil || d.Source != SourceUnconfigured {
		t.Fatal(d, err)
	}
}

// 10. selecting/discovering writes nothing but the locator: no runtime,
// model or cache state anywhere, including the selected home.
func TestLocatorWritesOnlyLocator(t *testing.T) {
	base := t.TempDir()
	l := Locator{Path: filepath.Join(base, "Hachidori", "bootstrap.json")}
	root := t.TempDir()
	if _, err := l.Save(root); err != nil {
		t.Fatal(err)
	}
	if _, err := discover("", env(nil), &l); err != nil {
		t.Fatal(err)
	}
	if got := snapshot(t, root); len(got) != 1 {
		t.Fatalf("selected home written to: %v", got)
	}
	if got := snapshot(t, base); len(got) != 3 || got[1] != "Hachidori" || got[2] != filepath.Join("Hachidori", "bootstrap.json") {
		t.Fatalf("unexpected out-of-home state: %v", got)
	}
	var raw map[string]any
	if err := ReadJSON(l.Path, &raw); err != nil || len(raw) != 2 {
		t.Fatalf("record is not minimal: %v %v", raw, err)
	}
}

// discover without a locator (the non-Windows seam) is explicit/env only.
func TestDiscoverWithoutLocator(t *testing.T) {
	d, err := discover("", env(nil), nil)
	if err != nil || d.Source != SourceUnconfigured {
		t.Fatal(d, err)
	}
	d, err = discover("", env(map[string]string{"HACHIDORI_HOME": "rel"}), nil)
	if err != nil || d.Source != SourceEnv || !filepath.IsAbs(d.Home.Root) {
		t.Fatal(d, err)
	}
}
