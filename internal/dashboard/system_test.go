package dashboard

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/history"
	"github.com/yohn-jp/hachidori/internal/ui"
)

// Every workstation workspace renders with the one visual system
// (docs/desktop.md): the shared token and primitive stylesheet is inlined
// once, and no workspace redefines a color, typography or spacing token.
func TestWorkspacesUseTheOneVisualSystem(t *testing.T) {
	e := newEnv(t)
	withSettings(e, &fakeSettings{}, nil)
	sys := string(ui.CSS())
	for _, p := range []string{"/", "/workbench", "/experiments", "/errors", "/diagnostics", "/settings"} {
		body := e.get(t, p).Body.String()
		if strings.Count(body, sys) != 1 {
			t.Errorf("%s does not inline the visual system exactly once", p)
		}
		page := strings.Replace(body, sys, "", 1)
		for _, tok := range []string{"--bg:", "--text:", "--accent:", "--fs-", "--sp-", "--r:", "--ctl-h:", "--focus:", "--dur"} {
			if regexp.MustCompile(regexp.QuoteMeta(tok) + `[a-z0-9-]*:`).MatchString(page) {
				t.Errorf("%s redefines visual-system token %s", p, tok)
			}
		}
	}
}

// Irreversible deletions ask first and read as destructive; the confirmation
// goes through the shell's one submit handler, not per-form script.
func TestIrreversibleActionsConfirm(t *testing.T) {
	e, _, _ := historyEnv(t)
	dataset, defs := expFiles(t)
	e.post(t, "/experiments/run", runForm(dataset, defs))
	waitExp(t, e)
	e.post(t, "/experiments/save", url.Values{"seq": {"1"}, "label": {"a"}})
	exp := e.get(t, "/experiments").Body.String()
	if !regexp.MustCompile(`<form method="post" action="/history/delete" class="inline" data-confirm="[^"]+">.*?class="btn danger">Delete</button>`).MatchString(exp) {
		t.Error("history delete is not a confirmed destructive action")
	}

	m := &fakeModels{state: ModelsState{Inventory: modelsInventory()}}
	withModels(e, m)
	withConnections(e, filepath.Join(t.TempDir(), "settings.json"))
	e.post(t, "/settings/connections/save", profileForm("dev", "dev@host", "7843", "7843"))
	set := e.get(t, "/settings").Body.String()
	if n := strings.Count(set, `action="/settings/models/remove" class="inline" data-confirm=`); n == 0 {
		t.Error("model/runtime removal is not confirmed")
	}
	if !strings.Contains(set, `action="/settings/connections/remove" class="inline" data-confirm=`) {
		t.Error("connection removal is not confirmed")
	}
	if strings.Contains(set+exp, "onsubmit=") {
		t.Error("a form confirms through its own inline handler")
	}
	if !strings.Contains(set, `var ask = (b && b.dataset.confirm) || f.dataset.confirm;`) {
		t.Error("the shell does not honor data-confirm")
	}
}

// The live refresh is bounded and quiet: it skips unchanged slots and hidden
// windows, and announces only the live/stale transition, never each tick.
func TestLiveRefreshIsBoundedAndQuiet(t *testing.T) {
	e := newEnv(t)
	body := e.get(t, "/").Body.String()
	for _, want := range []string{`if (document.hidden) return;`, `if (dst.innerHTML !== src.innerHTML) dst.innerHTML = src.innerHTML;`,
		`document.addEventListener("visibilitychange"`, `<span class="vh" id="sync-live" role="status"></span>`} {
		if !strings.Contains(body, want) {
			t.Errorf("runtime page lacks %q", want)
		}
	}
	if regexp.MustCompile(`id="sync-text"[^>]*(role=|aria-live)`).MatchString(body) {
		t.Error("the ticking sync timestamp is a live region")
	}
}

// The history table is a bounded page of the newest entries with keyboard
// links to older pages, while Compare still offers every entry.
func TestHistoryTableIsPaged(t *testing.T) {
	e, _, root := historyEnv(t)
	dataset, defs := expFiles(t)
	e.post(t, "/experiments/run", runForm(dataset, defs))
	waitExp(t, e)
	e.post(t, "/experiments/save", url.Values{"seq": {"1"}, "label": {"seed"}})
	list, _, err := e.d.hist.List()
	if err != nil || len(list) != 1 {
		t.Fatal(list, err)
	}
	const n = 2*historyPage + 50
	entries := make([]history.Summary, n)
	for i := range entries {
		sm := list[0]
		if i > 0 {
			sm.ID = fmt.Sprintf("20200101T000000Z-%012x", i)
			if err := os.Mkdir(filepath.Join(root, "entries", sm.ID), 0o755); err != nil {
				t.Fatal(err)
			}
		}
		entries[i] = sm
	}
	ix, _ := json.Marshal(map[string]any{"schema": history.IndexSchema, "entries": entries})
	if err := os.WriteFile(filepath.Join(root, "index.json"), ix, 0o644); err != nil {
		t.Fatal(err)
	}
	rows := func(b string) int { return strings.Count(b, `action="/history/open"`) }
	first := e.get(t, "/experiments").Body.String()
	if rows(first) != historyPage || !strings.Contains(first, `href="/experiments?history=100#hist-h">Older</a>`) || strings.Contains(first, ">Newer</a>") {
		t.Fatalf("first page: %d rows", rows(first))
	}
	if got := strings.Count(first, `<option value="`+entries[n-1].ID+`"`); got != 2 {
		t.Errorf("compare offers the oldest entry %d times, want 2", got)
	}
	last := e.get(t, "/experiments?history=200").Body.String()
	if rows(last) != n-2*historyPage || !strings.Contains(last, `href="/experiments?history=100#hist-h">Newer</a>`) || strings.Contains(last, ">Older</a>") ||
		!strings.Contains(last, "201–250 / 250") {
		t.Fatalf("last page: %d rows", rows(last))
	}
	if b := e.get(t, "/experiments?history=99999").Body.String(); rows(b) != n-2*historyPage {
		t.Error("an out-of-range offset is not clamped to the last page")
	}
}
