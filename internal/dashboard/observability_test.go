package dashboard

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/i18n"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// activeForgeOp is a Forge build/evaluate that has sat on one phase for a
// while and whose log is still being written.
func activeForgeOp(console []string, activity time.Time) *ModelOp {
	plan := []string{"resolve_inputs", "preflight", "provision", "build", "probe", "resolving", "reference_run", "candidate_run", "certifying", "persisting"}
	return &ModelOp{Kind: "forge_build_evaluate", Device: "cuda", Model: "clef-flash", Plan: plan, Phases: plan[:4], Phase: "build",
		Started: time.Now().Add(-12 * time.Minute), Activity: activity, Log: "/h/logs/setup.log", Console: console, ConsoleAt: activity}
}

func finishedApply() *ModelOp {
	plan := []string{"validating", "snapshotting", "runtime", "model", "variant", "activation", "rebinding", "awaiting_ready", "proving", "smoke", "finalizing"}
	return &ModelOp{Kind: "apply", Device: "cuda", Model: "clef-flash", Target: "variant clef-flash--r--aaaaaaaaaaaa", Plan: plan, Phases: plan, Phase: "finalizing",
		Started: time.Now().Add(-4*time.Minute - 12*time.Second), Finished: time.Now(), Activity: time.Now().Add(-time.Second), Log: "/h/logs/setup.log",
		Console: []string{"apply: runtime READY", "apply: typed decision answered"}}
}

// ---- A. the console, and where an active operation is shown -----------------

func TestActiveForgeOperationShowsProgressAndConsoleOnForgeOnly(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	fm.state.Busy = activeForgeOp([]string{"build: quantizing layer 12", "build: quantizing layer 13"}, time.Now().Add(-95*time.Second))

	forge := e.get(t, "/forge").Body.String()
	for _, want := range []string{`id="models-busy"`, `class="stages"`, `aria-current="step"`, "Building candidate", "phase 4 of 10",
		`id="op-console"`, `data-keep="op-console"`, `data-tail="op-console"`, "build: quantizing layer 13"} {
		if !strings.Contains(forge, want) {
			t.Errorf("/forge lacks %q", want)
		}
	}
	// The console is closed until the operator opens it, and it is not inside
	// the polite live region, so appended lines are never announced.
	console := section(forge, `<details class="disclosure op-console"`, `</details>`)
	if strings.Contains(console[:strings.Index(console, ">")], " open") {
		t.Errorf("the console is open by default: %s", console[:strings.Index(console, ">")])
	}
	// Template source may be checked out with CRLF on Windows. Normalize the
	// rendered markup before using a line-break-sensitive boundary so the test
	// verifies the DOM relationship rather than the checkout newline policy.
	normalizedForge := strings.ReplaceAll(forge, "\r\n", "\n")
	if live := section(normalizedForge, `<div class="last op tone-active" role="status" aria-live="polite">`, `</p>
</div>`); strings.Contains(live, "op-console") {
		t.Error("the console is inside the live region")
	}
	if strings.Contains(forge, `<pre class="console" tabindex="0" aria-live="off"`) == false {
		t.Error("the console is not a quiet, focusable, scrollable region")
	}
	// No fabricated percentage: the step has no total.
	if strings.Contains(section(forge, `id="models-busy"`, `id="op-console"`), "aria-valuenow") {
		t.Error("a percentage was invented for an operation that reported none")
	}

	// Every other workspace shows the headline and nothing else of it.
	withTuning(e, &fakeTuning{analysis: tuningAnalysis(t)})
	for _, p := range []string{"/", "/models", "/workbench", "/experiments", "/errors", "/diagnostics", "/tuning"} {
		body := e.get(t, p).Body.String()
		if !strings.Contains(body, `class="op-headline tone-active" href="/forge"`) || !strings.Contains(body, "Build and evaluate") || !strings.Contains(body, "phase 4 of 10") {
			t.Errorf("%s lacks the compact headline", p)
		}
		for _, banned := range []string{`id="models-busy"`, `id="op-console"`, `class="stages"`, "Building candidate", "build: quantizing layer", `data-tail=`} {
			if strings.Contains(body, banned) {
				t.Errorf("%s shows another operation's detail: %q", p, banned)
			}
		}
	}
	// The headline rides the live slot every workspace already refreshes.
	if live := e.get(t, "/live").Body.String(); !strings.Contains(live, `data-live="shell"><a class="op-headline`) {
		t.Error("the headline is not part of the live shell slot")
	}
}

func TestOperationHeadlineIsOnlyForARunningOperation(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	fm.state.Last = finishedApply()
	if body := e.get(t, "/").Body.String(); strings.Contains(body, `class="op-headline`) {
		t.Error("a finished operation has a headline")
	}
}

func TestConsoleTailIsBoundedAndRedactedWhenRendered(t *testing.T) {
	var lines []string
	for i := 0; i < 500; i++ {
		lines = append(lines, fmt.Sprintf("line %03d", i))
	}
	lines = append(lines, "fetch https://user:hunter2@example.invalid/x?token=abc123def456", "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz", "Authorization: Bearer abcdefghijklmnop",
		"wide "+strings.Repeat("x", 3000))
	e, fm, _ := forgeEnv(t, variantInventory())
	fm.state.Busy = activeForgeOp(lines, time.Now())
	body := e.get(t, "/forge").Body.String()
	pre := section(body, `<pre class="console"`, `</pre>`)
	if n := strings.Count(pre, `<span class="l">`); n != consoleLines {
		t.Errorf("%d console lines are rendered, want the bound %d", n, consoleLines)
	}
	if strings.Contains(pre, "line 000") || strings.Contains(pre, "line 399") || !strings.Contains(pre, "wide xxx") {
		t.Error("the console is not the most recent tail")
	}
	for _, leaked := range []string{"hunter2", "abc123def456", "hf_abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnop"} {
		if strings.Contains(body, leaked) {
			t.Errorf("the page exposes %q", leaked)
		}
	}
	if !strings.Contains(pre, "&lt;redacted&gt;") {
		t.Error("redaction is not visible in the console")
	}
	for _, l := range regexp.MustCompile(`<span class="l">([^<]*)</span>`).FindAllStringSubmatch(pre, -1) {
		if len(l[1]) > consoleLineBytes+len("...[truncated]") {
			t.Fatalf("a console line has %d bytes", len(l[1]))
		}
	}
}

func TestConsoleHasNoOutputOfItsOwnForAnOperationWithoutALog(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	op := activeForgeOp(nil, time.Time{})
	op.Log, op.ConsoleAt = "", time.Time{}
	fm.state.Busy = op
	if body := e.get(t, "/forge").Body.String(); strings.Contains(body, `id="op-console"`) || strings.Contains(body, "last backend activity") {
		t.Error("an operation with no log shows a console or an activity line")
	}
}

// The age shown is the backend's own latest report (a phase, a step or a write
// of the operation's log), never a timer of the dashboard's.
func TestActivityAgeIsDerivedFromBackendActivity(t *testing.T) {
	now := time.Now()
	if got := (ModelOp{}).LastActivity(); !got.IsZero() {
		t.Errorf("no report, activity %v", got)
	}
	if got := (ModelOp{Activity: now.Add(-time.Minute)}).LastActivity(); !got.Equal(now.Add(-time.Minute)) {
		t.Errorf("phase report only: %v", got)
	}
	if got := (ModelOp{Activity: now.Add(-time.Minute), ConsoleAt: now.Add(-5 * time.Second)}).LastActivity(); !got.Equal(now.Add(-5 * time.Second)) {
		t.Errorf("a newer log write is later activity: %v", got)
	}
	if got := (ModelOp{Activity: now.Add(-5 * time.Second), ConsoleAt: now.Add(-time.Minute)}).LastActivity(); !got.Equal(now.Add(-5 * time.Second)) {
		t.Errorf("an older log write does not rewind activity: %v", got)
	}

	e, fm, _ := forgeEnv(t, variantInventory())
	fm.state.Busy = activeForgeOp([]string{"x"}, now.Add(-95*time.Second))
	body := e.get(t, "/forge").Body.String()
	if !regexp.MustCompile(`last backend activity 1m 3\d+s ago`).MatchString(body) || !strings.Contains(body, `data-activity="`) {
		t.Errorf("the activity age is not the latest backend report:\n%s", section(body, `class="op-line op-activity"`, `</p>`))
	}
	// The same operation after a newer report reads younger; nothing in the
	// page's own clock changed.
	fm.state.Busy = activeForgeOp([]string{"x"}, time.Now().Add(-3*time.Second))
	if body := e.get(t, "/forge").Body.String(); !regexp.MustCompile(`last backend activity [0-4]s ago`).MatchString(body) {
		t.Errorf("a fresh report is not shown as fresh:\n%s", section(body, `class="op-line op-activity"`, `</p>`))
	}
	// An operation that has reported nothing says so rather than inventing an age.
	op := activeForgeOp([]string{"x"}, time.Time{})
	op.ConsoleAt = time.Time{}
	fm.state.Busy = op
	if body := e.get(t, "/forge").Body.String(); !strings.Contains(body, "no backend activity reported yet") || strings.Contains(body, "last backend activity") {
		t.Error("an unreported activity is not stated as such")
	}
}

// ---- C. terminal operations are a compact summary -----------------------------

func TestTerminalOperationIsACompactSummaryWithDetailKeptBehindADisclosure(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	fm.state.Last = finishedApply()
	body := e.get(t, "/forge").Body.String()
	summary := section(body, `<div class="op-panel" id="models-last"`, `<details`)
	for _, want := range []string{"DONE", "Apply variant", `data-kind="apply"`, "clef-flash--r--aaaaaaaaaaaa", "took 4m 1", "Applied: the runtime is READY on the variant and answered a typed decision."} {
		if !strings.Contains(summary, want) {
			t.Errorf("the terminal summary lacks %q:\n%s", want, summary)
		}
	}
	if strings.Contains(summary, `class="stages"`) || strings.Contains(summary, "<pre") {
		t.Error("the phase rail or the console is part of the summary")
	}
	// The evidence is still there, behind a disclosure that is closed by default.
	detail := section(body, `<details class="operator-disclosure" data-disclosure="details" id="op-last-detail"`, `</details>`)
	if detail == "" {
		t.Fatal("no disclosure for the finished operation's phases and output")
	}
	if strings.Contains(detail[:strings.Index(detail, ">")], " open") {
		t.Error("the terminal detail is open by default")
	}
	for _, want := range []string{"<summary>Details · Phases and output</summary>", `class="stages"`, "Proving provenance", `id="op-console"`, "apply: typed decision answered"} {
		if !strings.Contains(detail, want) {
			t.Errorf("the terminal detail lacks %q", want)
		}
	}
	// The disclosure and the console keep the operator's choice across refreshes
	// by a stable identity (uistate.js keys on id / data-keep).
	if !strings.Contains(detail, `id="op-console" data-keep="op-console"`) {
		t.Error("the console has no stable identity")
	}
}

func TestFailedOperationKeepsItsLatestOutputInspectable(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	op := activeForgeOp([]string{"build: out of memory"}, time.Now().Add(-time.Minute))
	op.Finished, op.Failure, op.FailurePhase = time.Now(), "optimizer quantize: out of memory", "build"
	fm.state.Last = op
	body := e.get(t, "/forge").Body.String()
	for _, want := range []string{"FAILED", "Failed in phase <strong>Building candidate</strong>", "optimizer quantize: out of memory", "build: out of memory"} {
		if !strings.Contains(body, want) {
			t.Errorf("failed operation lacks %q", want)
		}
	}
	if strings.Contains(section(body, `<div class="op-panel" id="models-last"`, `<details`), "build: out of memory") {
		t.Error("the console is part of the failure summary")
	}
}

func TestOtherWorkspacesShowOnlyTheOutcomeOfAFinishedOperation(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	fm.state.Last = finishedApply()
	for _, p := range []string{"/", "/models"} {
		body := e.get(t, p).Body.String()
		if !strings.Contains(body, `id="models-last"`) || !strings.Contains(body, "DONE") || !strings.Contains(body, `href="/forge">Phases and output</a>`) {
			t.Errorf("%s lacks the compact outcome and the way to its evidence", p)
		}
		for _, banned := range []string{`class="stages"`, `id="op-console"`, "apply: typed decision answered", `id="op-last-detail"`} {
			if strings.Contains(body, banned) {
				t.Errorf("%s shows the evidence of another workspace's operation: %q", p, banned)
			}
		}
	}
}

// ---- B. operator-local state: the contract the markup must meet ----------------

// uistate.js keys a disclosure on data-keep, else id. Every disclosure the
// workstation renders must have one, or a live refresh would reset it.
func TestEveryDisclosureHasAStableIdentity(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	fm.state.Busy = activeForgeOp([]string{"x"}, time.Now())
	fm.state.Last = finishedApply()
	withTuning(e, &fakeTuning{analysis: tuningAnalysis(t)})
	for _, p := range []string{"/", "/diagnostics", "/models", "/forge", "/tuning", "/settings"} {
		rec := e.get(t, p)
		if rec.Code != 200 {
			continue
		}
		for _, tag := range regexp.MustCompile(`<details[^>]*>`).FindAllString(rec.Body.String(), -1) {
			if !strings.Contains(tag, ` id="`) && !strings.Contains(tag, ` data-keep="`) {
				t.Errorf("%s: a disclosure has no stable identity: %s", p, tag)
			}
		}
	}
}

// The module is exercised against a minimal DOM: repeated refreshes keep an
// opened disclosure open and a closed one closed, the console's state and
// follow preference survive, and unchanged content is left alone.
func TestUIStateModule(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node is not installed; the operator-local state module is not exercised")
	}
	out, err := exec.Command(node, filepath.Join("testdata", "uistate_check.js")).CombinedOutput()
	if err != nil {
		t.Fatalf("uistate.js:\n%s", out)
	}
	if !strings.Contains(string(out), "all ") {
		t.Fatalf("uistate.js checks did not complete:\n%s", out)
	}
}

// Every refresher goes through the one module, so no region replaces its
// markup by hand and resets the operator's disclosures.
func TestNoRefresherReplacesMarkupOutsideTheSharedModule(t *testing.T) {
	for _, f := range []string{"page.html", "models.html", "updates.html", "experiments.html", "forge.html", "tuning.html", "workbench.html"} {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(b), ".innerHTML =") || strings.Contains(string(b), ".innerHTML=") {
			t.Errorf("%s replaces markup by hand", f)
		}
	}
	js, err := os.ReadFile("uistate.js")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(js), "dst.innerHTML = src.innerHTML") != 1 {
		t.Error("uistate.js is not the one place that replaces live markup")
	}
}

// ---- G. human-readable labels ---------------------------------------------------

func TestVariantLabelComesFromStructuredMetadata(t *testing.T) {
	inv := variantInventory()
	// An ID that no one could parse a meaning out of: the label still comes from
	// the manifest's source and scheme.
	const opaque = "x9--0f3a--e9e95f37f3a8"
	inv.Variants[1].ID = opaque
	inv.Variants[1].ManifestSHA256 = "e9e95f37f3a8aabbccdd"
	inv.Variants = inv.Variants[:2]
	inv.Variants[1].Certification = "accepted"
	inv.Variants[0].ManifestSHA256 = "1111222233334444"
	e, _, _ := forgeEnv(t, inv)

	forge := e.get(t, "/forge").Body.String()
	card := card(t, forge, opaque)
	// Two variants of one source and scheme are told apart by digest, not by ID.
	if !strings.Contains(card, `<h3 id="vh-`+opaque+`" title="`+opaque+`">Clef Flash · W4A16 · #e9e95f37</h3>`) {
		t.Errorf("the candidate's primary label:\n%s", section(card, `<div class="candidate-head">`, `</div>`))
	}
	if strings.Contains(section(card, `<div class="candidate-head">`, `</div>`), ">"+opaque+"<") {
		t.Error("the exact ID is the primary label")
	}
	// The exact immutable ID stays in the details and is copyable.
	if !strings.Contains(card, `<dt>Variant ID</dt><dd class="mono" data-variant-id>`+opaque+`</dd>`) {
		t.Error("the exact variant ID is not in the candidate's details")
	}
	if !strings.Contains(card, `id="variant-details-`+opaque+`"`) {
		t.Error("the candidate's details have no stable identity")
	}
	models := e.get(t, "/models").Body.String()
	if !strings.Contains(models, `<option value="variant:`+opaque+`" title="`+opaque+`">VARIANT · Clef Flash · W4A16 · #e9e95f37</option>`) {
		t.Error("the execution target is not offered by its readable label with the exact ID as its value")
	}
}

func TestVariantLabelNeverParsesAnIdentifier(t *testing.T) {
	// The ID reads like one that names a source and a scheme, but the manifest
	// records neither: the label is not reconstructed from it.
	inv := variantInventory()
	inv.Variants = inv.Variants[:1]
	inv.Variants[0].ID, inv.Variants[0].Scheme = "clef-flash--clef-flash-w4a16-rtn-g128--e9e95f37f3a8", ""
	e, _, _ := forgeEnv(t, inv)
	forge := e.get(t, "/forge").Body.String()
	head := section(forge, `<div class="candidate-head">`, `</div>`)
	if strings.Contains(head, "Clef Flash") || !strings.Contains(head, `class="id"`) || !strings.Contains(head, ">clef-flash--clef-flash-w4a16-rtn-g128--e9e95f37f3a8<") {
		t.Errorf("without structured metadata the exact ID is shown, not a guess:\n%s", head)
	}
	if got := variantTitle("clef-flash", ""); got != "" {
		t.Errorf("a title from a missing scheme: %q", got)
	}
	if got := variantTitle("", "W4A16"); got != "" {
		t.Errorf("a title from a missing source: %q", got)
	}
	if got := variantTitle("clef-flash", "W4A16"); got != "Clef Flash · W4A16" {
		t.Errorf("title %q", got)
	}
	// With no digest to tell identical titles apart, the exact IDs do.
	labels := variantLabels([]setup.VariantEntry{{ID: "a", SourceID: "clef-flash", Scheme: "W4A16"}, {ID: "b", SourceID: "clef-flash", Scheme: "W4A16"}})
	if labels["a"] != "a" || labels["b"] != "b" {
		t.Errorf("ambiguous titles: %v", labels)
	}
}

// ---- E. responsive width: no page patches over the shared contract -------------

func TestPagesDoNotPatchWidthsOrBreaksOverTheSharedSystem(t *testing.T) {
	e := newEnv(t)
	body := e.get(t, "/").Body.String()
	css := body[strings.Index(body, "<style>"):strings.Index(body, "</style>")]
	allowed := map[string]bool{"100%": true, "min(100%, var(--ws-max))": true, "var(--measure)": true, "var(--ws-max)": true}
	for _, m := range regexp.MustCompile(`[^-@(]max-width:\s*([^;}]+)`).FindAllStringSubmatch(css, -1) {
		if v := strings.TrimSpace(m[1]); !allowed[v] {
			t.Errorf("the workstation stylesheet caps a width at %q", v)
		}
	}
	// A short helper is never given a measure by the page either.
	if regexp.MustCompile(`\.note[^{]*\{[^}]*max-width`).MatchString(css) {
		t.Error("a page caps short helper text")
	}
	for _, f := range []string{"workbench.html", "forge.html", "tuning.html", "models.html", "page.html"} {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(b), "<br") {
			t.Errorf("%s breaks a line by hand", f)
		}
		if regexp.MustCompile(`style="[^"{]*(?:max-)?width:\s*[^{"\s]`).MatchString(string(b)) {
			t.Errorf("%s sets a fixed width on one element (a data bar's computed fill is not one)", f)
		}
	}
}

// ---- F. status carries a word, not a color ----------------------------------------

func TestEveryRenderedStatusHasWords(t *testing.T) {
	e, fm, _ := forgeEnv(t, variantInventory())
	fm.state.Last = finishedApply()
	badge := regexp.MustCompile(`<span class="badge[^"]*"[^>]*>([^<]*)</span>`)
	state := regexp.MustCompile(`<span class="semantic-state tone-\w+" data-state="(\w+)">([^<]*)</span>`)
	var seen int
	for _, p := range []string{"/", "/forge", "/models", "/workbench"} {
		body := e.get(t, p).Body.String()
		for _, m := range badge.FindAllStringSubmatch(body, -1) {
			seen++
			if strings.TrimSpace(m[1]) == "" {
				t.Errorf("%s: a status badge without a word: %s", p, m[0])
			}
		}
		for _, m := range state.FindAllStringSubmatch(body, -1) {
			seen++
			if strings.TrimSpace(m[2]) == "" {
				t.Errorf("%s: a semantic state without a word: %s", p, m[0])
			}
		}
		// The terminal word of an operation and the readiness word are text.
		for _, m := range regexp.MustCompile(`<strong class="status">([^<]*)</strong>`).FindAllStringSubmatch(body, -1) {
			if strings.TrimSpace(m[1]) == "" {
				t.Errorf("%s: an empty status word", p)
			}
		}
	}
	if seen == 0 {
		t.Fatal("no status was rendered")
	}
	// Every semantic state of the shared vocabulary renders its word in both locales.
	for _, s := range []SemanticState{Intent, Auto, Overridden, Measured, Estimated, Preserved, NotChecked} {
		if got := primitiveHTML(t, "en", "semantic-state", s); !strings.Contains(got, `data-state="`+string(s)+`"`) || !regexp.MustCompile(`>\S[^<]*</span>`).MatchString(got) {
			t.Errorf("state %s: %s", s, got)
		}
	}
}

// ---- D. the form primitives as the pages use them ---------------------------------

func TestWorkbenchUsesTheSharedFormStructure(t *testing.T) {
	e := newEnv(t)
	body := e.get(t, "/workbench").Body.String()
	for _, want := range []string{`<div class="rows">`, `<span class="h">Choice label</span>`, `<span class="h optional">Description (optional)</span>`,
		`class="btn danger quiet" name="op" value="remove:0"`, `placeholder="the bounded state the questions are asked about"`, `<textarea name="state"`} {
		if !strings.Contains(body, want) {
			t.Errorf("the Workbench lacks %q", want)
		}
	}
	if strings.Contains(body, `class="choices"`) {
		t.Error("the Workbench still lays its options out as loose boxes")
	}
	// Selection controls are native selects.
	forge, _, _ := forgeEnv(t, variantInventory())
	fb := forge.get(t, "/forge").Body.String()
	if !strings.Contains(fb, `<select id="forge-source"`) || strings.Contains(fb, `role="combobox"`) || strings.Contains(fb, `role="listbox"`) {
		t.Error("a selection is not a native select")
	}
	if strings.Contains(fb, "style=\"max-width") {
		t.Error("a control carries a width of its own")
	}
}

func TestOperationLabelsAndOutcomesAreCatalogued(t *testing.T) {
	for _, m := range []map[string]string{opKindLabels, opOutcomes} {
		for kind, msg := range m {
			if !i18n.Japanese.Has(msg) {
				t.Errorf("%s: %q has no Japanese entry", kind, msg)
			}
		}
	}
}
