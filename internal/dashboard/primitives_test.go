package dashboard

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/i18n"
)

func primitiveHTML(t *testing.T, locale i18n.Locale, name string, data any) string {
	t.Helper()
	var b bytes.Buffer
	if err := pages[locale].ExecuteTemplate(&b, name, data); err != nil {
		t.Fatal(err)
	}
	return b.String()
}

func TestSharedControlProjection(t *testing.T) {
	for _, mode := range []SemanticState{Intent, Auto, Overridden} {
		body := primitiveHTML(t, i18n.English, "operator-row", ControlProjection{Label: "Device", Mode: mode, Value: "example-device <exact>", Reason: "backend resolution"})
		for _, want := range []string{`data-control="` + string(mode) + `"`, `data-state="` + string(mode) + `"`, "example-device &lt;exact&gt;", "backend resolution"} {
			if !strings.Contains(body, want) {
				t.Errorf("%s lacks %q: %s", mode, want, body)
			}
		}
		if strings.Contains(body, "<script") || strings.Contains(body, "<input") {
			t.Error("projection introduced value resolution or input")
		}
	}
	for _, state := range []SemanticState{Auto, Overridden, Measured, Estimated, Preserved, NotChecked} {
		body := primitiveHTML(t, i18n.Japanese, "semantic-state", state)
		if !i18n.Japanese.Has(string(state)) || !strings.Contains(body, i18n.Japanese.T(string(state))) {
			t.Errorf("state %s is not localized", state)
		}
	}
}

func TestSharedDisclosureComposition(t *testing.T) {
	// Compose native controls and output in the production template set. Data
	// remains escaped, and disclosure content needs no template.HTML bypass.
	tmpl, err := pageBase.Clone()
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err = tmpl.Parse(`{{define "composed-primitives"}}{{template "instrument-section-open" .Section}}{{template "operator-advanced-open" .Disclosure}}<label>Override<input name="override" value="{{.Value}}"></label>{{template "operator-disclosure-close"}}{{template "operator-details-open" .Disclosure}}<output>{{.Value}}</output>{{template "operator-disclosure-close"}}{{template "operator-evidence-open" .Disclosure}}<p>{{.Value}}</p>{{template "operator-disclosure-close"}}{{template "instrument-section-close"}}{{end}}`)
	if err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	err = tmpl.ExecuteTemplate(&b, "composed-primitives", struct {
		Section    SectionProjection
		Disclosure DisclosureProjection
		Value      string
	}{SectionProjection{ID: "example", Label: "Device"}, DisclosureProjection{Label: "Device"}, "<exact-value>"})
	if err != nil {
		t.Fatal(err)
	}
	body := b.String()
	for _, want := range []string{`aria-labelledby="example-heading"`, `id="example-heading"`, `data-disclosure="advanced"`, `data-disclosure="details"`, `data-disclosure="evidence"`, `<summary>Advanced`, `<summary>Details`, `<summary>Evidence`, `value="&lt;exact-value&gt;"`, `<output>&lt;exact-value&gt;</output>`} {
		if !strings.Contains(body, want) {
			t.Errorf("composition lacks %q", want)
		}
	}
	if strings.Contains(body, " open") {
		t.Error("optional disclosure is open by default")
	}
}

func TestSharedOperationPhaseProjection(t *testing.T) {
	plan := []string{"validating", "rebinding", "rolling_back"}
	busy := ModelOp{Kind: "apply", Plan: plan, Phases: plan[:2], Phase: "rebinding"}
	body := primitiveHTML(t, i18n.English, "operation-phases", busy)
	for _, want := range []string{`class="done">Validating`, `class="current" aria-current="step">Rebinding runtime`, `class="pending">Rolling back`} {
		if !strings.Contains(body, want) {
			t.Errorf("busy projection lacks %q", want)
		}
	}
	failed := ModelOp{Kind: "apply", Plan: plan, Phases: plan, Phase: "rolling_back", FailurePhase: "rebinding", Failure: "exact target refused; previous serving target restored", Finished: time.Now()}
	body = primitiveHTML(t, i18n.English, "op-last", failed)
	for _, want := range []string{`class="failed">Rebinding runtime`, `class="done">Rolling back`, failed.Failure} {
		if !strings.Contains(body, want) {
			t.Errorf("failure/rollback projection lacks %q", want)
		}
	}
}

func TestSharedPrimitiveStylesReachProductionShell(t *testing.T) {
	body := newEnv(t).get(t, "/").Body.String()
	for _, selector := range []string{".instrument-section", ".operator-row", ".operator-disclosure"} {
		rule := cssRule(t, body, selector)
		for _, banned := range []string{"background", "box-shadow", "border-radius"} {
			if strings.Contains(rule, banned) {
				t.Errorf("%s carries %s", selector, banned)
			}
		}
	}
	for _, want := range []string{"@media (max-width: 42rem)", ".instrument-heading, .operator-row { grid-template-columns: minmax(0, 1fr); }", ":focus-visible", "prefers-reduced-motion"} {
		if !strings.Contains(body, want) {
			t.Errorf("production shell lacks %q", want)
		}
	}
}
