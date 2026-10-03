package dashboard

import (
	"net/http"
	"path/filepath"
	"slices"
	"strings"
)

type resourceChoice struct {
	Label      string
	FormAction string
	Name       string
	Value      string
}

// resourceOption is one known resource: Value is the exact path (or the
// newline-joined set of paths); Label names it without its directory.
type resourceOption struct {
	Value, Label string
	Selected     bool
}

// resourceSelection is the shared dashboard presentation for choosing one
// resource by its meaning. Known resources are offered by name; a native
// picker adds a new one; the exact path stays observable under a disclosure
// and is editable there only when no native picker exists.
type resourceSelection struct {
	Native      bool
	Label       string
	Name        string
	Value       string
	Placeholder string
	Required    bool
	Multiline   bool
	Choices     []resourceChoice
	// Known lists the resources this host has used, the current one first.
	Known []resourceOption
	// None labels the empty choice of an optional resource.
	None string
}

func resourceInput(native bool, label, name, value, placeholder string, required, multiline bool, formAction, choiceName string, choices ...string) resourceSelection {
	s := resourceSelection{
		Native: native, Label: label, Name: name, Value: value,
		Placeholder: placeholder, Required: required, Multiline: multiline,
	}
	for i := 0; i+1 < len(choices); i += 2 {
		s.Choices = append(s.Choices, resourceChoice{
			Label: choices[i+1], FormAction: formAction, Name: choiceName, Value: choices[i],
		})
	}
	return s
}

// Resource kinds of the known-resource catalog.
const (
	resDataset   = "dataset"
	resQuestions = "questions"
	resPolicy    = "policy"
)

// maxKnownResources bounds each kind of the known-resource catalog.
const maxKnownResources = 12

// rememberResource records a resource the operator used, newest first. The
// catalog lives in this process only: exact paths stay in the evidence that
// recorded them, and nothing is written outside HACHIDORI_HOME.
func (d *Dashboard) rememberResource(kind, value string) {
	value = strings.TrimSpace(value)
	if value == "" {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.recent == nil {
		d.recent = map[string][]string{}
	}
	list := slices.DeleteFunc(slices.Clone(d.recent[kind]), func(v string) bool { return v == value })
	d.recent[kind] = append([]string{value}, list...)[:min(len(list)+1, maxKnownResources)]
}

// sampleResource is the install-local Inari sample for one semantic kind.
// It is a normal operator-visible path under HACHIDORI_HOME, not hidden state.
func (d *Dashboard) sampleResource(kind string) string {
	s := d.cfg.EvaluationSample
	switch kind {
	case resDataset:
		return s.Dataset
	case resQuestions:
		return s.Questions
	case resPolicy:
		return s.Policy
	}
	return ""
}

// resourceCatalog is the catalog of one kind: the install-local sample first,
// then resources used in this session and, for datasets, saved history.
func (d *Dashboard) resourceCatalog(kind string) []string {
	var values []string
	if sample := d.sampleResource(kind); sample != "" {
		values = append(values, sample)
	}
	d.mu.Lock()
	values = append(values, slices.Clone(d.recent[kind])...)
	d.mu.Unlock()
	if kind == resDataset && d.hist != nil {
		if entries, _, err := d.hist.List(); err == nil {
			for _, e := range entries {
				values = append(values, e.Dataset)
			}
		}
	}
	return values
}

// knownOptions lists the current value first, then the catalog, once each.
func knownOptions(catalog []string, current string) []resourceOption {
	current = strings.TrimSpace(current)
	values := catalog
	if current != "" {
		values = append([]string{current}, catalog...)
	}
	var out []resourceOption
	seen := map[string]bool{}
	for _, v := range values {
		if v == "" || seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, resourceOption{Value: v, Label: resourceLabel(v), Selected: v == current})
	}
	return out
}

// resourceLabel names a resource by its file names. The fixed install-local
// Inari bundle gets a semantic label; its exact path remains the option value.
func resourceLabel(value string) string {
	if !strings.Contains(value, "\n") {
		clean := filepath.ToSlash(filepath.Clean(value))
		switch {
		case strings.HasSuffix(clean, "/resources/evaluation/inari/dataset.jsonl"):
			return "Inari sample · dataset"
		case strings.HasSuffix(clean, "/resources/evaluation/inari/questions"):
			return "Inari sample · evaluation questions"
		case strings.HasSuffix(clean, "/resources/evaluation/inari/policy.json"):
			return "Inari sample · certification policy"
		}
	}
	var names []string
	for _, l := range lines(value) {
		names = append(names, filepath.Base(filepath.Clean(l)))
	}
	return strings.Join(names, ", ")
}

// resourceValue is the posted resource: an exact path typed under the
// disclosure wins over the chosen known resource.
func resourceValue(r *http.Request, name string) string {
	if v := r.PostFormValue(name + "_path"); strings.TrimSpace(v) != "" {
		return v
	}
	return r.PostFormValue(name)
}

// semanticResource is a resourceSelection over a known-resource catalog.
// none labels the empty choice and makes the resource optional.
func semanticResource(catalog []string, native bool, label, name, value, none string, multiline bool, formAction string, choices ...string) resourceSelection {
	s := resourceInput(native, label, name, value, "", none == "", multiline, formAction, "pick", choices...)
	s.Known, s.None = knownOptions(catalog, value), none
	return s
}
