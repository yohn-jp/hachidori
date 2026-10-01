// Package route is the deterministic routing boundary above the resident
// workers: it decides, per question, which resident answers first and
// whether the answer is handed off to the alternate resident.
//
// A routing Policy is declarative data supplied by the operator. It names
// resident models by their catalog IDs and measurement families by name; the
// package knows no question set, no provider and no model of its own. The
// condition of a handoff is a small closed set of deterministic checks on the
// first-path observation (an unconditional handoff, a set of choices, a
// confidence threshold, a top-two probability margin); it is not a rules
// engine and no check is universal. Thresholds belong to the (family, first
// model) pair of one rule, so each is calibrated against that pair's
// evidence (see Verify).
//
// The router never runs both residents for a question unless the policy hands
// that question off, never reloads or restarts a resident, and never answers
// from a resident the policy did not select. A failure that the policy
// cannot absorb is an explicit routing_failed error, not a weaker result.
package route

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// PolicySchema identifies the routing policy document.
const PolicySchema = "hachidori.routing-policy.v1"

// Policy is the declarative routing policy. Families maps a question id to a
// measurement family name (the same convention as `hachidori eval --family`).
// A question whose family has a rule follows that rule; every other question
// follows Default. A policy with no rule and no family routes everything by
// Default.
type Policy struct {
	Schema   string            `json:"schema"`
	ID       string            `json:"id"`
	Families map[string]string `json:"families,omitempty"`
	Rules    []Rule            `json:"rules,omitempty"`
	Default  Rule              `json:"default"`
}

// Rule routes the questions of one family. First is the resident that
// answers first; Handoff, when present, may send the result to another.
type Rule struct {
	Family  string   `json:"family,omitempty"`
	First   string   `json:"first"`
	Handoff *Handoff `json:"handoff,omitempty"`
}

// Handoff sends a question to resident To when When holds for the first-path
// result. A handoff is required unless Optional is set: when To cannot answer
// a required handoff the whole request fails explicitly; when To cannot answer
// an optional handoff the first-path result is kept and the reason code says
// so.
type Handoff struct {
	To       string `json:"to"`
	Optional bool   `json:"optional,omitempty"`
	When     When   `json:"when"`
}

// When is the handoff condition. Each present check is evaluated in the fixed
// order always, choice_in, confidence_below, margin_below and the first that
// holds is the reason. At least one check must be present.
type When struct {
	Always          bool     `json:"always,omitempty"`
	ChoiceIn        []string `json:"choice_in,omitempty"`
	ConfidenceBelow *float64 `json:"confidence_below,omitempty"`
	MarginBelow     *float64 `json:"margin_below,omitempty"`
}

func (w When) empty() bool {
	return !w.Always && len(w.ChoiceIn) == 0 && w.ConfidenceBelow == nil && w.MarginBelow == nil
}

// Parse decodes and validates one policy document. Unknown fields are
// rejected.
func Parse(data []byte) (Policy, error) {
	var p Policy
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&p); err != nil {
		return Policy{}, fmt.Errorf("routing policy: %w", err)
	}
	if dec.More() {
		return Policy{}, fmt.Errorf("routing policy: trailing data after the document")
	}
	if err := p.Validate(); err != nil {
		return Policy{}, fmt.Errorf("routing policy: %w", err)
	}
	return p, nil
}

// Validate checks the policy is closed and unambiguous. Whether the models it
// names are resident is checked when the router is built.
func (p Policy) Validate() error {
	if p.Schema != PolicySchema {
		return fmt.Errorf("schema must be %q, got %q", PolicySchema, p.Schema)
	}
	if !token(p.ID) {
		return fmt.Errorf("id must be a non-empty token (letters, digits, '-', '_', '.')")
	}
	if p.Default.Family != "" {
		return fmt.Errorf("default rule must not name a family")
	}
	if err := p.Default.validate("default"); err != nil {
		return err
	}
	seen := map[string]bool{}
	for i, r := range p.Rules {
		if !token(r.Family) {
			return fmt.Errorf("rules[%d]: family must be a non-empty token", i)
		}
		if seen[r.Family] {
			return fmt.Errorf("rules[%d]: duplicate rule for family %q", i, r.Family)
		}
		seen[r.Family] = true
		if err := r.validate(fmt.Sprintf("rules[%d]", i)); err != nil {
			return err
		}
	}
	for id, f := range p.Families {
		if id == "" || !seen[f] {
			return fmt.Errorf("families: question %q is assigned family %q, which has no rule", id, f)
		}
	}
	return nil
}

func (r Rule) validate(where string) error {
	if !token(r.First) {
		return fmt.Errorf("%s: first must be a catalog model ID", where)
	}
	h := r.Handoff
	if h == nil {
		return nil
	}
	if !token(h.To) || h.To == r.First {
		return fmt.Errorf("%s: handoff.to must be a catalog model ID other than first (%s)", where, r.First)
	}
	if h.When.empty() {
		return fmt.Errorf("%s: handoff.when needs at least one check (always, choice_in, confidence_below, margin_below)", where)
	}
	for _, c := range h.When.ChoiceIn {
		if c == "" {
			return fmt.Errorf("%s: handoff.when.choice_in must not contain an empty label", where)
		}
	}
	for name, v := range map[string]*float64{"confidence_below": h.When.ConfidenceBelow, "margin_below": h.When.MarginBelow} {
		if v != nil && !(*v > 0 && *v <= 1) {
			return fmt.Errorf("%s: handoff.when.%s must be in (0, 1]", where, name)
		}
	}
	return nil
}

func token(s string) bool {
	if s == "" || len(s) > 128 {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
			return false
		}
	}
	return true
}

// Digest is "sha256:" + hex SHA-256 of the canonical encoding: fixed field
// order, map keys sorted, no insignificant whitespace. File formatting and
// field order in the document do not participate.
func (p Policy) Digest() string {
	b, err := json.Marshal(p)
	if err != nil {
		panic(err) // only strings, bools, floats and string maps: cannot fail
	}
	s := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(s[:])
}

// Models lists every resident model the policy can use, once each, in order
// of first appearance (default rule first, then rules in order; a rule's
// first before its handoff target). It is the one deterministic order in
// which the router runs groups of questions.
func (p Policy) Models() []string {
	var out []string
	add := func(m string) {
		if !slices.Contains(out, m) {
			out = append(out, m)
		}
	}
	for _, r := range append([]Rule{p.Default}, p.Rules...) {
		add(r.First)
		if r.Handoff != nil {
			add(r.Handoff.To)
		}
	}
	return out
}

// rule is the rule that routes question id, and its family ("" for Default).
func (p Policy) rule(id string) (Rule, string) {
	if f, ok := p.Families[id]; ok {
		for _, r := range p.Rules {
			if r.Family == f {
				return r, f
			}
		}
	}
	return p.Default, ""
}

// fires evaluates the handoff condition against a first-path observation and
// returns the reason code of the first check that holds, "" when none does.
func (w When) fires(choice string, confidence float64, probabilities map[string]float64) string {
	switch {
	case w.Always:
		return reasonAlways
	case slices.Contains(w.ChoiceIn, choice):
		return reasonChoice
	case w.ConfidenceBelow != nil && confidence < *w.ConfidenceBelow:
		return reasonLowConfidence
	case w.MarginBelow != nil && margin(probabilities) < *w.MarginBelow:
		return reasonLowMargin
	}
	return ""
}

// margin is the probability gap between the two most probable choices. Fewer
// than two probabilities leave it undefined, which counts as no margin (0):
// the policy asked for a margin check the evidence cannot support, so it
// escalates rather than trusts.
func margin(p map[string]float64) float64 {
	var top, second float64
	n := 0
	for _, v := range p {
		n++
		switch {
		case v > top:
			top, second = v, top
		case v > second:
			second = v
		}
	}
	if n < 2 {
		return 0
	}
	return top - second
}

func joinNonEmpty(parts ...string) string {
	var out []string
	for _, p := range parts {
		if p != "" {
			out = append(out, p)
		}
	}
	return strings.Join(out, ": ")
}
