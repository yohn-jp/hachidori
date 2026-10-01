// Package ui holds the Hachidori desktop visual system: the single definition
// of its tokens (color, typography, spacing, geometry, focus, motion) and the
// primitives every surface shares (text roles, state vocabulary, controls,
// operator preferences). The workstation dashboard and the first-run page
// both inline it, so one semantic role renders the same everywhere.
// docs/desktop.md is the contract.
package ui

import (
	_ "embed"
	"html/template"
)

//go:embed system.css
var systemCSS string

// CSS is the visual system stylesheet, for inlining into a page's <style>
// (the pages' Content-Security-Policy admits inline styles only).
func CSS() template.CSS { return template.CSS(systemCSS) }
