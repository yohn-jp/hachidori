// Package py embeds the private optimizer entrypoint that the optimizer
// runtime carries. It is a separate package so that internal/setup can embed
// and pin the script without importing the optimizer builder.
package py

import _ "embed"

// Script is hachidori_optimizer.py.
//
//go:embed hachidori_optimizer.py
var Script []byte
