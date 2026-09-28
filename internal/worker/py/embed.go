// Package py embeds the private Python worker so that setup can materialize
// it into the immutable runtime directory.
package py

import _ "embed"

// Script is the worker source, materialized as runtime/<version>/worker/hachidori_worker.py.
//
//go:embed hachidori_worker.py
var Script []byte
