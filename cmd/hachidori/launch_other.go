//go:build !windows

package main

// noArgLaunch is nil outside Windows: with no arguments the CLI prints its
// usage exactly as before.
var noArgLaunch func() error
