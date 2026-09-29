//go:build !windows

package home

func retryableBootstrapReadError(error) bool    { return false }
func retryableBootstrapReplaceError(error) bool { return false }
