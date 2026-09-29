//go:build !windows

package home

func retryableBootstrapReadError(error) bool { return false }
