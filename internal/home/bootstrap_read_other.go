//go:build !windows

package home

import "os"

func openBootstrapReadFile(path string) (*os.File, error) { return os.Open(path) }

func replaceBootstrapFile(replacement, replaced string) error {
	return os.Rename(replacement, replaced)
}

func retryableBootstrapReadError(error) bool    { return false }
func retryableBootstrapReplaceError(error) bool { return false }
