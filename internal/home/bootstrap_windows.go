//go:build windows

package home

import (
	"errors"
	"os"
	"path/filepath"
)

// DefaultLocator is %LOCALAPPDATA%\Hachidori\bootstrap.json, the per-user
// (non-roaming) application-data location. It does not touch the filesystem.
func DefaultLocator() (Locator, error) {
	base := os.Getenv("LOCALAPPDATA")
	if base == "" || !filepath.IsAbs(base) {
		return Locator{}, errors.New("LOCALAPPDATA is not set to an absolute path; cannot locate the bootstrap locator")
	}
	return Locator{Path: filepath.Join(base, "Hachidori", "bootstrap.json")}, nil
}
