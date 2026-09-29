//go:build windows

package desktop

import (
	"errors"

	"golang.org/x/sys/windows/registry"
)

// NativeStartup is the per-user Run-key startup entry (see RunKeyPath). It
// only ever opens HKEY_CURRENT_USER, so it never needs elevation.
func NativeStartup() Startup { return runKey{name: StartupValueName} }

type runKey struct{ name string }

func (r runKey) Enabled() (bool, error) {
	k, err := registry.OpenKey(registry.CURRENT_USER, RunKeyPath, registry.QUERY_VALUE)
	if errors.Is(err, registry.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	defer k.Close()
	_, _, err = k.GetStringValue(r.name)
	if errors.Is(err, registry.ErrNotExist) {
		return false, nil
	}
	return err == nil, err
}

func (r runKey) Enable(command string) error {
	k, _, err := registry.CreateKey(registry.CURRENT_USER, RunKeyPath, registry.QUERY_VALUE|registry.SET_VALUE)
	if err != nil {
		return err
	}
	defer k.Close()
	if cur, _, err := k.GetStringValue(r.name); err == nil && cur == command {
		return nil // already enabled with this exact command
	}
	return k.SetStringValue(r.name, command)
}

func (r runKey) Disable() error {
	k, err := registry.OpenKey(registry.CURRENT_USER, RunKeyPath, registry.SET_VALUE)
	if errors.Is(err, registry.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer k.Close()
	if err := k.DeleteValue(r.name); err != nil && !errors.Is(err, registry.ErrNotExist) {
		return err
	}
	return nil
}
