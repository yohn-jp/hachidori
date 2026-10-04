package home

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// StateRef is the content address of the exact UTF-8 State bytes. No normalization
// is performed: the worker receives exactly the bytes that were registered.
func StateRef(state string) string {
	sum := sha256.Sum256([]byte(state))
	return "sha256:" + hex.EncodeToString(sum[:])
}

// RegisterState stores an immutable State beneath HACHIDORI_HOME. Concurrent
// registrations never replace an existing object, even if its contents differ.
func (h Home) RegisterState(state string) (string, error) {
	if strings.TrimSpace(state) == "" || len(state) > 64<<10 {
		return "", errors.New("state must be non-empty and at most 65536 bytes")
	}
	ref := StateRef(state)
	dir := h.Path("state", "registered")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	path := filepath.Join(dir, strings.TrimPrefix(ref, "sha256:"))
	f, err := os.CreateTemp(dir, ".state-*")
	if err != nil {
		return "", err
	}
	defer os.Remove(f.Name())
	if err = f.Chmod(0o600); err != nil {
		f.Close()
		return "", err
	}
	if _, err = f.WriteString(state); err != nil {
		f.Close()
		return "", err
	}
	if err = f.Sync(); err != nil {
		f.Close()
		return "", err
	}
	if err = f.Close(); err != nil {
		return "", err
	}
	if err = os.Link(f.Name(), path); errors.Is(err, os.ErrExist) {
		_, err = h.ResolveState(ref)
		return ref, err
	}
	if err != nil {
		return "", err
	}
	return ref, nil
}

// ResolveState verifies both the reference syntax and stored content before use.
// Missing and corrupted objects fail closed; neither can select another State.
func (h Home) ResolveState(ref string) (string, error) {
	digest := strings.TrimPrefix(ref, "sha256:")
	if len(ref) != 71 || !strings.HasPrefix(ref, "sha256:") || len(digest) != 64 {
		return "", fmt.Errorf("invalid state_ref %q", ref)
	}
	raw, err := hex.DecodeString(digest)
	if err != nil || len(raw) != sha256.Size || hex.EncodeToString(raw) != digest {
		return "", fmt.Errorf("invalid state_ref %q", ref)
	}
	// Root confines the canonical digest lookup to the registered-State store.
	// The public reference never becomes an unrestricted filesystem path.
	root, err := os.OpenRoot(h.Path("state", "registered"))
	if err != nil {
		return "", fmt.Errorf("unavailable state_ref %q: %w", ref, err)
	}
	defer root.Close()
	f, err := root.Open(digest)
	if err != nil {
		return "", fmt.Errorf("unavailable state_ref %q: %w", ref, err)
	}
	data, err := io.ReadAll(io.LimitReader(f, (64<<10)+1))
	closeErr := f.Close()
	if err != nil {
		return "", fmt.Errorf("unavailable state_ref %q: %w", ref, err)
	}
	if closeErr != nil {
		return "", fmt.Errorf("unavailable state_ref %q: %w", ref, closeErr)
	}
	if StateRef(string(data)) != ref || len(data) > 64<<10 || strings.TrimSpace(string(data)) == "" {
		return "", fmt.Errorf("corrupt state_ref %q", ref)
	}
	return string(data), nil
}
