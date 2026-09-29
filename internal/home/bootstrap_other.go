//go:build !windows

package home

// DefaultLocator reports ErrBootstrapUnsupported: non-Windows builds have no
// bootstrap locator and never create hidden bootstrap state.
func DefaultLocator() (Locator, error) {
	return Locator{}, ErrBootstrapUnsupported
}
