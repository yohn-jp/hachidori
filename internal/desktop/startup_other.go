//go:build !windows

package desktop

// NativeStartup reports ErrUnsupported: only Windows has a start-at-sign-in
// mechanism in Hachidori.
func NativeStartup() Startup { return unsupportedStartup{} }

type unsupportedStartup struct{}

func (unsupportedStartup) Enabled() (bool, error) { return false, ErrUnsupported }
func (unsupportedStartup) Enable(string) error    { return ErrUnsupported }
func (unsupportedStartup) Disable() error         { return ErrUnsupported }
