//go:build !windows

package desktop

// ShowFatal is a no-op outside Windows; the CLI reports errors on stderr.
func ShowFatal(title, message string) {}
