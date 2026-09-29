//go:build !windows

package firstrun

// DefaultFreeSpace is unknown outside Windows: nothing is estimated.
func DefaultFreeSpace(string) (uint64, bool) { return 0, false }
