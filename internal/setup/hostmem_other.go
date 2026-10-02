//go:build !linux && !windows

package setup

// hostMemory is unknown where the platform's RAM report is not read: nothing
// is estimated.
func hostMemory() Memory { return Memory{} }
