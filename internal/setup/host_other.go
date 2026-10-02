//go:build !unix && !windows

package setup

func freeDisk(string) (uint64, bool) { return 0, false }
