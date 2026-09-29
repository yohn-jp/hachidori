package desktop

// shouldHideOwnedConsole is deliberately platform-neutral so the ownership
// rule is covered by normal CI. A console is ours to hide only when this
// process is the sole process attached to it. An existing terminal has at
// least one other attached process and must never be hidden.
func shouldHideOwnedConsole(current uint32, attached []uint32) bool {
	return current != 0 && len(attached) == 1 && attached[0] == current
}
