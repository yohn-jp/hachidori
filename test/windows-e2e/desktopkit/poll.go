package desktopkit

import (
	"fmt"
	"time"
)

// PollInterval is how often Poll evaluates its condition.
const PollInterval = 100 * time.Millisecond

// Poll evaluates probe until it reports done, fails permanently or timeout
// passes. It is the only waiting primitive of the shards: the oracle is the
// observable condition (a file, an endpoint, a pid, a process exit), and the
// timeout is only the bound after which the scenario fails with the last
// observation. A probe error is treated as "not yet"; it is reported only when
// the deadline expires. A probe that returns fatal=true ends the wait at once.
func Poll(timeout time.Duration, what string, probe func() (done bool, err error)) error {
	return poll(timeout, PollInterval, what, probe)
}

func poll(timeout, interval time.Duration, what string, probe func() (bool, error)) error {
	deadline := time.Now().Add(timeout)
	var last error
	for {
		done, err := probe()
		if done {
			return nil
		}
		if err != nil {
			if f, ok := err.(Fatal); ok {
				return fmt.Errorf("%s: %w", what, f.Err)
			}
			last = err
		}
		if time.Now().After(deadline) {
			if last != nil {
				return fmt.Errorf("%s: not observed within %s: %w", what, timeout, last)
			}
			return fmt.Errorf("%s: not observed within %s", what, timeout)
		}
		time.Sleep(interval)
	}
}

// Fatal wraps an error that makes further polling pointless (for example the
// process under observation has exited).
type Fatal struct{ Err error }

func (f Fatal) Error() string { return f.Err.Error() }
func (f Fatal) Unwrap() error { return f.Err }
