package update

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// This file is the replacement helper's whole logic. It is deliberately not a
// general tool: it opens no network connection, it takes no source path (the
// source is the one staged executable the ready record names, under
// HACHIDORI_HOME/state/updates), and the only file it replaces is the
// destination the ready record was prepared for. Both are checked again here,
// after the application has exited.

// Plan is everything the helper is given.
type Plan struct {
	Home        string // HACHIDORI_HOME that holds the ready record
	PID         int    // the application process to wait for
	Target      string // the executable to replace; must be the ready record's
	RestartHome string // when set, restart as "desktop --home <it>"
}

// ApplyEnv are the helper's OS effects. Production uses DefaultApplyEnv; tests
// substitute them to prove each failure path.
type ApplyEnv struct {
	// WaitExit returns once the process has exited, or an error after timeout.
	WaitExit func(pid int, timeout time.Duration) error
	// Restart starts exe with args, detached, and does not wait for it.
	Restart func(exe string, args []string) error
	Rename  func(oldpath, newpath string) error
	Sleep   func(time.Duration)
	Now     func() time.Time
	// ExitTimeout bounds the wait for the application to exit.
	ExitTimeout time.Duration
}

// DefaultApplyEnv is the production environment.
func DefaultApplyEnv() ApplyEnv {
	return ApplyEnv{WaitExit: waitExit, Restart: StartDetached, Rename: os.Rename, Sleep: time.Sleep, Now: time.Now, ExitTimeout: time.Minute}
}

const (
	renameTries = 20
	renameDelay = 250 * time.Millisecond
)

func suffixOld(target string) string { return target + ".old" }
func suffixNew(target string) string { return target + ".new" }

// Apply replaces Plan.Target with the staged, verified executable and restarts
// it. The sequence never leaves the destination without a usable executable:
//
//  1. validate the plan against the ready record and the staged file;
//  2. wait for the application to exit (the destination is untouched until then);
//  3. copy the staged file next to the destination (target.new) and verify the copy;
//  4. rename the destination to target.old, then target.new to the destination;
//     a failure of the second rename restores target.old;
//  5. restart the destination: the new executable after a replacement, the
//     previous one after a failed attempt.
//
// target.old is the recoverable previous executable. Apply reports its outcome
// in state/updates/result.json and returns it; a non-applied outcome is never
// a partial replacement.
func Apply(p Plan, env ApplyEnv) Result {
	res := Result{Target: p.Target, Time: env.Now()}
	finish := func(outcome, format string, a ...any) Result {
		res.Outcome, res.Message, res.Time = outcome, fmt.Sprintf(format, a...), env.Now()
		_ = writeResult(p.Home, res)
		return res
	}

	ready, ver, err := p.validate()
	if err != nil {
		return finish(OutcomeRefused, "update refused: %v", err)
	}
	res.Tag, res.SHA256 = ready.Tag, ready.SHA256

	if err := env.WaitExit(p.PID, env.ExitTimeout); err != nil {
		// The application is still running from the destination: touch nothing.
		return finish(OutcomeFailed, "the application did not exit, so nothing was replaced: %v", err)
	}
	// From here the application is gone: every outcome restarts the
	// destination, replaced or not.
	staged := StagedPath(p.Home, ver)
	if err := verifyFile(staged, ready); err != nil {
		r := finish(OutcomeFailed, "the staged update is not usable, so nothing was replaced: %v", err)
		return restartAfter(p, env, r, false)
	}
	if err := checkDestination(p.Target); err != nil {
		// Not a Windows executable: nothing is replaced and nothing is run.
		return finish(OutcomeFailed, "the destination is not a replaceable executable, so nothing was replaced or restarted: %v", err)
	}

	next, old := suffixNew(p.Target), suffixOld(p.Target)
	if err := copyVerified(staged, next, ready); err != nil {
		os.Remove(next)
		r := finish(OutcomeFailed, "the update could not be prepared beside the destination: %v", err)
		return restartAfter(p, env, r, false)
	}
	_ = os.Remove(old)
	if err := renameRetry(env, p.Target, old); err != nil {
		os.Remove(next)
		r := finish(OutcomeFailed, "the current executable could not be moved aside, so it was kept: %v", err)
		return restartAfter(p, env, r, false)
	}
	if err := renameRetry(env, next, p.Target); err != nil {
		msg := fmt.Sprintf("the update could not be put in place: %v", err)
		if rerr := renameRetry(env, old, p.Target); rerr != nil {
			msg += fmt.Sprintf("; the previous executable could not be restored either and is at %s: %v", old, rerr)
		} else {
			msg += "; the previous executable was restored"
		}
		os.Remove(next)
		r := finish(OutcomeFailed, "%s", msg)
		return restartAfter(p, env, r, false)
	}

	// Committed: the destination is the verified update.
	_ = os.Remove(readyPath(p.Home))
	_ = os.RemoveAll(filepath.Dir(staged))
	r := finish(OutcomeApplied, "replaced with %s; the previous executable is kept as %s", ready.Tag, filepath.Base(old))
	return restartAfter(p, env, r, true)
}

// restartAfter restarts the destination and settles the reported outcome: a
// restart failure after a committed replacement is restart_failed (the update
// is in place); after a failed replacement the report keeps its failure and
// says the previous executable could not be restarted.
func restartAfter(p Plan, env ApplyEnv, r Result, committed bool) Result {
	args := []string(nil)
	if p.RestartHome != "" {
		args = []string{"desktop", "--home", p.RestartHome}
	}
	if err := env.Restart(p.Target, args); err != nil {
		if committed {
			r.Outcome = OutcomeRestart
		}
		r.Message += fmt.Sprintf("; restarting %s failed: %v", p.Target, err)
		r.Time = env.Now()
		_ = writeResult(p.Home, r)
	}
	return r
}

// validate checks the plan against the ready record: the destination must be
// the one the update was prepared for, in the shape the helper replaces.
func (p Plan) validate() (Ready, Version, error) {
	switch {
	case p.PID <= 0 || p.PID == os.Getpid():
		return Ready{}, Version{}, errors.New("no application process to wait for")
	case !filepath.IsAbs(p.Home) || filepath.Clean(p.Home) != p.Home:
		return Ready{}, Version{}, errors.New("home is not a clean absolute path")
	case p.RestartHome != "" && (!filepath.IsAbs(p.RestartHome) || filepath.Clean(p.RestartHome) != p.RestartHome):
		return Ready{}, Version{}, errors.New("restart home is not a clean absolute path")
	case !filepath.IsAbs(p.Target) || filepath.Clean(p.Target) != p.Target:
		return Ready{}, Version{}, errors.New("destination is not a clean absolute path")
	}
	ready, ver, err := LoadReady(p.Home)
	if err != nil {
		return Ready{}, Version{}, fmt.Errorf("no verified update is ready: %w", err)
	}
	if !samePath(ready.Target, p.Target) {
		return Ready{}, Version{}, fmt.Errorf("the update was prepared for %s, not %s", ready.Target, p.Target)
	}
	if rel, err := filepath.Rel(Dir(p.Home), p.Target); err == nil && !strings.HasPrefix(rel, "..") {
		return Ready{}, Version{}, errors.New("destination is inside the update area")
	}
	return ready, ver, nil
}

// checkDestination accepts only an existing regular Windows executable file
// named *.exe.
func checkDestination(path string) error {
	if !strings.HasSuffix(strings.ToLower(path), ".exe") {
		return errors.New("not an .exe file")
	}
	_, err := checkExecutable(path)
	return err
}

// verifyFile checks the staged file is the recorded verified executable.
func verifyFile(path string, r Ready) error {
	fi, err := checkExecutable(path)
	if err != nil {
		return err
	}
	if fi.Size() != r.Size {
		return fmt.Errorf("size %d differs from the verified %d", fi.Size(), r.Size)
	}
	got, err := FileSHA256(path)
	if err != nil {
		return err
	}
	if got != r.SHA256 {
		return errors.New("SHA-256 differs from the verified digest")
	}
	return nil
}

// copyVerified copies src to a new file dst and checks the copy's digest.
func copyVerified(src, dst string, r Ready) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
	if err != nil {
		return err
	}
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(out, h), in)
	if err == nil {
		err = out.Sync()
	}
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return err
	}
	if n != r.Size || hex.EncodeToString(h.Sum(nil)) != r.SHA256 {
		return errors.New("the copy does not match the verified digest")
	}
	return nil
}

// renameRetry renames, retrying briefly: on Windows a virus scanner or indexer
// can hold a freshly exited executable for a moment. The bound is fixed.
func renameRetry(env ApplyEnv, from, to string) error {
	var err error
	for i := 0; i < renameTries; i++ {
		if err = env.Rename(from, to); err == nil {
			return nil
		}
		env.Sleep(renameDelay)
	}
	return err
}
