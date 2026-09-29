package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/yohn-jp/hachidori/internal/desktop"
)

type fakeDesktop struct {
	held      bool
	activated int
	opened    int
	reported  []string
	activeErr error
}

func (f *fakeDesktop) RuntimeVersion() (string, error) { return "129.0.0.0", nil }
func (f *fakeDesktop) AcquireInstance() (func(), error) {
	if f.held {
		return nil, desktop.ErrAlreadyRunning
	}
	f.held = true
	return func() { f.held = false }, nil
}
func (f *fakeDesktop) Open(context.Context, desktop.Window) error { f.opened++; return nil }
func (f *fakeDesktop) Activate() error                            { f.activated++; return f.activeErr }
func (f *fakeDesktop) ReportError(title, msg string)              { f.reported = append(f.reported, msg) }

type noStartup struct{}

func (noStartup) Enabled() (bool, error) { return false, nil }
func (noStartup) Enable(string) error    { return nil }
func (noStartup) Disable() error         { return nil }

// 4: a duplicate launch activates the running instance and starts nothing:
// no worker log, no runtime, no window of its own.
func TestDuplicateLaunchActivatesExistingInstance(t *testing.T) {
	t.Setenv("HACHIDORI_HOME", "")
	homeDir := t.TempDir()
	f := &fakeDesktop{held: true} // another shell of this user is running
	err := runDesktop(f, noStartup{}, filepath.Join(t.TempDir(), "desktop.json"), []string{"--home", homeDir})
	if err != nil {
		t.Fatalf("duplicate launch returned %v; activating the running instance is a success", err)
	}
	if f.activated != 1 || f.opened != 0 || len(f.reported) != 0 {
		t.Fatalf("activated=%d opened=%d reported=%v", f.activated, f.opened, f.reported)
	}
	if _, err := os.Stat(filepath.Join(homeDir, "logs")); !os.IsNotExist(err) {
		t.Fatalf("a duplicate launch touched the home (a second runtime owner started): %v", err)
	}
}

func TestDuplicateLaunchWhoseActivationFailsStaysAnError(t *testing.T) {
	t.Setenv("HACHIDORI_HOME", "")
	f := &fakeDesktop{held: true, activeErr: errors.New("no window")}
	err := runDesktop(f, noStartup{}, "", []string{"--home", t.TempDir()})
	if !errors.Is(err, desktop.ErrAlreadyRunning) || f.opened != 0 {
		t.Fatalf("err = %v opened=%d", err, f.opened)
	}
}

// 9: a background start that cannot bring the runtime up fails visibly (a
// message the user can see, exit status 1) and is not retried by the app.
func TestBackgroundStartFailureIsVisibleAndNotRetried(t *testing.T) {
	t.Setenv("HACHIDORI_HOME", "")
	homeDir := t.TempDir() // no active runtime
	if err := os.MkdirAll(filepath.Join(homeDir, "logs"), 0o755); err != nil {
		t.Fatal(err)
	}
	f := &fakeDesktop{}
	err := runDesktop(f, noStartup{}, "", []string{"--background", "--home", homeDir})
	if err == nil {
		t.Fatal("start without an installed runtime succeeded")
	}
	if len(f.reported) != 1 || f.reported[0] == "" {
		t.Fatalf("failure not reported to the user: %v", f.reported)
	}
	if f.opened != 0 {
		t.Fatal("a window opened although the runtime was never bound")
	}
	if f.held {
		t.Fatal("the instance guard was not released after the failure")
	}
}

// An interactive (non-background) failure is on the console only; no dialog.
func TestInteractiveStartFailureDoesNotPopUpDialog(t *testing.T) {
	t.Setenv("HACHIDORI_HOME", "")
	homeDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(homeDir, "logs"), 0o755); err != nil {
		t.Fatal(err)
	}
	f := &fakeDesktop{}
	if err := runDesktop(f, noStartup{}, "", []string{"--home", homeDir}); err == nil {
		t.Fatal("expected failure")
	}
	if len(f.reported) != 0 {
		t.Fatalf("interactive failure raised a dialog: %v", f.reported)
	}
}

func TestDesktopRequiresAHome(t *testing.T) {
	t.Setenv("HACHIDORI_HOME", "")
	f := &fakeDesktop{}
	if err := runDesktop(f, noStartup{}, "", nil); err == nil || f.opened != 0 {
		t.Fatalf("err = %v opened=%d", err, f.opened)
	}
}
