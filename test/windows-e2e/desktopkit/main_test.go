package desktopkit

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	MaybeRunStub()
	if mode := os.Getenv("DESKTOPKIT_FAKE_EXE"); mode != "" {
		fakeExe(mode)
		return
	}
	os.Exit(m.Run())
}

// fakeExe stands in for the candidate in the launcher tests: it logs the lines
// the desktop logs, reports what it was given, and then stays up or exits.
func fakeExe(mode string) {
	fmt.Fprintf(os.Stderr, "args=%s\n", strings.Join(os.Args[1:], " "))
	fmt.Fprintf(os.Stderr, "LOCALAPPDATA=%s\n", os.Getenv("LOCALAPPDATA"))
	fmt.Fprintf(os.Stderr, "HACHIDORI_HOME=%q\n", os.Getenv("HACHIDORI_HOME"))
	if mode == "die" {
		fmt.Fprintln(os.Stderr, "hachidori: the fake desktop failed")
		os.Exit(1)
	}
	fmt.Fprintln(os.Stderr, "hachidori: desktop start: recovery_invalid")
	fmt.Fprintln(os.Stderr, "hachidori: startup first navigation completed +5ms since entry")
	time.Sleep(time.Minute)
}
