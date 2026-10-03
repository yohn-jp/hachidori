package lab

import (
	"os"
	"reflect"
	"testing"

	"github.com/yohn-jp/hachidori/test/windows-e2e/update/stubapp"
)

// TestMain lets this test binary play the stand-in application roles, as the
// shard's does, so the replacement scenarios can be driven from here too.
func TestMain(m *testing.M) {
	if code, ok := stubapp.Run(os.Args[1:]); ok {
		os.Exit(code)
	}
	os.Exit(m.Run())
}

// The portable scenarios run here on every platform against the local stub, so
// the scenario logic itself is verified wherever Go runs; the certification run
// executes the same functions through e2e.Begin on the Windows runner.

func TestCheckDownloadProgress(t *testing.T) { CheckDownloadProgress(t, Quiet(t)) }
func TestNoNetworkOnOpen(t *testing.T)       { NoNetworkOnOpen(t, Quiet(t)) }
func TestSingleCheckInFlight(t *testing.T)   { SingleCheckInFlight(t, Quiet(t)) }
func TestCorruptionRejected(t *testing.T)    { CorruptionRejected(t, Quiet(t)) }
func TestNetworkFailureBounded(t *testing.T) { NetworkFailureBounded(t, Quiet(t)) }

func TestExeIsDeterministicAndWindowsShaped(t *testing.T) {
	a, b := Exe("x", 1000), Exe("x", 1000)
	if string(a) != string(b) || string(a[:2]) != "MZ" || len(a) != 1000 {
		t.Fatal("Exe must be reproducible and start with the PE signature")
	}
	if string(Exe("y", 1000)) == string(a) {
		t.Fatal("different seeds must differ")
	}
	if len(Exe("z", 0)) != 2 {
		t.Fatal("Exe has a two byte minimum")
	}
}

func TestWithFlagReplacesOnlyTheNamedValue(t *testing.T) {
	in := []string{"apply-update", "--home", `C:\h`, "--pid", "10", "--target", `C:\a.exe`, "--restart-home", `C:\h`}
	out, err := withFlag(in, "--pid", "4242")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"apply-update", "--home", `C:\h`, "--pid", "4242", "--target", `C:\a.exe`, "--restart-home", `C:\h`}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("%v", out)
	}
	if in[4] != "10" {
		t.Fatal("withFlag modified its input")
	}
	if _, err := withFlag(in, "--missing", "1"); err == nil {
		t.Fatal("a missing flag must be an error")
	}
}
