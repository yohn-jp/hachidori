package lab

import (
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/test/windows-e2e/update/disposable"
)

func TestTailBufferKeepsTheNewestBytesOnly(t *testing.T) {
	var b tailBuffer
	b.Write([]byte(strings.Repeat("a", stderrTail)))
	b.Write([]byte("tail"))
	got := b.String()
	if len(got) != stderrTail || !strings.HasSuffix(got, "tail") || strings.Count(got, "a") != stderrTail-3 /* "tail" holds one a */ {
		t.Fatalf("len %d", len(got))
	}
}

func TestTrimKeepsTheEnd(t *testing.T) {
	if got := trim("  "+strings.Repeat("x", 10)+"END  ", 5); got != "...xxEND" {
		t.Fatalf("%q", got)
	}
	if trim("short", 50) != "short" {
		t.Fatal("short text must be unchanged")
	}
}

func TestFreeAddrIsLoopback(t *testing.T) {
	a, err := freeAddr()
	if err != nil || !strings.HasPrefix(a, "127.0.0.1:") {
		t.Fatalf("%q %v", a, err)
	}
}

// localCandidate lets a developer run the scenarios against any hachidori
// executable of the host OS (for example one built with `go build ./cmd/hachidori`):
//
//	HACHIDORI_E2E_LAB_EXE=/path/to/hachidori go test ./test/windows-e2e/diagnostics/lab
//
// Certification runs the same functions against the workflow's Windows candidate.
func localCandidate(t *testing.T) Candidate {
	t.Helper()
	p := os.Getenv("HACHIDORI_E2E_LAB_EXE")
	if p == "" {
		t.Skip("set HACHIDORI_E2E_LAB_EXE to run against a local executable")
	}
	sum, err := disposable.FileSHA256(p)
	if err != nil {
		t.Fatal(err)
	}
	fi, _ := os.Stat(p)
	return Candidate{Path: p, File: "hachidori-local.exe", SHA256: sum, Size: fi.Size(), GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, GoVersion: runtime.Version()}
}

type quiet struct{ t *testing.T }

func (q quiet) Logf(f string, a ...any)     { q.t.Logf(f, a...) }
func (quiet) Attach(string, []byte, string) {}

func TestLocalBundleSchemaIdentity(t *testing.T) {
	BundleSchemaIdentity(t, quiet{t}, localCandidate(t))
}
func TestLocalBundleSecrecy(t *testing.T) { BundleSecrecy(t, quiet{t}, localCandidate(t)) }
