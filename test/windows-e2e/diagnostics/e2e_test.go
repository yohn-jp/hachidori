package diagnostics

import (
	"os"
	"testing"

	"github.com/yohn-jp/hachidori/test/windows-e2e/diagnostics/lab"
	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

func TestMain(m *testing.M) { os.Exit(e2e.Main(m, e2e.ShardDiagnostics)) }

// TestCandidateIdentity is the shard's shared precondition: it certifies the
// exact candidate bytes the workflow built once.
func TestCandidateIdentity(t *testing.T) { e2e.VerifyCandidateScenario(t) }

func candidate(s *e2e.Scenario) lab.Candidate {
	c := s.Candidate()
	s.Logf("certifying %s sha256=%s commit=%s", c.File, c.SHA256, c.SourceCommit)
	return lab.Candidate{Path: c.Path, File: c.File, SHA256: c.SHA256, Size: c.Size, GOOS: c.GOOS, GOARCH: c.GOARCH, GoVersion: c.GoVersion}
}

// W11.1, W11.2: Export bundle against the real candidate on a disposable home.
func TestDiagnosticsBundleSchemaIdentity(t *testing.T) {
	s := e2e.Begin(t, "diagnostics-bundle-schema-identity")
	lab.BundleSchemaIdentity(t, s, candidate(s))
}

// W11.4 (and the bound of W11.1): canary secrets never reach the bundle.
func TestDiagnosticsBundleSecrecy(t *testing.T) {
	s := e2e.Begin(t, "diagnostics-bundle-secrecy")
	lab.BundleSecrecy(t, s, candidate(s))
}
