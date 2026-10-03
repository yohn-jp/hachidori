package update

import (
	"os"
	"testing"

	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
	"github.com/yohn-jp/hachidori/test/windows-e2e/update/lab"
	"github.com/yohn-jp/hachidori/test/windows-e2e/update/stubapp"
)

// TestMain first lets this test binary play the two small stand-in roles the
// replacement scenarios need (the updated application the helper reopens, and
// the running application it waits for); both are selected by their first
// argument, which no go-test invocation uses. Otherwise it is the shard.
func TestMain(m *testing.M) {
	if code, ok := stubapp.Run(os.Args[1:]); ok {
		os.Exit(code)
	}
	os.Exit(e2e.Main(m, e2e.ShardUpdate))
}

// TestCandidateIdentity is the shard's shared precondition: it certifies the
// exact candidate bytes the workflow built once.
func TestCandidateIdentity(t *testing.T) { e2e.VerifyCandidateScenario(t) }

// The portable scenarios run against the real update service, the real
// Settings > Updates handler and the local deterministic release stub; no
// request leaves the machine.

// W18.2, W29.3
func TestUpdateCheckDownloadProgress(t *testing.T) {
	s := e2e.Begin(t, "update-check-download-progress")
	lab.CheckDownloadProgress(t, s)
}

// W18.1
func TestUpdateNoNetworkOnOpen(t *testing.T) {
	s := e2e.Begin(t, "update-no-network-on-open")
	lab.NoNetworkOnOpen(t, s)
}

// W29.2
func TestUpdateSingleCheckInFlight(t *testing.T) {
	s := e2e.Begin(t, "update-single-check-flight")
	lab.SingleCheckInFlight(t, s)
}

// W18.3
func TestUpdateCorruptionRejected(t *testing.T) {
	s := e2e.Begin(t, "update-corruption-rejected")
	lab.CorruptionRejected(t, s)
}

// W29.4
func TestUpdateNetworkFailureBounded(t *testing.T) {
	s := e2e.Begin(t, "update-network-failure-bounded")
	lab.NetworkFailureBounded(t, s)
}

// The replacement scenarios run the certified candidate's own helper against a
// disposable copy of it.

func candidate(s *e2e.Scenario) lab.Candidate {
	c := s.Candidate()
	s.Logf("certifying %s sha256=%s commit=%s", c.File, c.SHA256, c.SourceCommit)
	return lab.Candidate{Path: c.Path, SHA256: c.SHA256}
}

// W18.4
func TestUpdateReplaceRestart(t *testing.T) {
	s := e2e.Begin(t, "update-replace-restart")
	lab.ReplaceAndRestart(t, s, candidate(s))
}

// W18.5
func TestUpdateFileLockKeepsPrevious(t *testing.T) {
	s := e2e.Begin(t, "update-file-lock-keeps-previous")
	lab.FileLockKeepsPrevious(t, s, candidate(s))
}
