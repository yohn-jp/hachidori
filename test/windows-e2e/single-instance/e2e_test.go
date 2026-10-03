package singleinstance

import (
	"os"
	"testing"

	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

func TestMain(m *testing.M) { os.Exit(e2e.Main(m, e2e.ShardSingleInstance)) }

// TestCandidateIdentity is the shard's shared precondition: it certifies the
// exact candidate bytes the workflow built once.
func TestCandidateIdentity(t *testing.T) { e2e.VerifyCandidateScenario(t) }
