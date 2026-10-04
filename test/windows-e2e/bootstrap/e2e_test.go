package bootstrap

import (
	"os"
	"testing"

	"github.com/yohn-jp/hachidori/test/windows-e2e/desktopkit"
	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

func TestMain(m *testing.M) {
	// The installed-home fixture uses this test binary as the worker stand-in.
	desktopkit.MaybeRunStub()
	os.Exit(e2e.Main(m, e2e.ShardBootstrap))
}
