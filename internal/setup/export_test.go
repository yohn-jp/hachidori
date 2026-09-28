package setup

import (
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
)

// MaterializeFake runs the real setup reconciliation against the fake uv and
// artifact server, for tests outside this package.
func MaterializeFake(t *testing.T, device string) home.Home {
	f := newFixture(t)
	f.mustRun(device)
	return f.H
}
