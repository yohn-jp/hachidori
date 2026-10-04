package bootstrap

import (
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/firstrun"
	"github.com/yohn-jp/hachidori/test/windows-e2e/desktopkit"
	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

// TestCandidateIdentity is the shard's shared prerequisite: it certifies the
// exact candidate bytes before any scenario can use the executable.
func TestCandidateIdentity(t *testing.T) { e2e.VerifyCandidateScenario(t) }

// TestCleanProfileStartsFirstRun proves W01.2 against the packaged executable:
// a clean profile (no bootstrap.json, no HACHIDORI_HOME) starts in the
// first-run state and writes nothing before Install.
func TestCleanProfileStartsFirstRun(t *testing.T) {
	s := e2e.Begin(t, "bootstrap-clean-profile")
	r := desktopkit.NewRun(t, s)
	notExist(t, r.Profile.LocatorPath(), "bootstrap.json before launch")

	i := r.Desktop("clean-profile")
	v := wantMode(t, i, firstrun.ModeFirstRun)
	if v.State != app.Unconfigured || v.Stage != firstrun.StageSelect || v.Selection != nil ||
		v.Operation != nil || v.Dashboard || v.StoredHome != "" || v.Notice != "" || v.Failure != nil {
		t.Errorf("first-run view is not the untouched selection screen: %+v", v)
	}
	s.Logf("first run: state=%s stage=%s mode=%s", v.State, v.Stage, v.Mode)

	// Nothing but the desktop's own preferences may exist: no locator, and no
	// trace of an installation, anywhere in the disposable profile.
	notExist(t, r.Profile.LocatorPath(), "bootstrap.json while the first-run screen is shown")
	noInstall(t, r.Root)
	tree := desktopkit.Tree(r.Profile.LocatorDir(), 20)
	s.Logf("profile Hachidori folder: %s", strings.Join(tree, ", "))
	for _, e := range tree {
		if strings.HasSuffix(e, "bootstrap.json") {
			t.Errorf("the profile Hachidori folder holds %s", e)
		}
	}

	r.Stop(i)
	notExist(t, r.Profile.LocatorPath(), "bootstrap.json after the first-run process ended")
}
