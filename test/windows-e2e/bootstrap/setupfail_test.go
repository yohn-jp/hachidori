package bootstrap

import (
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/test/windows-e2e/desktopkit"
	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

// setupWait bounds a setup that must fail fast: its first network request goes
// to an address nothing listens on.
const setupWait = 2 * time.Minute

// TestFailedSetupLeavesNoOwnedStaging runs the packaged executable's real
// `setup` command against a home with spaces, Unicode and a supported deep path
// while every network request is refused (a dead loopback proxy: a
// deterministic network failure, no download of any kind). It certifies the
// failure -> evidence -> bounded behavior -> consistent persisted state chain
// of the bootstrap path on Windows: the command fails within the bound and
// says why, it creates the home layout it owns, and it leaves no activation
// record, no runtime or model, no staging or partial download, and no
// bootstrap.json.
func TestFailedSetupLeavesNoOwnedStaging(t *testing.T) {
	s := e2e.Begin(t, "bootstrap-failed-setup-cleanup")
	r := desktopkit.NewRun(t, s)
	dead, err := desktopkit.FreeAddr()
	if err != nil {
		t.Fatal(err)
	}
	proxy := "http://" + dead
	r.L.Extra = []string{
		"HTTPS_PROXY=" + proxy, "HTTP_PROXY=" + proxy, "ALL_PROXY=" + proxy,
		"NO_PROXY=", "no_proxy=",
	}

	for _, shape := range desktopkit.Shapes() {
		func() {
			homeRoot, err := shape.Path(r.Path("setup", shape.Name))
			if err != nil {
				t.Fatal(err)
			}
			p := r.Cmd("setup-"+shape.Name, "setup", "--home", homeRoot, "--device", "cpu")
			code, err := p.WaitExit(setupWait)
			if err != nil {
				t.Fatalf("%s: setup did not finish within the bound: %v\noutput:\n%s", shape.Name, err, p.Output())
			}
			out := p.Output()
			if code == 0 {
				t.Fatalf("%s: setup succeeded with every network request refused; output:\n%s", shape.Name, out)
			}
			if !strings.Contains(out, "uv") {
				t.Errorf("%s: the failure does not say which step failed (want the private uv download); output:\n%s", shape.Name, out)
			}
			s.Logf("%s home (%d characters): setup failed with exit %d", shape.Name, len(homeRoot), code)

			// Consistent persisted state: the home layout setup owns exists, and
			// nothing that would claim an installation or leave work half done.
			tree := desktopkit.Tree(homeRoot, 0)
			for _, want := range []string{"state/", "runtime/", "cache/"} {
				if !desktopkit.Has(tree, want) {
					t.Errorf("%s: the home layout lacks %s: %v", shape.Name, want, desktopkit.Tree(homeRoot, 40))
				}
			}
			if m := desktopkit.InstallMarkers(homeRoot); len(m) != 0 {
				t.Errorf("%s: a failed setup left installation state %v", shape.Name, m)
			}
			if rs := desktopkit.Residue(homeRoot); len(rs) != 0 {
				t.Errorf("%s: a failed setup left owned staging or partial files: %v", shape.Name, rs)
			}
			for _, dir := range []string{"runtime/", "models/", "workers/"} {
				if got := desktopkit.Below(tree, dir); len(got) != 0 {
					t.Errorf("%s: a failed setup published or staged %v", shape.Name, got)
				}
			}
			// The CLI never remembers a home; only the desktop flow does.
			notExist(t, r.Profile.LocatorPath(), "bootstrap.json after a failed CLI setup")
		}()
	}
}
