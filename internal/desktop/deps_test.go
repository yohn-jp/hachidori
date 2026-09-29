package desktop

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// windowsUIDeps are the packages only the Windows desktop shell may link.
var windowsUIDeps = []string{
	"github.com/wailsapp/go-webview2",
	"golang.org/x/sys/windows",
}

func goListDeps(t *testing.T, goos, pkg string) []string {
	t.Helper()
	gobin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go tool not on PATH")
	}
	cmd := exec.Command(gobin, "list", "-deps", "-f", "{{.ImportPath}}", pkg)
	cmd.Dir = filepath.Join("..", "..")
	cmd.Env = append(os.Environ(), "GOOS="+goos, "GOARCH=amd64", "CGO_ENABLED=0")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("GOOS=%s go list -deps %s: %v\n%s", goos, pkg, err, out)
	}
	return strings.Fields(string(out))
}

func linksAny(deps []string, prefix string) bool {
	for _, d := range deps {
		if d == prefix || strings.HasPrefix(d, prefix+"/") {
			return true
		}
	}
	return false
}

// TestNonWindowsBuildsDoNotImportWindowsUI proves the Linux/NixOS (and other
// non-Windows) CLI never links the WebView2 binding or Windows system
// packages, while the Windows build of the same command does.
func TestNonWindowsBuildsDoNotImportWindowsUI(t *testing.T) {
	if testing.Short() {
		t.Skip("runs go list")
	}
	const cmd = "./cmd/hachidori"
	for _, goos := range []string{"linux", "darwin"} {
		deps := goListDeps(t, goos, cmd)
		if !linksAny(deps, "github.com/yohn-jp/hachidori/internal/desktop") {
			t.Fatalf("GOOS=%s: %s does not link internal/desktop; the check is vacuous", goos, cmd)
		}
		for _, p := range windowsUIDeps {
			if linksAny(deps, p) {
				t.Errorf("GOOS=%s: %s links Windows UI dependency %s", goos, cmd, p)
			}
		}
	}
	deps := goListDeps(t, "windows", cmd)
	for _, p := range windowsUIDeps {
		if !linksAny(deps, p) {
			t.Errorf("GOOS=windows: %s does not link %s; desktop shell not wired", cmd, p)
		}
	}
}
