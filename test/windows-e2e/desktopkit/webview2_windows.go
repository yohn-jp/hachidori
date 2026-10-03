//go:build windows

package desktopkit

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"golang.org/x/sys/windows/registry"
)

// webView2Installed reads the installed runtime version from the machine or
// the user hive (either registry view).
// webView2Bootstrapper is Microsoft's evergreen WebView2 Runtime bootstrapper.
const webView2Bootstrapper = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"

func webView2Installed() (string, error) {
	type loc struct {
		root registry.Key
		path string
	}
	locs := []loc{
		{registry.LOCAL_MACHINE, `SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\` + WebView2ClientID},
		{registry.LOCAL_MACHINE, `SOFTWARE\Microsoft\EdgeUpdate\Clients\` + WebView2ClientID},
		{registry.CURRENT_USER, `Software\Microsoft\EdgeUpdate\Clients\` + WebView2ClientID},
	}
	var last error
	for _, l := range locs {
		k, err := registry.OpenKey(l.root, l.path, registry.QUERY_VALUE)
		if err != nil {
			last = err
			continue
		}
		pv, _, err := k.GetStringValue("pv")
		_ = k.Close()
		if err == nil && webView2Present(pv) {
			return pv, nil
		}
		last = err
	}
	return "", last
}

func installWebView2() error {
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Minute)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, webView2Bootstrapper, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("download the bootstrapper: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download the bootstrapper: HTTP %d", resp.StatusCode)
	}
	dir, err := os.MkdirTemp("", "webview2-")
	if err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(dir) }()
	exe := filepath.Join(dir, "MicrosoftEdgeWebview2Setup.exe")
	f, err := os.Create(exe)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, io.LimitReader(resp.Body, 64<<20)); err != nil {
		_ = f.Close()
		return fmt.Errorf("download the bootstrapper: %w", err)
	}
	if err := f.Close(); err != nil {
		return err
	}
	if out, err := exec.CommandContext(ctx, exe, "/silent", "/install").CombinedOutput(); err != nil {
		return fmt.Errorf("run the bootstrapper: %w: %s", err, out)
	}
	return nil
}
