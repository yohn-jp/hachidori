package desktopkit

import (
	"fmt"
	"strings"
	"sync"
)

// WebView2ClientID is the Evergreen WebView2 Runtime's Edge Update client ID;
// its "pv" value under Clients is the installed version.
const WebView2ClientID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"

// webView2Present reports whether an Edge Update "pv" value names an installed
// runtime (Edge Update writes 0.0.0.0 for an uninstalled client).
func webView2Present(pv string) bool {
	pv = strings.TrimSpace(pv)
	return pv != "" && pv != "0.0.0.0"
}

var (
	webView2Once    sync.Once
	webView2Version string
	webView2Err     error
)

// EnsureWebView2 makes sure the Microsoft Edge WebView2 Runtime, which the
// packaged executable requires and never installs itself, is present on this
// runner, and returns its version. When it is absent it installs Microsoft's
// evergreen bootstrapper silently (a prerequisite of the runner image, not
// product behavior: the candidate itself never downloads or installs it) and
// checks again; a runner that still has none fails the shard rather than
// letting every scenario fail with a less direct error. It runs once per test
// binary.
func EnsureWebView2() (string, error) {
	webView2Once.Do(func() {
		v, err := webView2Installed()
		if err == nil && webView2Present(v) {
			webView2Version = v
			return
		}
		if ierr := installWebView2(); ierr != nil {
			webView2Err = fmt.Errorf("the WebView2 Runtime is not installed on this runner and installing it failed: %w", ierr)
			return
		}
		v, err = webView2Installed()
		if err != nil || !webView2Present(v) {
			webView2Err = fmt.Errorf("the WebView2 Runtime is still not installed after running its bootstrapper (version %q, %v)", v, err)
			return
		}
		webView2Version = v
	})
	return webView2Version, webView2Err
}
