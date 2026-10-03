//go:build !windows

package desktopkit

import "errors"

func webView2Installed() (string, error) { return "", errNotWindows }

func installWebView2() error { return errors.New("the WebView2 Runtime is a Windows component") }
