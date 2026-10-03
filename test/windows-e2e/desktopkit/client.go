package desktopkit

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/firstrun"
	"github.com/yohn-jp/hachidori/internal/server"
)

var httpClient = &http.Client{
	Timeout: 10 * time.Second,
	// A form post answers with a redirect; the post itself is the observation.
	CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
}

func get(addr, path string) ([]byte, int, error) {
	resp, err := httpClient.Get("http://" + addr + path)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	return b, resp.StatusCode, err
}

// WizardState reads the first-run/recovery state the desktop serves on its
// dashboard origin dashAddr (GET /wizard/state). It is the same projection the
// window renders, so it observes the screen's content without a window.
func WizardState(dashAddr string) (firstrun.View, error) {
	b, code, err := get(dashAddr, "/wizard/state")
	if err != nil {
		return firstrun.View{}, err
	}
	if code != http.StatusOK {
		return firstrun.View{}, fmt.Errorf("GET /wizard/state: HTTP %d", code)
	}
	var v firstrun.View
	if err := json.Unmarshal(b, &v); err != nil {
		return firstrun.View{}, fmt.Errorf("decode wizard state: %w", err)
	}
	return v, nil
}

var tokenPattern = regexp.MustCompile(`name="token" value="([0-9a-f]+)"`)

// ExtractToken finds the per-process form token in a dashboard page.
func ExtractToken(page string) (string, bool) {
	m := tokenPattern.FindStringSubmatch(page)
	if m == nil {
		return "", false
	}
	return m[1], true
}

// DashboardToken reads the dashboard home page of dashAddr and returns its
// form token.
func DashboardToken(dashAddr string) (string, error) {
	b, code, err := get(dashAddr, "/")
	if err != nil {
		return "", err
	}
	if code != http.StatusOK {
		return "", fmt.Errorf("GET /: HTTP %d", code)
	}
	tok, ok := ExtractToken(string(b))
	if !ok {
		return "", fmt.Errorf("the dashboard page carries no form token")
	}
	return tok, nil
}

// PostRuntime posts the dashboard's runtime action (start, stop, restart) the
// way its own button does: a same-origin form post carrying the token.
func PostRuntime(dashAddr, op, token string) error {
	form := url.Values{"token": {token}}
	req, err := http.NewRequest(http.MethodPost, "http://"+dashAddr+"/runtime/"+op, strings.NewReader(form.Encode()))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Origin", "http://"+dashAddr)
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode >= 400 {
		return fmt.Errorf("POST /runtime/%s: HTTP %d", op, resp.StatusCode)
	}
	return nil
}

// Status reads /v1/status from the inference API at apiAddr.
func Status(apiAddr string) (server.Status, error) {
	b, code, err := get(apiAddr, "/v1/status")
	if err != nil {
		return server.Status{}, err
	}
	if code != http.StatusOK {
		return server.Status{}, fmt.Errorf("GET /v1/status: HTTP %d", code)
	}
	var st server.Status
	if err := json.Unmarshal(b, &st); err != nil {
		return server.Status{}, fmt.Errorf("decode status: %w", err)
	}
	return st, nil
}

// Ready reports whether GET /health on apiAddr answers READY.
func Ready(apiAddr string) (bool, error) {
	_, code, err := get(apiAddr, "/health")
	if err != nil {
		return false, err
	}
	return code == http.StatusOK, nil
}
