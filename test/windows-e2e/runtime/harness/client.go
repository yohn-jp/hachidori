package harness

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// Client observes and drives a running hachidori.exe over its two loopback
// endpoints, exactly as a caller and the window do: the HTTP API (health,
// status, typed decisions) and the dashboard (the Start, Stop and Restart
// forms with their token, the Diagnostics page, the first-run state).
type Client struct {
	APIAddr  string
	DashAddr string
	http     *http.Client
}

// NewClient returns a client for the two loopback host:port addresses. It
// never follows redirects: the dashboard answers an action with 303.
func NewClient(apiAddr, dashAddr string) *Client {
	return &Client{APIAddr: apiAddr, DashAddr: dashAddr, http: &http.Client{
		Timeout:       60 * time.Second,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}}
}

func (c *Client) get(base, path string) (int, []byte, error) {
	resp, err := c.http.Get("http://" + base + path)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	return resp.StatusCode, b, err
}

// Health is GET /health: the status code and the body.
func (c *Client) Health() (int, api.Health, error) {
	code, b, err := c.get(c.APIAddr, "/health")
	if err != nil {
		return 0, api.Health{}, err
	}
	var h api.Health
	if err := json.Unmarshal(b, &h); err != nil {
		return code, h, fmt.Errorf("/health: %w", err)
	}
	return code, h, nil
}

// Status is GET /v1/status. A non-200 answer (the API is bound before the
// runtime is) is an error.
func (c *Client) Status() (server.Status, error) {
	code, b, err := c.get(c.APIAddr, "/v1/status")
	if err != nil {
		return server.Status{}, err
	}
	if code != http.StatusOK {
		return server.Status{}, fmt.Errorf("/v1/status answered %d", code)
	}
	var st server.Status
	if err := json.Unmarshal(b, &st); err != nil {
		return st, fmt.Errorf("/v1/status: %w", err)
	}
	return st, nil
}

// DecideResult is the answer to POST /v1/decide.
type DecideResult struct {
	Code     int
	Response api.DecideResponse
	Error    api.ErrorBody
}

// Decide sends one typed request over the real HTTP API.
func (c *Client) Decide(req api.DecideRequest) (DecideResult, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return DecideResult{}, err
	}
	resp, err := c.http.Post("http://"+c.APIAddr+"/v1/decide", "application/json", bytes.NewReader(body))
	if err != nil {
		return DecideResult{}, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return DecideResult{}, err
	}
	r := DecideResult{Code: resp.StatusCode}
	if resp.StatusCode == http.StatusOK {
		err = json.Unmarshal(b, &r.Response)
	} else {
		err = json.Unmarshal(b, &r.Error)
	}
	if err != nil {
		return r, fmt.Errorf("/v1/decide answered %d with unreadable body: %w", resp.StatusCode, err)
	}
	return r, nil
}

var tokenPattern = regexp.MustCompile(`name="token" value="([^"]+)"`)

// Token reads the dashboard's form token the way the page's own forms carry
// it.
func (c *Client) Token() (string, error) {
	code, b, err := c.get(c.DashAddr, "/")
	if err != nil {
		return "", err
	}
	if code != http.StatusOK {
		return "", fmt.Errorf("dashboard / answered %d", code)
	}
	m := tokenPattern.FindSubmatch(b)
	if m == nil {
		return "", fmt.Errorf("dashboard / carries no form token")
	}
	return string(m[1]), nil
}

// RuntimeAction posts the dashboard's own lifecycle form: op is "start",
// "stop" or "restart". The dashboard answers a performed action with 303.
func (c *Client) RuntimeAction(op string) error {
	tok, err := c.Token()
	if err != nil {
		return err
	}
	form := url.Values{"token": {tok}}
	resp, err := c.http.PostForm("http://"+c.DashAddr+"/runtime/"+op, form)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode != http.StatusSeeOther {
		return fmt.Errorf("POST /runtime/%s answered %d", op, resp.StatusCode)
	}
	return nil
}

// Diagnostics is the dashboard's Diagnostics page.
func (c *Client) Diagnostics() (string, error) {
	code, b, err := c.get(c.DashAddr, "/diagnostics")
	if err != nil {
		return "", err
	}
	if code != http.StatusOK {
		return "", fmt.Errorf("dashboard /diagnostics answered %d", code)
	}
	return string(b), nil
}

// WizardFailure is the failure the application controller keeps.
type WizardFailure struct {
	Source  string `json:"source"`
	Class   string `json:"class"`
	Phase   string `json:"phase"`
	Message string `json:"message"`
}

// WizardView is the part of GET /wizard/state the scenarios read: how the
// desktop was entered and the application controller's own state.
type WizardView struct {
	Mode      string         `json:"mode"`
	State     string         `json:"state"`
	Stage     string         `json:"stage"`
	Dashboard bool           `json:"dashboard"`
	Failure   *WizardFailure `json:"failure,omitempty"`
}

// Wizard reads the first-run/recovery state, which also carries the
// application controller's state ("ready", "installed" for stopped, "failed"
// for Needs attention).
func (c *Client) Wizard() (WizardView, error) {
	code, b, err := c.get(c.DashAddr, "/wizard/state")
	if err != nil {
		return WizardView{}, err
	}
	if code != http.StatusOK {
		return WizardView{}, fmt.Errorf("/wizard/state answered %d", code)
	}
	var v WizardView
	if err := json.Unmarshal(b, &v); err != nil {
		return v, fmt.Errorf("/wizard/state: %w", err)
	}
	return v, nil
}

var recoveryPattern = regexp.MustCompile(`data-recovery="([a-z_]+)"`)

// RecoveryNotice returns the recovery notice of a Diagnostics page:
// "recovering", "gave_up" or "" when there is none.
func RecoveryNotice(page string) string {
	m := recoveryPattern.FindStringSubmatch(page)
	if m == nil {
		return ""
	}
	return m[1]
}

// Listening reports whether something accepts connections on addr.
func Listening(addr string) bool {
	c, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		return false
	}
	_ = c.Close()
	return true
}

// FreeAddrs returns n distinct free loopback addresses.
func FreeAddrs(n int) ([]string, error) {
	var ls []net.Listener
	var out []string
	defer func() {
		for _, l := range ls {
			_ = l.Close()
		}
	}()
	for i := 0; i < n; i++ {
		l, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			return nil, err
		}
		ls = append(ls, l)
		out = append(out, l.Addr().String())
	}
	return out, nil
}

// StubQuestion is the one typed question the scenarios ask; its closed choices
// are scored by the fixture model's documented rule.
func StubQuestion(id string, choices ...string) api.Question {
	return api.Question{ID: id, Type: "choice", Instructions: "Which option does the state favour?", Choices: choices}
}

// StubAnswer is the fixture model's rule (stubs/laya): weight(c) = 1 + the
// whole-word, case-insensitive occurrences of c in state, normalised; the first
// choice with the largest probability wins. The scenarios compute the expected
// answer here and compare it with what came back through the real stack, so a
// pass proves the request reached the worker's inference path, not a canned
// reply.
func StubAnswer(state string, choices []string) (string, map[string]float64) {
	text := strings.ToLower(state)
	weights := make([]float64, len(choices))
	total := 0.0
	for i, c := range choices {
		re := regexp.MustCompile(`\b` + regexp.QuoteMeta(strings.ToLower(c)) + `\b`)
		weights[i] = 1 + float64(len(re.FindAllStringIndex(text, -1)))
		total += weights[i]
	}
	probs := map[string]float64{}
	best := 0
	for i, c := range choices {
		probs[c] = weights[i] / total
		if weights[i] > weights[best] {
			best = i
		}
	}
	return choices[best], probs
}

// CheckAnswer verifies a decision against the stub rule.
func CheckAnswer(res api.DecideResponse, qid, state string, choices []string) error {
	if res.Schema != api.SchemaV1 || len(res.Results) != 1 {
		return fmt.Errorf("unexpected response shape: schema %q, %d results", res.Schema, len(res.Results))
	}
	r := res.Results[0]
	want, probs := StubAnswer(state, choices)
	if r.ID != qid || r.Type != "choice" || r.Choice != want {
		return fmt.Errorf("answered id %q type %q choice %q, want id %q choice %q", r.ID, r.Type, r.Choice, qid, want)
	}
	for c, p := range probs {
		if d := r.Probabilities[c] - p; d > 1e-9 || d < -1e-9 {
			return fmt.Errorf("probability of %q is %v, want %v", c, r.Probabilities[c], p)
		}
	}
	if d := r.Confidence - probs[want]; d > 1e-9 || d < -1e-9 {
		return fmt.Errorf("confidence %v, want %v", r.Confidence, probs[want])
	}
	return nil
}

// WaitReady waits, bounded, until /health is 200 and the worker reports ready
// with a pid, and returns that status.
func (c *Client) WaitReady(timeout time.Duration) (server.Status, error) {
	var st server.Status
	err := Eventually(timeout, 200*time.Millisecond, "worker READY", func() (bool, string, error) {
		code, h, err := c.Health()
		if err != nil {
			return false, "health unreachable: " + err.Error(), nil
		}
		s, err := c.Status()
		if err != nil {
			return false, fmt.Sprintf("health %d %q; %v", code, h.State, err), nil
		}
		st = s
		if code == 200 && h.Ready && s.Worker.State == worker.StateReady && s.Worker.PID > 0 {
			return true, "", nil
		}
		return false, fmt.Sprintf("health %d %q; worker %s phase %q", code, h.State, s.Worker.State, s.Worker.Phase), nil
	})
	return st, err
}

// WaitWorkerState waits, bounded, until the worker reports state.
func (c *Client) WaitWorkerState(state string, timeout time.Duration) (server.Status, error) {
	var st server.Status
	err := Eventually(timeout, 100*time.Millisecond, "worker state "+state, func() (bool, string, error) {
		s, err := c.Status()
		if err != nil {
			return false, err.Error(), nil
		}
		st = s
		return s.Worker.State == state, fmt.Sprintf("worker %s phase %q restarts %d", s.Worker.State, s.Worker.Phase, s.Worker.Restarts), nil
	})
	return st, err
}

// TypedRequest builds the typed decision request the scenarios send: one
// closed-choice question against a state.
func TypedRequest(state, qid string, choices []string) api.DecideRequest {
	return api.DecideRequest{Schema: api.SchemaV1, State: state, Questions: []api.Question{StubQuestion(qid, choices...)}}
}
