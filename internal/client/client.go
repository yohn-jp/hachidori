// Package client calls a Hachidori endpoint over the public v1 contract.
package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
)

// DefaultEndpoint is used when HACHIDORI_ENDPOINT is unset.
const DefaultEndpoint = "http://127.0.0.1:7843"

// Client is a minimal HTTP client for the v1 API.
type Client struct {
	Endpoint string
	HTTP     *http.Client
}

// New resolves the endpoint from flag, then HACHIDORI_ENDPOINT, then the default.
func New(flag string) *Client {
	ep := flag
	if ep == "" {
		ep = os.Getenv("HACHIDORI_ENDPOINT")
	}
	if ep == "" {
		ep = DefaultEndpoint
	}
	return &Client{Endpoint: strings.TrimRight(ep, "/"), HTTP: &http.Client{Timeout: 5 * time.Minute}}
}

// APIError is a structured error returned by the endpoint.
type APIError struct {
	Status int
	api.ErrorInfo
}

func (e *APIError) Error() string {
	return fmt.Sprintf("%s (HTTP %d): %s", e.Class, e.Status, e.Message)
}

// Decide calls POST /v1/decide.
func (c *Client) Decide(req api.DecideRequest) (api.DecideResponse, error) {
	return c.DecideContext(context.Background(), req)
}

// DecideContext calls POST /v1/decide and follows the caller's cancellation.
func (c *Client) DecideContext(ctx context.Context, req api.DecideRequest) (api.DecideResponse, error) {
	var out api.DecideResponse
	err := c.doContext(ctx, "POST", "/v1/decide", req, &out)
	return out, err
}

// RegisterState calls POST /v1/states.
func (c *Client) RegisterState(req api.RegisterState) (api.StateReference, error) {
	return c.RegisterStateContext(context.Background(), req)
}

// RegisterStateContext calls POST /v1/states and follows the caller's cancellation.
func (c *Client) RegisterStateContext(ctx context.Context, req api.RegisterState) (api.StateReference, error) {
	var out api.StateReference
	err := c.doContext(ctx, "POST", "/v1/states", req, &out)
	return out, err
}

// DecideBatch calls POST /v1/decide/batch.
func (c *Client) DecideBatch(req api.BatchRequest) (api.BatchResponse, error) {
	var out api.BatchResponse
	err := c.do("POST", "/v1/decide/batch", req, &out)
	return out, err
}

// Status calls GET /v1/status and returns the raw JSON document.
func (c *Client) Status() (json.RawMessage, error) {
	var out json.RawMessage
	err := c.do("GET", "/v1/status", nil, &out)
	return out, err
}

// Health calls GET /health.
func (c *Client) Health() (api.Health, error) {
	var out api.Health
	err := c.do("GET", "/health", nil, &out)
	if ae, ok := err.(*APIError); ok && ae.Status == http.StatusServiceUnavailable {
		return out, nil
	}
	return out, err
}

func (c *Client) do(method, path string, body, out any) error {
	return c.doContext(context.Background(), method, path, body, out)
}

func (c *Client) doContext(ctx context.Context, method, path string, body, out any) error {
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.Endpoint+path, rd)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		var eb api.ErrorBody
		if json.Unmarshal(data, &eb) == nil && eb.Error.Class != "" {
			if out != nil {
				_ = json.Unmarshal(data, out)
			}
			return &APIError{Status: resp.StatusCode, ErrorInfo: eb.Error}
		}
		if out != nil && json.Unmarshal(data, out) == nil && resp.StatusCode == http.StatusServiceUnavailable {
			return &APIError{Status: resp.StatusCode, ErrorInfo: api.ErrorInfo{Class: api.ErrNotReady}}
		}
		return fmt.Errorf("%s %s: HTTP %d: %s", method, path, resp.StatusCode, strings.TrimSpace(string(data)))
	}
	return json.Unmarshal(data, out)
}
