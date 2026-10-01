// Package worker runs and supervises the private Python inference worker.
//
// The worker speaks newline-delimited JSON over its stdin/stdout (see
// py/hachidori_worker.py). Go owns all three stdio streams: stdin carries
// requests, stdout carries protocol messages only, stderr is a log stream.
package worker

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/subprocess"
)

// Failure classes reported by the worker or detected by the supervisor.
const (
	ClassPreflight     = "preflight"          // the runtime was refused before any process started
	ClassStartup       = "worker_startup"     // process could not be started or died before hello
	ClassProviderInit  = "provider_import"    // provider stack failed to import/initialize
	ClassDevice        = "device_unavailable" // requested accelerator unavailable
	ClassModelLoad     = "model_load"
	ClassWarmup        = "warmup"
	ClassCrash         = "worker_crash" // died or violated the protocol after start
	ClassUnresponsive  = "worker_unresponsive"
	ClassStartTimeout  = "startup_timeout"
	ClassProtocolError = "protocol"
)

// Failure is a worker-level failure. It is never a semantic result.
type Failure struct {
	Class   string
	Message string
	Stderr  []string // recent worker stderr lines, for diagnostics
}

func (f *Failure) Error() string { return f.Class + ": " + f.Message }

// RequestError is a per-request failure reported by a healthy worker.
type RequestError struct {
	Class   string
	Message string
}

func (e *RequestError) Error() string { return e.Class + ": " + e.Message }

// Config describes how to launch one worker process.
type Config struct {
	Python         string   // absolute path to the private interpreter
	Args           []string // interpreter arguments, including the worker script
	Env            []string // complete environment; nothing is inherited
	Dir            string
	Log            io.Writer // receives worker stderr
	StartTimeout   time.Duration
	RequestTimeout time.Duration
	// Preflight, if set, runs before every launch. A non-nil error refuses
	// the launch: no process is started and the supervisor fails with
	// ClassPreflight and the error's text. It is for conditions that make a
	// launch pointless and whose cause the operator can fix.
	Preflight func() error
}

// Info is what the worker reports once READY.
type Info map[string]any

// Item is one decide unit sent to the worker.
type Item struct {
	State     string         `json:"state"`
	Questions []api.Question `json:"questions"`
}

type message struct {
	Event       string         `json:"event"`
	Phase       string         `json:"phase"`
	Class       string         `json:"class"`
	Message     string         `json:"message"`
	PID         int            `json:"pid"`
	Info        Info           `json:"info"`
	ID          int64          `json:"id"`
	OK          bool           `json:"ok"`
	Results     [][]api.Result `json:"results"`
	InferenceMS float64        `json:"inference_ms"`
	Stats       map[string]any `json:"stats"`
	Error       *RequestError  `json:"error"`
}

// Process is one running worker.
type Process struct {
	cfg    Config
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	msgs   chan message
	done   chan struct{}
	tail   *tail
	Info   Info
	PID    int
	mu     sync.Mutex // serializes requests; the worker is single-threaded
	nextID int64

	statsMu sync.Mutex // guards the last accelerator statistics
	stats   map[string]any

	exitOnce sync.Once
	exitErr  *Failure
}

// Start launches the worker and blocks until it is READY or fails.
// onPhase, if set, observes lifecycle phases (importing, loading, warming).
func Start(ctx context.Context, cfg Config, onPhase func(string)) (*Process, error) {
	cmd := exec.Command(cfg.Python, cfg.Args...)
	subprocess.Configure(cmd)
	cmd.Env = cfg.Env
	cmd.Dir = cfg.Dir
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, err
	}
	p := &Process{cfg: cfg, cmd: cmd, stdin: stdin, msgs: make(chan message, 16),
		done: make(chan struct{}), tail: newTail(64)}
	if err := cmd.Start(); err != nil {
		return nil, &Failure{Class: ClassStartup, Message: err.Error()}
	}
	p.PID = cmd.Process.Pid

	var readers sync.WaitGroup
	readers.Add(2)
	go func() { defer readers.Done(); p.copyStderr(stderr) }()
	go func() { defer readers.Done(); p.readStdout(stdout) }()
	go func() {
		readers.Wait()
		err := cmd.Wait()
		p.setExit(&Failure{Class: ClassCrash, Message: fmt.Sprintf("worker exited: %v", exitText(err))})
		close(p.done)
	}()

	timeout := time.NewTimer(cfg.StartTimeout)
	defer timeout.Stop()
	hello := false
	// handle applies one startup message. It returns done=true once startup
	// has ended: with the process when ready, or with the failure.
	handle := func(m message) (done bool, err error) {
		switch m.Event {
		case "hello":
			hello = true
		case "phase":
			if onPhase != nil {
				onPhase(m.Phase)
			}
		case "fatal":
			p.setExit(&Failure{Class: m.Class, Message: m.Message})
			p.kill()
			<-p.done
			return true, p.failure()
		case "ready":
			p.Info = m.Info
			return true, nil
		default:
			p.setExit(&Failure{Class: ClassProtocolError, Message: "unexpected message before ready"})
			p.kill()
			<-p.done
			return true, p.failure()
		}
		return false, nil
	}
	for {
		select {
		case m := <-p.msgs:
			if done, err := handle(m); done {
				if err != nil {
					return nil, err
				}
				return p, nil
			}
		case <-p.done:
			// The process is gone, but everything it wrote before exiting
			// is already queued (p.done closes only after both pipes were
			// read to the end). Apply it first: a worker that reported its
			// phase or a fatal class and then exited must be attributed to
			// that phase and class, not to a startup that never began.
			for {
				select {
				case m := <-p.msgs:
					if done, err := handle(m); done {
						if err != nil {
							return nil, err
						}
						return p, nil
					}
					continue
				default:
				}
				break
			}
			f := p.failure()
			if !hello {
				f.Class = ClassStartup
			}
			return nil, f
		case <-timeout.C:
			p.setExit(&Failure{Class: ClassStartTimeout, Message: fmt.Sprintf("not ready after %s", cfg.StartTimeout)})
			p.kill()
			<-p.done
			return nil, p.failure()
		case <-ctx.Done():
			p.setExit(&Failure{Class: ClassStartup, Message: "startup cancelled"})
			p.kill()
			<-p.done
			return nil, p.failure()
		}
	}
}

// Done is closed when the worker process has exited.
func (p *Process) Done() <-chan struct{} { return p.done }

// ExitFailure describes why the worker exited; valid after Done is closed.
func (p *Process) ExitFailure() *Failure { return p.failure() }

func (p *Process) failure() *Failure {
	f := *p.exitErr
	f.Stderr = p.tail.lines()
	return &f
}

func (p *Process) setExit(f *Failure) { p.exitOnce.Do(func() { p.exitErr = f }) }

func (p *Process) kill() {
	if p.cmd.Process != nil {
		_ = p.cmd.Process.Kill()
	}
}

// Decide answers items on the resident model.
func (p *Process) Decide(items []Item) ([][]api.Result, float64, error) {
	m, err := p.call(map[string]any{"op": "decide", "items": items})
	if err != nil {
		return nil, 0, err
	}
	if len(m.Results) != len(items) {
		p.setExit(&Failure{Class: ClassProtocolError, Message: "result count mismatch"})
		p.kill()
		return nil, 0, p.failure()
	}
	return m.Results, m.InferenceMS, nil
}

// Stats returns accelerator statistics from the worker. It waits behind an
// in-flight request, because the worker serves one request at a time.
func (p *Process) Stats() (map[string]any, error) {
	m, err := p.call(map[string]any{"op": "stats"})
	if err != nil {
		return nil, err
	}
	p.keepStats(m.Stats)
	return m.Stats, nil
}

// TryStats is Stats for observers that must not wait: when a request is in
// flight it returns the last statistics taken while the worker was idle and
// stale=true, without sending anything to the worker. The worker's serialized
// request contract is unchanged; nothing is queued behind a long inference.
func (p *Process) TryStats() (stats map[string]any, stale bool, err error) {
	if !p.mu.TryLock() {
		return p.lastStats(), true, nil
	}
	m, err := p.callLocked(map[string]any{"op": "stats"})
	p.mu.Unlock()
	if err != nil {
		return nil, false, err
	}
	p.keepStats(m.Stats)
	return m.Stats, false, nil
}

func (p *Process) keepStats(st map[string]any) {
	p.statsMu.Lock()
	p.stats = st
	p.statsMu.Unlock()
}

func (p *Process) lastStats() map[string]any {
	p.statsMu.Lock()
	defer p.statsMu.Unlock()
	return p.stats
}

// call sends one request and waits for its response. An in-flight forward
// pass cannot be abandoned, so callers cannot cancel; only RequestTimeout
// (unresponsive worker) ends the wait early.
func (p *Process) call(req map[string]any) (message, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.callLocked(req)
}

// callLocked is call with p.mu already held.
func (p *Process) callLocked(req map[string]any) (message, error) {
	select {
	case <-p.done:
		return message{}, p.failure()
	default:
	}
	p.nextID++
	id := p.nextID
	req["id"] = id
	line, _ := json.Marshal(req)
	if _, err := p.stdin.Write(append(line, '\n')); err != nil {
		p.kill()
		<-p.done
		return message{}, p.failure()
	}
	timer := time.NewTimer(p.cfg.RequestTimeout)
	defer timer.Stop()
	for {
		select {
		case m := <-p.msgs:
			if m.Event == "fatal" {
				p.setExit(&Failure{Class: m.Class, Message: m.Message})
				continue // wait for exit
			}
			if m.ID != id {
				p.setExit(&Failure{Class: ClassProtocolError, Message: fmt.Sprintf("response id %d, want %d", m.ID, id)})
				p.kill()
				<-p.done
				return message{}, p.failure()
			}
			if !m.OK {
				if m.Error == nil {
					m.Error = &RequestError{Class: api.ErrInferenceFailed, Message: "unspecified worker error"}
				}
				return m, m.Error
			}
			return m, nil
		case <-p.done:
			return message{}, p.failure()
		case <-timer.C:
			p.setExit(&Failure{Class: ClassUnresponsive, Message: fmt.Sprintf("no response within %s", p.cfg.RequestTimeout)})
			p.kill()
			<-p.done
			return message{}, p.failure()
		}
	}
}

// Close asks the worker to shut down, then kills it after a grace period.
func (p *Process) Close() {
	p.setExit(&Failure{Class: "stopped", Message: "shutdown requested"})
	_, _ = p.stdin.Write([]byte(`{"op":"shutdown"}` + "\n"))
	_ = p.stdin.Close()
	select {
	case <-p.done:
	case <-time.After(10 * time.Second):
		p.kill()
		<-p.done
	}
}

func (p *Process) readStdout(r io.Reader) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 16<<20)
	for sc.Scan() {
		var m message
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			p.setExit(&Failure{Class: ClassProtocolError, Message: "non-protocol output on stdout: " + truncate(sc.Text(), 200)})
			p.kill()
			break
		}
		if m.Event == "fatal" {
			// The worker's own account of why it is ending. Recorded here,
			// by the reader that always finishes before the exit is judged,
			// so a consumer that is slow to take the message cannot let the
			// generic "worker exited" take its place.
			p.setExit(&Failure{Class: m.Class, Message: m.Message})
		}
		select {
		case p.msgs <- m:
		default:
			// Responses are strictly one per serialized request; a full
			// buffer means the worker is emitting unsolicited messages.
			p.setExit(&Failure{Class: ClassProtocolError, Message: "unsolicited worker output"})
			p.kill()
		}
	}
	// A protocol line over the limit (or an unreadable pipe) ends the scan
	// with the worker still alive. Nothing it sends could be read any more,
	// so report it as the protocol failure it is now, instead of letting the
	// next request time out as an unresponsive worker.
	if err := sc.Err(); err != nil {
		p.setExit(&Failure{Class: ClassProtocolError, Message: "worker protocol output unreadable: " + err.Error()})
		p.kill()
	}
	_, _ = io.Copy(io.Discard, r)
}

// maxTailLine bounds one retained stderr line. The tail is republished by
// /v1/status and the dashboard, so one pathological line must not grow them.
// The full line still goes to the worker log.
const maxTailLine = 4 << 10

func (p *Process) copyStderr(r io.Reader) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		p.tail.add(truncate(line, maxTailLine))
		if p.cfg.Log != nil {
			fmt.Fprintln(p.cfg.Log, line)
		}
	}
	// A line over the scanner limit ends the scan with the pipe still open.
	// Keep draining it: a worker blocked writing stderr stops answering and
	// would be misreported as unresponsive.
	if err := sc.Err(); err != nil {
		p.tail.add("worker stderr line exceeded the capture limit: " + err.Error())
		_, _ = io.Copy(io.Discard, r)
	}
}

func exitText(err error) string {
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		return ee.String()
	}
	if err == nil {
		return "exit status 0"
	}
	return err.Error()
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

type tail struct {
	mu  sync.Mutex
	buf []string
	max int
}

func newTail(n int) *tail { return &tail{max: n} }

func (t *tail) add(s string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf = append(t.buf, s)
	if len(t.buf) > t.max {
		t.buf = t.buf[len(t.buf)-t.max:]
	}
}

func (t *tail) lines() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]string(nil), t.buf...)
}
