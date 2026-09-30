// Package tunnel supervises one operator-requested SSH reverse tunnel that
// carries a caller's loopback endpoint to the loopback Hachidori API.
//
// It is a thin launcher around the host's already-configured `ssh` client.
// SSH identity, keys, agents, known_hosts and server configuration remain
// external authority: this package never reads, writes or prompts for them.
// The child is started with a fixed argument vector (never through a shell)
// built only from a validated destination and validated addresses/ports.
package tunnel

import (
	"bytes"
	"errors"
	"fmt"
	"net"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/yohn-jp/hachidori/internal/subprocess"
)

// Tunnel states.
const (
	StateIdle    = "idle"    // nothing started in this process
	StateRunning = "running" // the managed ssh child is alive
	StateStopped = "stopped" // Disconnect terminated the child
	StateExited  = "exited"  // the child exited (or failed to start) without Disconnect
)

// Defaults for the dashboard form.
const (
	DefaultRemoteBind = "127.0.0.1"
	DefaultPort       = 7843
)

// Spec is the complete, non-secret description of one reverse tunnel.
type Spec struct {
	Destination string `json:"destination"` // [user@]host or an ssh_config Host alias
	RemoteBind  string `json:"remote_bind"` // loopback address on the caller host
	RemotePort  int    `json:"remote_port"`
	LocalPort   int    `json:"local_port"` // Hachidori's port on this host's 127.0.0.1
}

var (
	userRe = regexp.MustCompile(`^[A-Za-z0-9_][A-Za-z0-9._-]{0,63}$`)
	hostRe = regexp.MustCompile(`^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$`)
)

// Validate accepts only a plain destination and a loopback-to-loopback
// forward. Remote binds are loopback-only: a non-loopback bind would expose
// the unauthenticated inference API beyond the caller host.
func (s Spec) Validate() error {
	host := s.Destination
	if i := strings.IndexByte(s.Destination, '@'); i >= 0 {
		if user := s.Destination[:i]; !userRe.MatchString(user) {
			return fmt.Errorf("invalid ssh user %q", user)
		}
		host = s.Destination[i+1:]
	}
	if !hostRe.MatchString(host) {
		return fmt.Errorf("invalid ssh destination %q: want [user@]host or an ssh_config Host alias", s.Destination)
	}
	if !isLoopback(s.RemoteBind) {
		return fmt.Errorf("remote bind %q must be a loopback address (127.0.0.1, ::1 or localhost)", s.RemoteBind)
	}
	if s.RemotePort < 1 || s.RemotePort > 65535 {
		return fmt.Errorf("remote port %d out of range 1-65535", s.RemotePort)
	}
	if s.LocalPort < 1 || s.LocalPort > 65535 {
		return fmt.Errorf("local port %d out of range 1-65535", s.LocalPort)
	}
	return nil
}

func isLoopback(h string) bool {
	if h == "localhost" {
		return true
	}
	ip := net.ParseIP(h)
	return ip != nil && ip.IsLoopback()
}

// Args is the ssh argument vector for s. The options are fixed: no remote
// command (-N), exit if the forward cannot be established, never prompt
// (authentication and host-key trust come from the host's own SSH setup),
// and notice a dead connection.
func (s Spec) Args() []string {
	bind := s.RemoteBind
	if ip := net.ParseIP(bind); ip != nil && ip.To4() == nil {
		bind = "[" + bind + "]"
	}
	return []string{
		"-N",
		"-o", "ExitOnForwardFailure=yes",
		"-o", "BatchMode=yes",
		"-o", "ServerAliveInterval=15",
		"-o", "ServerAliveCountMax=3",
		"-R", bind + ":" + strconv.Itoa(s.RemotePort) + ":127.0.0.1:" + strconv.Itoa(s.LocalPort),
		"--", s.Destination,
	}
}

// CallerEndpoint is the HACHIDORI_ENDPOINT the caller host should use.
func (s Spec) CallerEndpoint() string {
	return "http://" + net.JoinHostPort(s.RemoteBind, strconv.Itoa(s.RemotePort))
}

// Status is the observable tunnel state.
type Status struct {
	State     string    `json:"state"`
	Spec      *Spec     `json:"spec,omitempty"`
	Argv      []string  `json:"argv,omitempty"`
	PID       int       `json:"pid,omitempty"`
	Started   time.Time `json:"started,omitzero"`
	Exited    time.Time `json:"exited,omitzero"`
	LastError string    `json:"last_error,omitempty"`
	Stderr    []string  `json:"stderr_tail,omitempty"`
	Endpoint  string    `json:"caller_endpoint,omitempty"`
}

// ErrBusy is returned by Connect while a tunnel with a different spec runs.
var ErrBusy = errors.New("a different tunnel is running; disconnect it first")

// Manager owns at most one ssh child, and only one that it started itself.
type Manager struct {
	SSH   string        // ssh executable; resolved through PATH when not a path
	Grace time.Duration // wait after SIGTERM before killing

	op       sync.Mutex // serializes Connect/Disconnect
	mu       sync.Mutex // guards the fields below
	st       Status
	cmd      *exec.Cmd
	done     chan struct{}
	stopping bool
	tail     []string
}

// NewManager returns a manager for the given ssh executable ("ssh" if empty).
func NewManager(ssh string) *Manager {
	if ssh == "" {
		ssh = "ssh"
	}
	return &Manager{SSH: ssh, Grace: 3 * time.Second, st: Status{State: StateIdle}}
}

// Status returns a copy of the current state.
func (m *Manager) Status() Status {
	m.mu.Lock()
	defer m.mu.Unlock()
	st := m.st
	st.Stderr = append([]string(nil), m.tail...)
	return st
}

// Connect starts the tunnel. Connecting again with an identical spec while it
// runs is a no-op (started=false); a different spec returns ErrBusy.
func (m *Manager) Connect(s Spec) (started bool, err error) {
	if err := s.Validate(); err != nil {
		return false, err
	}
	m.op.Lock()
	defer m.op.Unlock()
	m.mu.Lock()
	running, same := m.cmd != nil, m.st.Spec != nil && *m.st.Spec == s
	m.mu.Unlock()
	if running && same {
		return false, nil
	}
	if running {
		return false, ErrBusy
	}

	exe, err := exec.LookPath(m.SSH)
	if err != nil {
		err = fmt.Errorf("ssh client not found: %w", err)
		m.fail(s, err)
		return false, err
	}
	argv := s.Args()
	cmd := exec.Command(exe, argv...)
	subprocess.Configure(cmd)
	cmd.Stderr = lineWriter{m}
	cmd.WaitDelay = time.Second // do not hang on stderr held open by an ssh helper
	m.mu.Lock()
	m.tail = nil
	m.mu.Unlock()
	if err := cmd.Start(); err != nil {
		m.fail(s, err)
		return false, err
	}
	done := make(chan struct{})
	m.mu.Lock()
	m.cmd, m.done, m.stopping = cmd, done, false
	m.st = Status{State: StateRunning, Spec: &s, Argv: append([]string{exe}, argv...), PID: cmd.Process.Pid,
		Started: time.Now(), Endpoint: s.CallerEndpoint()}
	m.mu.Unlock()
	go m.wait(cmd, done)
	return true, nil
}

func (m *Manager) fail(s Spec, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.st = Status{State: StateExited, Spec: &s, Exited: time.Now(), LastError: err.Error()}
}

func (m *Manager) wait(cmd *exec.Cmd, done chan struct{}) {
	err := cmd.Wait()
	m.mu.Lock()
	m.st.Exited = time.Now()
	if m.stopping {
		m.st.State = StateStopped
	} else {
		m.st.State = StateExited
		m.st.LastError = "ssh exited unexpectedly: " + exitText(err)
	}
	m.cmd = nil
	m.mu.Unlock()
	close(done)
}

// Disconnect terminates the managed child (SIGTERM, then kill after Grace;
// kill directly where signals are unsupported) and waits until it exited.
func (m *Manager) Disconnect() {
	m.op.Lock()
	defer m.op.Unlock()
	m.mu.Lock()
	cmd, done := m.cmd, m.done
	if cmd != nil {
		m.stopping = true
	}
	m.mu.Unlock()
	if cmd == nil {
		return
	}
	if err := cmd.Process.Signal(syscall.SIGTERM); err != nil {
		_ = cmd.Process.Kill()
	}
	select {
	case <-done:
	case <-time.After(m.Grace):
		_ = cmd.Process.Kill()
		<-done
	}
}

// lineWriter keeps the last lines of ssh stderr for display.
type lineWriter struct{ m *Manager }

func (w lineWriter) Write(p []byte) (int, error) {
	w.m.mu.Lock()
	defer w.m.mu.Unlock()
	for _, l := range bytes.Split(bytes.TrimRight(p, "\r\n"), []byte("\n")) {
		if l = bytes.TrimRight(l, "\r"); len(l) > 0 {
			if len(l) > 512 {
				l = l[:512]
			}
			w.m.tail = append(w.m.tail, string(l))
		}
	}
	if len(w.m.tail) > 20 {
		w.m.tail = w.m.tail[len(w.m.tail)-20:]
	}
	return len(p), nil
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
