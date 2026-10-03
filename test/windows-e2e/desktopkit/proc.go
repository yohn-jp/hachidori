package desktopkit

import (
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"sync"
	"time"
)

// TailBytes bounds the output a Proc retains.
const TailBytes = 64 << 10

// Tail is a goroutine-safe writer that keeps only the last TailBytes written.
type Tail struct {
	mu  sync.Mutex
	buf []byte
	cut bool
}

func (t *Tail) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf = append(t.buf, p...)
	if len(t.buf) > TailBytes {
		t.buf = append(t.buf[:0], t.buf[len(t.buf)-TailBytes:]...)
		t.cut = true
	}
	return len(p), nil
}

// String returns the retained output.
func (t *Tail) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.cut {
		return "[earlier output truncated]\n" + string(t.buf)
	}
	return string(t.buf)
}

// Proc is one running subprocess whose combined stdout and stderr are kept in
// a bounded tail.
type Proc struct {
	Name string
	Pid  int

	cmd  *exec.Cmd
	out  Tail
	done chan struct{}
	code int
	werr error
}

// Start runs name with args in env (a complete environment) and dir.
func Start(name string, args, env []string, dir string) (*Proc, error) {
	p := &Proc{Name: name, done: make(chan struct{})}
	p.cmd = exec.Command(name, args...)
	p.cmd.Env, p.cmd.Dir = env, dir
	p.cmd.Stdout, p.cmd.Stderr = &p.out, &p.out
	if err := p.cmd.Start(); err != nil {
		return nil, err
	}
	p.Pid = p.cmd.Process.Pid
	go func() {
		err := p.cmd.Wait()
		p.code = 0
		var ee *exec.ExitError
		switch {
		case err == nil:
		case errors.As(err, &ee):
			p.code = ee.ExitCode()
		default:
			p.code, p.werr = -1, err
		}
		close(p.done)
	}()
	return p, nil
}

// Done is closed when the process has exited.
func (p *Proc) Done() <-chan struct{} { return p.done }

// Exited reports the exit code once the process has exited.
func (p *Proc) Exited() (code int, exited bool) {
	select {
	case <-p.done:
		return p.code, true
	default:
		return 0, false
	}
}

// Output is the retained combined output.
func (p *Proc) Output() string { return p.out.String() }

// WaitExit waits up to timeout for the process to exit on its own and returns
// its exit code.
func (p *Proc) WaitExit(timeout time.Duration) (int, error) {
	select {
	case <-p.done:
		if p.werr != nil {
			return p.code, p.werr
		}
		return p.code, nil
	case <-time.After(timeout):
		return 0, fmt.Errorf("pid %d did not exit within %s", p.Pid, timeout)
	}
}

// Alive reports whether the process has not exited.
func (p *Proc) Alive() bool { _, exited := p.Exited(); return !exited }

// Kill terminates the process and everything it started, then waits a bounded
// time for it to be reaped. It is safe to call after the process exited.
func (p *Proc) Kill() {
	if !p.Alive() {
		return
	}
	killTree(p.cmd.Process)
	select {
	case <-p.done:
	case <-time.After(15 * time.Second):
	}
}

// FreeAddr returns a loopback host:port that was free a moment ago.
func FreeAddr() (string, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", err
	}
	addr := l.Addr().String()
	return addr, l.Close()
}

// FreeAddrs returns n distinct free loopback addresses. All listeners are held
// until every address is chosen, so the addresses differ from one another.
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

// Dialable reports whether something accepts TCP connections on addr.
func Dialable(addr string) bool {
	c, err := net.DialTimeout("tcp", addr, time.Second)
	if err != nil {
		return false
	}
	_ = c.Close()
	return true
}

// MkdirTemp creates a short disposable directory (short paths matter: the deep
// path scenarios budget the total path length) and returns it with a cleanup
// that ignores errors, because Windows may still hold a handle for a moment.
func MkdirTemp(prefix string) (string, func(), error) {
	dir, err := os.MkdirTemp("", prefix)
	if err != nil {
		return "", nil, err
	}
	return dir, func() { _ = os.RemoveAll(dir) }, nil
}
