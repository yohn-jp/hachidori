package tunnel

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// TestMain doubles as a fake ssh executable: when HACHIDORI_FAKE_SSH is set the
// test binary records its argv and behaves like ssh in the selected mode.
// No real SSH client or server is involved.
func TestMain(m *testing.M) {
	if mode := os.Getenv("HACHIDORI_FAKE_SSH"); mode != "" {
		fakeSSH(mode)
		return
	}
	os.Exit(m.Run())
}

func fakeSSH(mode string) {
	if path := os.Getenv("HACHIDORI_FAKE_SSH_ARGV"); path != "" {
		b, _ := json.Marshal(os.Args[1:])
		_ = os.WriteFile(path, b, 0o644)
	}
	switch mode {
	case "auth_fail":
		fmt.Fprintln(os.Stderr, "user@host: Permission denied (publickey).")
		os.Exit(255)
	case "die_later":
		fmt.Fprintln(os.Stderr, "Warning: remote port forwarding failed for listen port 7843")
		time.Sleep(300 * time.Millisecond)
		os.Exit(255)
	default: // "run": stay up until terminated, like ssh -N
		fmt.Fprintln(os.Stderr, "fake ssh up")
		time.Sleep(time.Minute)
		os.Exit(0)
	}
}

// fakeManager returns a Manager whose ssh is this test binary.
func fakeManager(t *testing.T, mode string) (*Manager, string) {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	argv := filepath.Join(t.TempDir(), "argv.json")
	t.Setenv("HACHIDORI_FAKE_SSH", mode)
	t.Setenv("HACHIDORI_FAKE_SSH_ARGV", argv)
	m := NewManager(exe)
	m.Grace = 2 * time.Second
	t.Cleanup(m.Disconnect)
	return m, argv
}

var spec = Spec{Destination: "dev@nixos", RemoteBind: "127.0.0.1", RemotePort: 7843, LocalPort: 7843}

func waitFor(t *testing.T, m *Manager, state string) Status {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		st := m.Status()
		if st.State == state {
			return st
		}
		if time.Now().After(deadline) {
			t.Fatalf("state = %s, want %s (%+v)", st.State, state, st)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestValidate(t *testing.T) {
	good := []Spec{
		spec,
		{Destination: "nixos-vm", RemoteBind: "localhost", RemotePort: 1, LocalPort: 65535},
		{Destination: "a.b-c.example", RemoteBind: "::1", RemotePort: 17843, LocalPort: 7843},
		{Destination: "user_1@192.168.1.20", RemoteBind: "127.0.0.2", RemotePort: 7843, LocalPort: 7843},
	}
	for _, s := range good {
		if err := s.Validate(); err != nil {
			t.Errorf("%+v: %v", s, err)
		}
	}
	bad := map[string]Spec{}
	for _, d := range []string{"", "-oProxyCommand=calc", "user@-oProxyCommand=x", "host;rm -rf /", "a b", "h$(id)",
		"user@", "@host", "u@h@x", "host:22", "ssh://h", "h\nx", "`id`", "-", "h|x", "..", "host/..", "-host"} {
		s := spec
		s.Destination = d
		bad["destination "+d] = s
	}
	for _, b := range []string{"0.0.0.0", "", "*", "192.168.1.2", "::", "example.com", "127.0.0.1 -oX", "localhost:1"} {
		s := spec
		s.RemoteBind = b
		bad["bind "+b] = s
	}
	for _, p := range []int{0, -1, 65536, 1 << 20} {
		s := spec
		s.RemotePort = p
		bad[fmt.Sprint("remote port ", p)] = s
		s = spec
		s.LocalPort = p
		bad[fmt.Sprint("local port ", p)] = s
	}
	for name, s := range bad {
		if s.Validate() == nil {
			t.Errorf("%s accepted", name)
		}
	}
}

func TestArgsAreAFixedVector(t *testing.T) {
	want := []string{"-N", "-o", "ExitOnForwardFailure=yes", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15",
		"-o", "ServerAliveCountMax=3", "-R", "127.0.0.1:7843:127.0.0.1:7843", "--", "dev@nixos"}
	if got := spec.Args(); !reflect.DeepEqual(got, want) {
		t.Fatalf("args = %q", got)
	}
	s := spec
	s.RemoteBind, s.RemotePort, s.LocalPort = "::1", 9000, 7844
	if got := s.Args()[10]; got != "[::1]:9000:127.0.0.1:7844" {
		t.Fatalf("ipv6 forward = %q", got)
	}
	if s.CallerEndpoint() != "http://[::1]:9000" || spec.CallerEndpoint() != "http://127.0.0.1:7843" {
		t.Fatal(s.CallerEndpoint(), spec.CallerEndpoint())
	}
}

func TestConnectExecsArgvDirectly(t *testing.T) {
	m, argvFile := fakeManager(t, "run")
	started, err := m.Connect(spec)
	if err != nil || !started {
		t.Fatalf("connect: %v %v", started, err)
	}
	st := waitFor(t, m, StateRunning)
	if st.PID == 0 || st.Started.IsZero() || st.Endpoint != "http://127.0.0.1:7843" || *st.Spec != spec {
		t.Fatalf("status %+v", st)
	}
	deadline := time.Now().Add(5 * time.Second)
	var got []string
	for {
		b, err := os.ReadFile(argvFile)
		if err == nil && json.Unmarshal(b, &got) == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("fake ssh did not record argv")
		}
		time.Sleep(10 * time.Millisecond)
	}
	// The child receives exactly the validated vector: no shell, no re-splitting.
	if !reflect.DeepEqual(got, spec.Args()) || !reflect.DeepEqual(st.Argv[1:], spec.Args()) {
		t.Fatalf("child argv = %q", got)
	}
}

func TestInvalidSpecStartsNothing(t *testing.T) {
	m, argvFile := fakeManager(t, "run")
	s := spec
	s.Destination = "x; touch pwned"
	if _, err := m.Connect(s); err == nil {
		t.Fatal("accepted")
	}
	time.Sleep(100 * time.Millisecond)
	if _, err := os.Stat(argvFile); err == nil || m.Status().State != StateIdle {
		t.Fatalf("a process was started: %+v", m.Status())
	}
}

func TestDuplicateConnect(t *testing.T) {
	m, _ := fakeManager(t, "run")
	if ok, err := m.Connect(spec); !ok || err != nil {
		t.Fatal(ok, err)
	}
	pid := waitFor(t, m, StateRunning).PID
	for i := 0; i < 3; i++ {
		if ok, err := m.Connect(spec); ok || err != nil {
			t.Fatalf("duplicate connect: started=%v err=%v", ok, err)
		}
	}
	other := spec
	other.RemotePort = 9999
	if _, err := m.Connect(other); err != ErrBusy {
		t.Fatalf("different spec while running: %v", err)
	}
	if st := m.Status(); st.PID != pid || st.State != StateRunning || *st.Spec != spec {
		t.Fatalf("status changed: %+v", st)
	}
}

func TestConcurrentConnectStartsOne(t *testing.T) {
	m, _ := fakeManager(t, "run")
	results := make(chan bool, 8)
	for i := 0; i < 8; i++ {
		go func() { ok, _ := m.Connect(spec); results <- ok }()
	}
	n := 0
	for i := 0; i < 8; i++ {
		if <-results {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("%d tunnels started", n)
	}
}

func TestDisconnectIsDeterministic(t *testing.T) {
	m, _ := fakeManager(t, "run")
	m.Disconnect() // nothing to do
	if m.Status().State != StateIdle {
		t.Fatal(m.Status())
	}
	for i := 0; i < 2; i++ {
		if ok, err := m.Connect(spec); !ok || err != nil {
			t.Fatalf("round %d: %v %v", i, ok, err)
		}
		waitFor(t, m, StateRunning)
		m.Disconnect()
		// Disconnect returns only after the child has been reaped.
		st := m.Status()
		if st.State != StateStopped || st.LastError != "" || st.Exited.IsZero() {
			t.Fatalf("round %d: after disconnect %+v", i, st)
		}
		m.Disconnect()
		if m.Status().State != StateStopped {
			t.Fatal("second disconnect changed state")
		}
	}
}

func TestUnexpectedExitIsObservable(t *testing.T) {
	m, _ := fakeManager(t, "die_later")
	if ok, err := m.Connect(spec); !ok || err != nil {
		t.Fatal(ok, err)
	}
	st := waitFor(t, m, StateExited)
	if !strings.Contains(st.LastError, "ssh exited unexpectedly") || !strings.Contains(st.LastError, "255") {
		t.Fatalf("last error %q", st.LastError)
	}
	if len(st.Stderr) == 0 || !strings.Contains(st.Stderr[0], "remote port forwarding failed") {
		t.Fatalf("stderr %q", st.Stderr)
	}
	// After an exit the same spec can be connected again.
	if ok, err := m.Connect(spec); !ok || err != nil {
		t.Fatal(ok, err)
	}
}

func TestAuthFailureIsSurfaced(t *testing.T) {
	m, _ := fakeManager(t, "auth_fail")
	if _, err := m.Connect(spec); err != nil {
		t.Fatal(err)
	}
	st := waitFor(t, m, StateExited)
	if len(st.Stderr) == 0 || !strings.Contains(strings.Join(st.Stderr, "\n"), "Permission denied") {
		t.Fatalf("%+v", st)
	}
}

func TestMissingSSHClient(t *testing.T) {
	m := NewManager(filepath.Join(t.TempDir(), "no-such-ssh"))
	if _, err := m.Connect(spec); err == nil {
		t.Fatal("connect without ssh succeeded")
	}
	if st := m.Status(); st.State != StateExited || !strings.Contains(st.LastError, "ssh client not found") {
		t.Fatalf("%+v", st)
	}
}
