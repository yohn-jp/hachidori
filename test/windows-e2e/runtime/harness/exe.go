package harness

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
)

// Profile is a disposable Windows user profile for a child process: every
// location the product reads or writes outside HACHIDORI_HOME (the bootstrap
// locator and desktop preferences under LOCALAPPDATA, the roaming profile, the
// user profile, temporary files) points into one scenario-owned directory, so
// a scenario can never touch a real Hachidori home or profile.
type Profile struct {
	Dir          string
	LocalAppData string
	AppData      string
	UserProfile  string
	Temp         string
}

// NewProfile creates the profile directories under dir.
func NewProfile(dir string) (Profile, error) {
	p := Profile{
		Dir:          dir,
		LocalAppData: filepath.Join(dir, "LocalAppData"),
		AppData:      filepath.Join(dir, "AppData"),
		UserProfile:  filepath.Join(dir, "User"),
		Temp:         filepath.Join(dir, "Temp"),
	}
	for _, d := range []string{p.LocalAppData, p.AppData, p.UserProfile, p.Temp} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return Profile{}, err
		}
	}
	return p, nil
}

// Locator is the profile's bootstrap locator (%LOCALAPPDATA%\Hachidori\bootstrap.json).
func (p Profile) Locator() home.Locator {
	return home.Locator{Path: filepath.Join(p.LocalAppData, "Hachidori", "bootstrap.json")}
}

// inherited is the allow-list of parent environment variables a launched
// executable receives: what Windows and a desktop process need to start, and
// nothing else. HACHIDORI_HOME, the profile variables, proxies and every token
// or secret of the runner are deliberately absent.
var inherited = []string{
	"SystemRoot", "SYSTEMROOT", "windir", "SystemDrive", "ComSpec", "PATHEXT", "PATH", "Path",
	"ProgramData", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "CommonProgramFiles",
	"CommonProgramFiles(x86)", "CommonProgramW6432", "ALLUSERSPROFILE", "PUBLIC", "OS",
	"PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "NUMBER_OF_PROCESSORS", "COMPUTERNAME",
	"USERNAME", "USERDOMAIN", "SESSIONNAME", "LOGONSERVER",
}

// Env builds the complete environment of a child: the allow-listed parent
// variables, then the disposable profile. extra entries ("K=V") come last.
func (p Profile) Env(parent func(string) (string, bool), extra ...string) []string {
	var env []string
	for _, k := range inherited {
		if v, ok := parent(k); ok {
			env = append(env, k+"="+v)
		}
	}
	vol := filepath.VolumeName(p.UserProfile)
	env = append(env,
		"LOCALAPPDATA="+p.LocalAppData,
		"APPDATA="+p.AppData,
		"USERPROFILE="+p.UserProfile,
		"HOME="+p.UserProfile,
		"HOMEDRIVE="+vol,
		"HOMEPATH="+strings.TrimPrefix(p.UserProfile, vol),
		"TEMP="+p.Temp,
		"TMP="+p.Temp,
	)
	return append(env, extra...)
}

// Exe is a launched hachidori.exe.
type Exe struct {
	Cmd     *exec.Cmd
	LogPath string
	APIAddr string
	Dash    string

	done chan struct{}
	mu   sync.Mutex
	err  error
}

// LaunchOptions describes one launch of the desktop executable.
type LaunchOptions struct {
	Exe     string
	Profile Profile
	// Home is passed as --home. Empty launches without it, so the home is the
	// one the profile's bootstrap locator remembers (the relaunch contract).
	Home    string
	APIAddr string
	Dash    string
	LogPath string
}

// Args are the executable's arguments: `hachidori.exe desktop` is the same
// composition as a no-argument launch with the loopback addresses chosen, so a
// disposable scenario never takes the default ports.
func (o LaunchOptions) Args() []string {
	args := []string{"desktop", "--listen", o.APIAddr, "--addr", o.Dash}
	if o.Home != "" {
		args = append(args, "--home", o.Home)
	}
	return args
}

// Launch starts the real executable in the disposable profile and returns at
// once; its stdout and stderr go to LogPath. The process ends only by Quit,
// Kill or its own failure; callers must end it (a scenario cleanup does).
func Launch(o LaunchOptions) (*Exe, error) {
	if o.Exe == "" || o.APIAddr == "" || o.Dash == "" || o.LogPath == "" {
		return nil, fmt.Errorf("launch options are incomplete: %+v", o)
	}
	log, err := os.Create(o.LogPath)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(o.Exe, o.Args()...)
	cmd.Env = o.Profile.Env(os.LookupEnv)
	cmd.Dir = o.Profile.Dir
	cmd.Stdout, cmd.Stderr = log, log
	cmd.Stdin = nil
	if err := cmd.Start(); err != nil {
		_ = log.Close()
		return nil, err
	}
	e := &Exe{Cmd: cmd, LogPath: o.LogPath, APIAddr: o.APIAddr, Dash: o.Dash, done: make(chan struct{})}
	go func() {
		err := cmd.Wait()
		_ = log.Close()
		e.mu.Lock()
		e.err = err
		e.mu.Unlock()
		close(e.done)
	}()
	return e, nil
}

// PID is the process ID of the executable.
func (e *Exe) PID() int { return e.Cmd.Process.Pid }

// Exited reports whether the executable has ended.
func (e *Exe) Exited() bool {
	select {
	case <-e.done:
		return true
	default:
		return false
	}
}

// Wait waits for the executable to end, bounded, and returns its exit code.
func (e *Exe) Wait(timeout time.Duration) (int, error) {
	select {
	case <-e.done:
		e.mu.Lock()
		defer e.mu.Unlock()
		if e.err == nil {
			return 0, nil
		}
		if ee, ok := e.err.(*exec.ExitError); ok {
			return ee.ExitCode(), nil
		}
		return -1, e.err
	case <-time.After(timeout):
		return 0, fmt.Errorf("hachidori.exe (pid %d) did not exit within %s: %w", e.PID(), timeout, ErrDeadline)
	}
}

// LogTail returns the last max bytes of the executable's output.
func (e *Exe) LogTail(max int) []byte { return tailFile(e.LogPath, max) }

func tailFile(path string, max int) []byte {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil
	}
	off := int64(0)
	if st.Size() > int64(max) {
		off = st.Size() - int64(max)
	}
	b := make([]byte, st.Size()-off)
	if _, err := f.ReadAt(b, off); err != nil && err != io.EOF {
		return nil
	}
	return b
}

// TailFile is tailFile for scenarios that attach a log of the home.
func TailFile(path string, max int) []byte { return tailFile(path, max) }

// Terminate ends the executable if it still runs and everything it owns. It is
// the cleanup of last resort; a scenario that means to Quit does so first.
func (e *Exe) Terminate() {
	if e.Exited() {
		return
	}
	if runtime.GOOS == "windows" {
		_ = KillTree(e.PID())
	} else {
		_ = e.Cmd.Process.Kill()
	}
	select {
	case <-e.done:
	case <-time.After(15 * time.Second):
	}
}
