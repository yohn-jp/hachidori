package desktopkit

import (
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// ProcEntry is one process of a snapshot.
type ProcEntry struct {
	PID  int
	PPID int
	Name string
}

// Named returns the entries whose image name equals name (case-insensitive).
func Named(all []ProcEntry, name string) []ProcEntry {
	var out []ProcEntry
	for _, e := range all {
		if strings.EqualFold(e.Name, name) {
			out = append(out, e)
		}
	}
	return out
}

// ChildrenNamed returns the entries named name whose parent is ppid.
func ChildrenNamed(all []ProcEntry, ppid int, name string) []ProcEntry {
	var out []ProcEntry
	for _, e := range Named(all, name) {
		if e.PPID == ppid {
			out = append(out, e)
		}
	}
	return out
}

// PIDs lists the pids of entries.
func PIDs(es []ProcEntry) []int {
	var out []int
	for _, e := range es {
		out = append(out, e.PID)
	}
	return out
}

// Listener is one listening TCP socket and its owning process.
type Listener struct {
	Host string
	Port int
	PID  int
}

// ParseNetstat reads the output of `netstat -ano -p TCP` and returns its
// LISTENING sockets. Lines that are not listeners are ignored.
func ParseNetstat(out string) []Listener {
	var ls []Listener
	for _, line := range strings.Split(out, "\n") {
		f := strings.Fields(line)
		if len(f) < 5 || !strings.EqualFold(f[0], "TCP") || !strings.EqualFold(f[3], "LISTENING") {
			continue
		}
		i := strings.LastIndex(f[1], ":")
		if i < 0 {
			continue
		}
		port, err := strconv.Atoi(f[1][i+1:])
		if err != nil {
			continue
		}
		pid, err := strconv.Atoi(f[4])
		if err != nil {
			continue
		}
		ls = append(ls, Listener{Host: strings.Trim(f[1][:i], "[]"), Port: port, PID: pid})
	}
	return ls
}

// Listeners runs netstat (Windows) and returns the listening TCP sockets.
func Listeners() ([]Listener, error) {
	if runtime.GOOS != "windows" {
		return nil, errNotWindows
	}
	out, err := exec.Command("netstat", "-ano", "-p", "TCP").Output()
	if err != nil {
		return nil, err
	}
	return ParseNetstat(string(out)), nil
}

// OnPort returns the listeners bound to port.
func OnPort(ls []Listener, port int) []Listener {
	var out []Listener
	for _, l := range ls {
		if l.Port == port {
			out = append(out, l)
		}
	}
	return out
}
