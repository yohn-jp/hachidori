package harness

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

// Process is one entry of the host process table.
type Process struct {
	PID     int
	PPID    int
	Command string // the full command line
}

// workerScript is the file name of the worker script every worker of this build
// runs; with the home it identifies a worker the runtime of that home owns.
const workerScript = "hachidori_worker.py"

func normPath(s string) string {
	s = filepath.Clean(s)
	if runtime.GOOS == "windows" {
		return strings.ToLower(s)
	}
	return s
}

// WorkerProcesses selects, from the process table, the processes that run this
// build's worker script out of the home root: the worker the Controller of that
// home owns. A home is disposable and unique per scenario, so no other process
// can match; it also matches the launcher of a Windows virtual environment and
// the interpreter it starts, which carry the same arguments.
func WorkerProcesses(all []Process, root string) []Process {
	root = normPath(root)
	var out []Process
	for _, p := range all {
		cmd := p.Command
		if runtime.GOOS == "windows" {
			cmd = strings.ToLower(cmd)
		}
		if strings.Contains(cmd, workerScript) && strings.Contains(cmd, root) {
			out = append(out, p)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].PID < out[j].PID })
	return out
}

// Roots returns the processes of set whose parent is not itself in set: one per
// worker, however many processes a launcher puts between the supervisor and
// the interpreter that really runs the script.
func Roots(set []Process) []Process {
	in := map[int]bool{}
	for _, p := range set {
		in[p.PID] = true
	}
	var out []Process
	for _, p := range set {
		if !in[p.PPID] {
			out = append(out, p)
		}
	}
	return out
}

// PIDs lists the process IDs of set.
func PIDs(set []Process) []int {
	var out []int
	for _, p := range set {
		out = append(out, p.PID)
	}
	return out
}

// ParseProcessJSON reads the PowerShell Get-CimInstance Win32_Process output
// (ConvertTo-Json of ProcessId, ParentProcessId, CommandLine): an array, or a
// bare object when there is exactly one process.
func ParseProcessJSON(b []byte) ([]Process, error) {
	b = []byte(strings.TrimSpace(strings.TrimPrefix(string(b), "\xef\xbb\xbf")))
	if len(b) == 0 {
		return nil, nil
	}
	if b[0] == '{' {
		b = append(append([]byte{'['}, b...), ']')
	}
	var rows []struct {
		ProcessID       int     `json:"ProcessId"`
		ParentProcessID int     `json:"ParentProcessId"`
		CommandLine     *string `json:"CommandLine"`
	}
	if err := json.Unmarshal(b, &rows); err != nil {
		return nil, fmt.Errorf("process table: %w", err)
	}
	out := make([]Process, 0, len(rows))
	for _, r := range rows {
		p := Process{PID: r.ProcessID, PPID: r.ParentProcessID}
		if r.CommandLine != nil {
			p.Command = *r.CommandLine
		}
		out = append(out, p)
	}
	return out, nil
}

// ParsePS reads `ps -axo pid=,ppid=,args=` output.
func ParsePS(b []byte) []Process {
	var out []Process
	for _, line := range strings.Split(string(b), "\n") {
		f := strings.Fields(line)
		if len(f) < 2 {
			continue
		}
		var p Process
		if _, err := fmt.Sscanf(f[0]+" "+f[1], "%d %d", &p.PID, &p.PPID); err != nil {
			continue
		}
		if i := strings.Index(line, f[1]); i >= 0 {
			p.Command = strings.TrimSpace(line[i+len(f[1]):])
		}
		out = append(out, p)
	}
	return out
}

// Descendants returns pid's descendants, deepest first, for ending a tree.
func Descendants(all []Process, pid int) []int {
	kids := map[int][]int{}
	for _, p := range all {
		kids[p.PPID] = append(kids[p.PPID], p.PID)
	}
	var out []int
	var walk func(int)
	walk = func(id int) {
		for _, k := range kids[id] {
			walk(k)
			out = append(out, k)
		}
	}
	walk(pid)
	return out
}
