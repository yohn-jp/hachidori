// Package stubapp lets the update E2E's own test binary play two small roles a
// scenario needs, selected by its first argument (never by a go-test flag):
//
//   - "desktop --home <home>": the updated application. The replacement helper
//     restarts the executable it replaced with exactly these arguments, so a
//     copy of the test binary standing in as the new release proves the same
//     path was reopened, by which bytes, and with which home. It writes a
//     marker file beside itself (never inside the home) and exits.
//   - "e2e-hold": the running application the helper waits for. It does
//     nothing until the scenario ends it, as Quit does in production.
//
// Neither role touches the network or any Hachidori home.
package stubapp

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"os"
	"strings"
	"time"
)

// Roles, as the first command-line argument.
const (
	RoleReopen = "desktop"
	RoleHold   = "e2e-hold"
)

// holdLimit bounds how long a forgotten stand-in application can live.
const holdLimit = 10 * time.Minute

// Marker is what the reopened executable records about itself.
type Marker struct {
	Schema string   `json:"schema"`
	Exe    string   `json:"exe"`
	SHA256 string   `json:"sha256"`
	Home   string   `json:"home"`
	Args   []string `json:"args"`
}

// MarkerSchema identifies the marker document.
const MarkerSchema = "hachidori.windows-e2e.reopened/v1"

// MarkerPath is where the executable at exe records that it was reopened.
func MarkerPath(exe string) string { return exe + ".reopened.json" }

// Run handles the stand-in roles for the current process. It reports false when
// args do not select a role, so the caller continues as an ordinary test binary.
func Run(args []string) (int, bool) {
	exe, err := os.Executable()
	if err != nil {
		return 2, len(args) > 0 && isRole(args[0])
	}
	return RunAs(exe, args)
}

func isRole(s string) bool { return s == RoleReopen || s == RoleHold }

// RunAs is Run for an explicit executable path (tests use it).
func RunAs(exe string, args []string) (int, bool) {
	if len(args) == 0 || !isRole(args[0]) {
		return 0, false
	}
	switch args[0] {
	case RoleHold:
		time.Sleep(holdLimit)
		return 0, true
	default:
		return reopen(exe, args), true
	}
}

func reopen(exe string, args []string) int {
	fs := flag.NewFlagSet(RoleReopen, flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	home := fs.String("home", "", "")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	sum, err := fileSHA256(exe)
	if err != nil {
		return 1
	}
	m := Marker{Schema: MarkerSchema, Exe: exe, SHA256: sum, Home: *home, Args: args}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return 1
	}
	// Written to a temporary name first so a reader never sees half a marker.
	tmp := MarkerPath(exe) + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o644); err != nil {
		return 1
	}
	if err := os.Rename(tmp, MarkerPath(exe)); err != nil {
		return 1
	}
	return 0
}

// ReadMarker reads the marker of the executable at exe; a missing marker is
// os.ErrNotExist.
func ReadMarker(exe string) (Marker, error) {
	data, err := os.ReadFile(MarkerPath(exe))
	if err != nil {
		return Marker{}, err
	}
	var m Marker
	if err := json.Unmarshal(data, &m); err != nil {
		return Marker{}, err
	}
	if m.Schema != MarkerSchema {
		return Marker{}, errors.New("unknown marker schema " + m.Schema)
	}
	return m, nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// SelectsRole reports whether args select a stand-in role (for tests).
func SelectsRole(args []string) bool {
	return len(args) > 0 && isRole(strings.TrimSpace(args[0]))
}
