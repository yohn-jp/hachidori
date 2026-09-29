package desktop

import (
	"errors"
	"strings"
)

// RunKeyPath and StartupValueName identify the Windows startup mechanism.
//
// Hachidori starts at sign-in through one value in the current user's Run key:
//
//	HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run
//	  Hachidori = "C:\...\hachidori.exe" desktop --background [--home "..."]
//
// This is the standard per-user startup mechanism: it needs no elevation
// (HKCU only), is idempotent (one named value that is overwritten or
// deleted), is removed cleanly (deleting the value leaves nothing behind),
// starts the same executable in desktop/background mode, and creates no
// Windows Service, scheduled task or machine-wide entry.
const (
	RunKeyPath       = `Software\Microsoft\Windows\CurrentVersion\Run`
	StartupValueName = "Hachidori"
)

// Startup is the per-user start-at-sign-in mechanism. Native startup is
// registry based on Windows; tests use a fake.
type Startup interface {
	// Enabled reports whether the startup entry exists.
	Enabled() (bool, error)
	// Enable creates the entry running command, or updates it when it
	// differs (for example after the executable moved). Enabling an enabled
	// entry with the same command changes nothing.
	Enable(command string) error
	// Disable removes the entry. Disabling an absent entry succeeds.
	Disable() error
}

// StartupCommand builds the startup command line: the quoted executable, the
// desktop command in background mode, and --home when the home is not one the
// desktop rediscovers by itself (bootstrap locator).
func StartupCommand(exe, homeRoot string) (string, error) {
	if exe == "" {
		return "", errors.New("executable path is empty")
	}
	if strings.ContainsAny(exe+homeRoot, "\"\x00\r\n") {
		return "", errors.New("paths containing a double quote or control character cannot be used in a startup command")
	}
	parts := []string{quoteArg(exe), "desktop", "--background"}
	if homeRoot != "" {
		parts = append(parts, "--home", quoteArg(homeRoot))
	}
	return strings.Join(parts, " "), nil
}

// quoteArg quotes a Windows command-line argument that holds no double quote:
// always quoted, and trailing backslashes doubled so they cannot escape the
// closing quote.
func quoteArg(s string) string {
	n := len(s) - len(strings.TrimRight(s, `\`))
	return `"` + s + strings.Repeat(`\`, n) + `"`
}
