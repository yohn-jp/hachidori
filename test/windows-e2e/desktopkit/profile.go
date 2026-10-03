package desktopkit

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Profile is a disposable Windows user profile. The candidate only ever sees
// these directories, so bootstrap.json and the desktop preferences it writes
// never land in a real profile.
type Profile struct {
	Root         string
	LocalAppData string
	AppData      string
	UserProfile  string
	Temp         string
}

// NewProfile creates the profile directories under root.
func NewProfile(root string) (Profile, error) {
	p := Profile{
		Root:         root,
		LocalAppData: filepath.Join(root, "profile", "AppData", "Local"),
		AppData:      filepath.Join(root, "profile", "AppData", "Roaming"),
		UserProfile:  filepath.Join(root, "profile"),
		Temp:         filepath.Join(root, "profile", "AppData", "Local", "Temp"),
	}
	for _, d := range []string{p.LocalAppData, p.AppData, p.UserProfile, p.Temp} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return Profile{}, fmt.Errorf("create disposable profile: %w", err)
		}
	}
	return p, nil
}

// LocatorPath is where the candidate keeps the bootstrap locator inside this
// profile: %LOCALAPPDATA%\Hachidori\bootstrap.json.
func (p Profile) LocatorPath() string {
	return filepath.Join(p.LocalAppData, "Hachidori", "bootstrap.json")
}

// LocatorDir is the directory that holds the locator and the desktop
// preferences beside it.
func (p Profile) LocatorDir() string { return filepath.Dir(p.LocatorPath()) }

// overridden are the variables Env replaces. Names compare case-insensitively
// because Windows environment names do.
var overridden = []string{"LOCALAPPDATA", "APPDATA", "USERPROFILE", "TEMP", "TMP", "HOME"}

// Env returns the environment of a child process confined to the profile. base
// is normally os.Environ(). Every HACHIDORI_* variable is dropped, so a real
// HACHIDORI_HOME or the certification variables can never reach the candidate;
// extra entries (KEY=value) are applied last and win.
func (p Profile) Env(base []string, extra ...string) []string {
	var out []string
	for _, kv := range base {
		k, _, _ := strings.Cut(kv, "=")
		if strings.HasPrefix(strings.ToUpper(k), "HACHIDORI_") || isOverridden(k) {
			continue
		}
		out = append(out, kv)
	}
	out = append(out,
		"LOCALAPPDATA="+p.LocalAppData,
		"APPDATA="+p.AppData,
		"USERPROFILE="+p.UserProfile,
		"HOME="+p.UserProfile,
		"TEMP="+p.Temp,
		"TMP="+p.Temp,
	)
	for _, kv := range extra {
		k, _, _ := strings.Cut(kv, "=")
		out = dropKey(out, k)
		out = append(out, kv)
	}
	return out
}

func isOverridden(k string) bool {
	for _, o := range overridden {
		if strings.EqualFold(k, o) {
			return true
		}
	}
	return false
}

func dropKey(env []string, key string) []string {
	out := env[:0:0]
	for _, kv := range env {
		k, _, _ := strings.Cut(kv, "=")
		if !strings.EqualFold(k, key) {
			out = append(out, kv)
		}
	}
	return out
}
