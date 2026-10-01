// Package redact is the one policy for text that may leave the operator's
// machine or be shared: the diagnostic bundle and the error text returned to
// API callers both go through it.
//
// It distinguishes three things:
//
//   - actual secrets (credentials in a URL, token-like query or key=value
//     pairs, bearer and cookie values) are always replaced;
//   - sensitive operational metadata (the HACHIDORI_HOME and user-profile
//     paths, in every spelling a Windows Python process produces) is replaced
//     by a placeholder, so the text still says where in the installation
//     something went wrong without naming the account;
//   - everything else (error classes, versions, ordinary words) is kept.
//
// It is deliberately small and deterministic. It cannot recognize an
// arbitrary secret in free text, and it is not applied to the operator's own
// local screens, which show real paths because the operator needs them.
package redact

import (
	"os"
	"regexp"
	"strings"
)

// Placeholders that replace what was removed.
const (
	HomePlaceholder    = "<HACHIDORI_HOME>"
	ProfilePlaceholder = "<USERPROFILE>"
	secret             = "<redacted>"
)

// Scrubber replaces local paths and secrets and bounds free text.
type Scrubber struct{ r *strings.Replacer }

// New returns a Scrubber for text produced under home (the HACHIDORI_HOME
// root; "" for none) by a process of the current user.
func New(home string) Scrubber {
	var pairs []string
	add := func(p, placeholder string) {
		if p = strings.TrimRight(p, `/\`); p == "" {
			return
		}
		// Native, slash and backslash-doubled spellings (the slash form is
		// derived the same way on every platform). Python quotes the
		// filename of an OSError with repr(), so a Windows path in a worker
		// failure message arrives as C:\\Users\\name\\....
		seen := map[string]bool{}
		for _, form := range []string{p, strings.ReplaceAll(p, `\`, `/`), strings.ReplaceAll(p, `\`, `\\`)} {
			if !seen[form] {
				seen[form] = true
				pairs = append(pairs, form, placeholder)
			}
		}
	}
	// The home first: strings.Replacer prefers the earlier pair, and the home
	// usually lies inside the user profile.
	add(home, HomePlaceholder)
	if uh, err := os.UserHomeDir(); err == nil {
		add(uh, ProfilePlaceholder)
	}
	return Scrubber{strings.NewReplacer(pairs...)}
}

var secretPatterns = []struct {
	re   *regexp.Regexp
	repl string
}{
	// user:password@ in any URL; the host stays.
	{regexp.MustCompile(`(?i)\b([a-z][a-z0-9+.-]*://)[^/\s'"<>@]*@`), "${1}" + secret + "@"},
	// token=..., api_key=..., HF_TOKEN=... in a query string or an environment
	// dump; the name must end in the secret word, so num_tokens=5 is kept.
	{regexp.MustCompile(`(?i)\b([\w.-]*(?:token|secret|passw(?:or)?d|credential|signature|api[_-]?key)=)[^&\s'"<>]+`), "${1}" + secret},
	// "Bearer abc...", "Basic abc..."
	{regexp.MustCompile(`(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}`), "${1} " + secret},
	// Authorization: <scheme> <value> and Cookie: <value>
	{regexp.MustCompile(`(?i)\b(authorization|cookie)(\s*[:=]\s*)(?:\w+\s+)?[^\s,;]+`), "${1}${2}" + secret},
}

// Line returns v as one line of at most max bytes (plus a truncation marker):
// paths and secrets replaced, line breaks turned into spaces, other control
// characters dropped, invalid UTF-8 replaced.
func (s Scrubber) Line(v string, max int) string {
	v = s.r.Replace(v)
	for _, p := range secretPatterns {
		v = p.re.ReplaceAllString(v, p.repl)
	}
	v = strings.Map(func(r rune) rune {
		switch {
		case r == '\n' || r == '\r':
			return ' '
		case r < 0x20 && r != '\t' || r == 0x7f:
			return -1
		}
		return r
	}, strings.ToValidUTF8(v, "?"))
	if len(v) > max {
		v = strings.ToValidUTF8(v[:max], "") + "...[truncated]"
	}
	return v
}
