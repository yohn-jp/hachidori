package redact

import (
	"strings"
	"testing"
)

// Every fixture below is synthetic; none resembles a real credential.
func TestPathsAreReplacedInEverySpelling(t *testing.T) {
	const homeDir, profile = `C:\Users\alice\Hachidori`, `C:\Users\alice`
	t.Setenv("HOME", profile) // os.UserHomeDir on every platform under test
	t.Setenv("USERPROFILE", profile)
	s := New(homeDir)
	cases := []struct{ name, in, want string }{
		{"native", `model under C:\Users\alice\Hachidori\models\m`, `model under <HACHIDORI_HOME>\models\m`},
		{"slashes", `model under C:/Users/alice/Hachidori/models/m`, `model under <HACHIDORI_HOME>/models/m`},
		{"python repr", `OSError: [Errno 2] No such file or directory: 'C:\\Users\\alice\\Hachidori\\models\\m'`,
			`OSError: [Errno 2] No such file or directory: '<HACHIDORI_HOME>\\models\\m'`},
		{"profile outside home, repr", `cache 'C:\\Users\\alice\\AppData\\Local\\x'`, `cache '<USERPROFILE>\\AppData\\Local\\x'`},
		{"unrelated paths kept", `D:\\data\\x and C:\\Users\\bob\\x`, `D:\\data\\x and C:\\Users\\bob\\x`},
	}
	for _, c := range cases {
		if got := s.Line(c.in, 512); got != c.want {
			t.Errorf("%s:\n got %q\nwant %q", c.name, got, c.want)
		}
	}

	t.Setenv("HOME", "/home/bob")
	t.Setenv("USERPROFILE", "/home/bob")
	u := New("/home/bob/.hachidori")
	if got, want := u.Line("open /home/bob/.hachidori/models/m: denied; cwd /home/bob/work", 512),
		"open <HACHIDORI_HOME>/models/m: denied; cwd <USERPROFILE>/work"; got != want {
		t.Errorf("unix:\n got %q\nwant %q", got, want)
	}
}

func TestSecretsAreReplacedAndTheRestKept(t *testing.T) {
	s := New("")
	cases := []struct{ name, in, want string }{
		{"url userinfo", `GET https://user:pass-example@models.example.invalid/m.bin failed`, `GET https://<redacted>@models.example.invalid/m.bin failed`},
		{"url user only", `fetch ssh://deploy@host.example.invalid/x`, `fetch ssh://<redacted>@host.example.invalid/x`},
		{"query token", `for url: https://hub.example.invalid/api?token=not-a-real-token&rev=main`, `for url: https://hub.example.invalid/api?token=<redacted>&rev=main`},
		{"query api key", `/x?api_key=EXAMPLE&q=1`, `/x?api_key=<redacted>&q=1`},
		{"env dump", `HF_TOKEN=not-a-real-token PATH=/bin`, `HF_TOKEN=<redacted> PATH=/bin`},
		{"bearer", `401: Authorization: Bearer example-opaque-value`, `401: Authorization: <redacted>`},
		{"bare bearer", `sent Bearer example-opaque-value to the hub`, `sent Bearer <redacted> to the hub`},
		{"cookie", `Cookie: session=example-session-value`, `Cookie: <redacted>`},
		{"bare hugging face token", `401 for hf_AbCdEf0123456789fakefake on the hub`, `401 for <redacted> on the hub`},
		{"bare github tokens", `ghp_0123456789abcdefFAKEFAKE and github_pat_11ABCDEFG0123456789_fakefakefake`, `<redacted> and <redacted>`},
		{"names that merely start like a token are kept", `hf_hub_download ghost_writer github_page`, `hf_hub_download ghost_writer github_page`},
		{"plain error kept", `RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB`, `RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB`},
		{"url without credentials kept", `see https://example.invalid/docs?page=2`, `see https://example.invalid/docs?page=2`},
		{"author is not auth", `author=bob`, `author=bob`},
		{"counters named tokens are kept", `num_tokens=5 max_token_length=64 token_type=bearer-like`, `num_tokens=5 max_token_length=64 token_type=bearer-like`},
	}
	for _, c := range cases {
		if got := s.Line(c.in, 512); got != c.want {
			t.Errorf("%s:\n got %q\nwant %q", c.name, got, c.want)
		}
	}
}

func TestLineIsOneBoundedLine(t *testing.T) {
	s := New("")
	if got := s.Line("a\r\nb\nc\x00\x1bd\te\xff", 512); got != "a  b cd\te?" {
		t.Errorf("control characters: %q", got)
	}
	long := strings.Repeat("é", 400) // 800 bytes
	got := s.Line(long, 101)
	if !strings.HasSuffix(got, "...[truncated]") || len(got) > 101+len("...[truncated]") || strings.ContainsRune(got, '\uFFFD') {
		t.Errorf("truncation: %d bytes %q", len(got), got[len(got)-20:])
	}
}
