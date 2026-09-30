package i18n

import (
	"strings"
	"testing"
)

func TestResolveIsDeterministic(t *testing.T) {
	for _, tc := range []struct {
		saved string
		host  []string
		want  Locale
	}{
		{"", nil, English},
		{"", []string{"ja_JP.UTF-8"}, Japanese},
		{"", []string{"ja-JP"}, Japanese},
		{"", []string{"fr_FR.UTF-8", "ja"}, Japanese},
		{"", []string{"fr_FR.UTF-8", "C.UTF-8"}, English},
		{"en", []string{"ja_JP.UTF-8"}, English}, // the explicit selection wins
		{"ja", []string{"en_US.UTF-8"}, Japanese},
		{"de", nil, English},
		{"JA", nil, English}, // saved values are exact
	} {
		if got := Resolve(tc.saved, tc.host...); got != tc.want {
			t.Errorf("Resolve(%q, %q) = %q, want %q", tc.saved, tc.host, got, tc.want)
		}
	}
}

func TestLookupFallsBackToEnglish(t *testing.T) {
	if got := Japanese.T("Runtime"); got != "ランタイム" {
		t.Errorf("ja Runtime = %q", got)
	}
	if got := English.T("Runtime"); got != "Runtime" {
		t.Errorf("en Runtime = %q", got)
	}
	const missing = "a message without a Japanese entry"
	if Japanese.Has(missing) || Japanese.T(missing) != missing {
		t.Error("missing Japanese entry does not fall back to English")
	}
	if got := Japanese.T("%d of %d slots in use.", 3, 4); got != "3 / 4 スロット使用中。" {
		t.Errorf("formatted = %q", got)
	}
	if got := English.T("%d of %d slots in use.", 3, 4); got != "3 of 4 slots in use." {
		t.Errorf("formatted = %q", got)
	}
	if got := Locale("xx").T("Runtime"); got != "Runtime" {
		t.Errorf("unsupported locale = %q", got)
	}
}

// Every Japanese entry keeps the English message's format verbs, so a
// translated message never changes what its arguments render as.
func TestJapaneseKeepsFormatVerbs(t *testing.T) {
	verbs := func(s string) string {
		var out []string
		for i := 0; i < len(s); i++ {
			if s[i] == '%' && i+1 < len(s) {
				j := i + 1
				for j < len(s) && strings.IndexByte(".0123456789", s[j]) >= 0 {
					j++
				}
				if j < len(s) {
					out = append(out, s[i:j+1])
				}
				i = j
			}
		}
		return strings.Join(out, " ")
	}
	for en, ja := range ja {
		if verbs(en) != verbs(ja) {
			t.Errorf("%q: verbs %q, Japanese %q", en, verbs(en), verbs(ja))
		}
	}
}

func TestTableCoversFirstRunScript(t *testing.T) {
	en, jp := English.Table(FirstRunScript), Japanese.Table(FirstRunScript)
	for _, k := range FirstRunScript {
		if en[k] != k {
			t.Errorf("en %q = %q", k, en[k])
		}
		if !Japanese.Has(k) || jp[k] == k {
			t.Errorf("first-run message %q has no Japanese entry", k)
		}
	}
}
