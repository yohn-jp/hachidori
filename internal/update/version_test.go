package update

import "testing"

func TestParseVersionAcceptsOnlyReleaseTags(t *testing.T) {
	for _, tag := range []string{"0.2.0", "0.2.5-dev", "10.20.30", "1.0.0-rc.1", "1.0.0-alpha.beta"} {
		v, err := ParseVersion(tag)
		if err != nil || v.String() != tag {
			t.Errorf("ParseVersion(%q) = %v, %v", tag, v, err)
		}
	}
	for _, tag := range []string{"", "dev-24", "v1.0.0", "1.0", "01.0.0", "1.0.0+build", "1.0.0-01", "1.0.0-", "1..0", "0.2.5-dev/../x",
		"99999999999999999999.0.0", "1.0.0-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"} {
		if v, err := ParseVersion(tag); err == nil {
			t.Errorf("ParseVersion(%q) accepted: %v", tag, v)
		}
	}
}

// The SemVer 2.0.0 precedence example chain, plus the repository's own tags.
func TestCompareIsSemVerPrecedenceNotLexical(t *testing.T) {
	for _, c := range [][]string{
		{"1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"},
		{"0.2.0-dev", "0.2.0", "0.2.1-dev", "0.2.2-dev", "0.2.9-dev", "0.2.10-dev", "0.2.10", "0.10.0", "1.0.0"},
	} {
		for i := 0; i+1 < len(c); i++ {
			a, _ := ParseVersion(c[i])
			b, _ := ParseVersion(c[i+1])
			if a.Compare(b) != -1 || b.Compare(a) != 1 || a.Compare(a) != 0 {
				t.Errorf("%s vs %s: %d %d", c[i], c[i+1], a.Compare(b), b.Compare(a))
			}
		}
	}
	a, _ := ParseVersion("0.2.9-dev")
	b, _ := ParseVersion("0.2.10-dev")
	if !("0.2.9-dev" > "0.2.10-dev") || a.Compare(b) != -1 {
		t.Error("0.2.9-dev must sort below 0.2.10-dev although it is lexically larger")
	}
}

func TestVersionKinds(t *testing.T) {
	for tag, kind := range map[string]string{"0.2.0": "stable", "0.2.5-dev": "dev", "0.3.0-rc.1": "other", "0.3.0-dev.1": "other"} {
		v, _ := ParseVersion(tag)
		got := "other"
		switch {
		case v.IsStable():
			got = "stable"
		case v.IsDevelopment():
			got = "dev"
		}
		if got != kind {
			t.Errorf("%s: %s, want %s", tag, got, kind)
		}
	}
}
