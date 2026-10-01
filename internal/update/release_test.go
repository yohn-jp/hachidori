package update

import (
	"strings"
	"testing"
)

const realSum = "9e1487565b960d777acb336e9c6a3d235ab158475e2cf8664f592afe5b944dc9  hachidori-windows-amd64.exe\r\n"

func TestParseChecksumFollowsTheReleaseWorkflowFile(t *testing.T) {
	got, err := ParseChecksum([]byte(realSum), ExeAsset)
	if err != nil || got != "9e1487565b960d777acb336e9c6a3d235ab158475e2cf8664f592afe5b944dc9" {
		t.Fatalf("real checksum file: %q %v", got, err)
	}
	h := strings.Repeat("a", 64)
	for name, body := range map[string]string{
		"lf":          h + "  " + ExeAsset + "\n",
		"binary mode": strings.ToUpper(h) + " *" + ExeAsset,
	} {
		if _, err := ParseChecksum([]byte(body), ExeAsset); err != nil {
			t.Errorf("%s rejected: %v", name, err)
		}
	}
	for name, body := range map[string]string{
		"empty":           "",
		"bare digest":     h + "\n",
		"other asset":     h + "  other.exe\n",
		"prefix of name":  h + "  " + ExeAsset + ".bak\n",
		"short digest":    strings.Repeat("a", 63) + "  " + ExeAsset,
		"non hex":         strings.Repeat("g", 64) + "  " + ExeAsset,
		"two lines":       h + "  " + ExeAsset + "\n" + h + "  " + ExeAsset + "\n",
		"one space":       h + " " + ExeAsset,
		"path in name":    h + "  ./" + ExeAsset,
		"trailing spaces": h + "  " + ExeAsset + " ",
		"html":            "<html>rate limited</html>",
	} {
		if got, err := ParseChecksum([]byte(body), ExeAsset); err == nil {
			t.Errorf("%s accepted: %q", name, got)
		}
	}
}

func asset(tag, name string, size int64) string {
	return `{"name":"` + name + `","size":` + itoa(size) + `,"state":"uploaded","browser_download_url":"` + assetURL(tag, name) + `"}`
}

func itoa(n int64) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for ; n > 0; n /= 10 {
		b = append([]byte{byte('0' + n%10)}, b...)
	}
	return string(b)
}

func rel(tag string, prerelease, draft bool, assets ...string) string {
	return `{"tag_name":"` + tag + `","prerelease":` + map[bool]string{true: "true", false: "false"}[prerelease] +
		`,"draft":` + map[bool]string{true: "true", false: "false"}[draft] + `,"published_at":"2026-10-01T13:44:50Z","assets":[` + strings.Join(assets, ",") + `]}`
}

func good(tag string) []string {
	return []string{asset(tag, ExeAsset, 16299520), asset(tag, SumAsset, 95)}
}

// The list shape the shared workflow produces, including the repository's real
// quirks: legacy dev-N tags and a development release not flagged pre-release.
func repoList() string {
	return "[" + strings.Join([]string{
		rel("0.2.5-dev", true, false, good("0.2.5-dev")...),
		rel("0.2.1-dev", false, false, good("0.2.1-dev")...),
		rel("0.2.0", false, false, good("0.2.0")...),
		rel("0.2.6-dev", true, true, good("0.2.6-dev")...), // draft
		rel("0.3.0-rc.1", true, false, good("0.3.0-rc.1")...),
		rel("dev-24", true, false, good("dev-24")...),
		rel("0.2.10-dev", true, false, good("0.2.10-dev")...),
		rel("0.2.9", true, false, good("0.2.9")...), // stable tag flagged pre-release
	}, ",") + "]"
}

func tags(cs []Candidate) []string {
	var out []string
	for _, c := range cs {
		out = append(out, c.Tag)
	}
	return out
}

func TestChannelEligibilityIsDeterministic(t *testing.T) {
	rels, listed, err := parseReleases([]byte(repoList()))
	if err != nil || listed != 8 {
		t.Fatal(rels, listed, err)
	}
	if got := tags(candidates(rels, Stable, nil)); strings.Join(got, " ") != "0.2.0" {
		t.Errorf("stable: %v", got)
	}
	// development: stable and -dev, ordered by SemVer (0.2.10-dev above 0.2.5-dev), the
	// pre-release flag only withdraws a stable tag (0.2.9), and dev-24, rc, drafts are not offered.
	if got := tags(candidates(rels, Development, nil)); strings.Join(got, " ") != "0.2.10-dev 0.2.5-dev 0.2.1-dev 0.2.0" {
		t.Errorf("development: %v", got)
	}
	again, _, _ := parseReleases([]byte(repoList()))
	if strings.Join(tags(candidates(again, Development, nil)), " ") != strings.Join(tags(candidates(rels, Development, nil)), " ") {
		t.Error("eligibility is not deterministic")
	}
}

func TestRelationToInstalledNeverOffersADowngrade(t *testing.T) {
	rels, _, _ := parseReleases([]byte(repoList()))
	inst, _ := ParseVersion("0.2.5-dev")
	rel := map[string]string{}
	for _, c := range candidates(rels, Development, &inst) {
		rel[c.Tag] = c.Relation
		if c.Downloadable() != (c.Relation == RelNewer) {
			t.Errorf("%s: downloadable=%v for %s", c.Tag, c.Downloadable(), c.Relation)
		}
	}
	if rel["0.2.10-dev"] != RelNewer || rel["0.2.5-dev"] != RelCurrent || rel["0.2.1-dev"] != RelOlder || rel["0.2.0"] != RelOlder {
		t.Errorf("relations: %v", rel)
	}
	// Switching to Stable while a development build is installed: the only
	// eligible release is older, so nothing is downloadable.
	for _, c := range candidates(rels, Stable, &inst) {
		if c.Downloadable() {
			t.Errorf("stable %s downloadable over installed %s", c.Tag, inst)
		}
	}
	for _, c := range candidates(rels, Development, nil) {
		if c.Relation != RelUnknown {
			t.Errorf("%s: %s with unknown installed", c.Tag, c.Relation)
		}
	}
}

func TestAssetBindingRejectsMissingAmbiguousAndForeignAssets(t *testing.T) {
	tag := "0.2.7-dev"
	exe, sum := asset(tag, ExeAsset, 100), asset(tag, SumAsset, 95)
	for name, tc := range map[string]struct {
		assets []string
		want   string
	}{
		"ok":               {[]string{exe, sum}, ""},
		"no checksum":      {[]string{exe}, "asset not found: " + SumAsset},
		"no executable":    {[]string{sum}, "asset not found: " + ExeAsset},
		"duplicate exe":    {[]string{exe, exe, sum}, "ambiguous"},
		"duplicate sum":    {[]string{exe, sum, sum}, "ambiguous"},
		"wrong name":       {[]string{strings.Replace(exe, ExeAsset, "hachidori-linux-amd64", 1), sum}, "asset not found: " + ExeAsset},
		"other release":    {[]string{strings.Replace(exe, tag, "0.2.8-dev", 1), sum}, "official release location"},
		"other host":       {[]string{strings.Replace(exe, "https://github.com", "https://evil.example", 1), sum}, "official release location"},
		"not uploaded":     {[]string{strings.Replace(exe, "uploaded", "starter", 1), sum}, "not fully uploaded"},
		"zero size":        {[]string{strings.Replace(exe, `"size":100`, `"size":0`, 1), sum}, "unacceptable size"},
		"huge size":        {[]string{strings.Replace(exe, `"size":100`, `"size":99999999999`, 1), sum}, "unacceptable size"},
		"oversized sum":    {[]string{exe, strings.Replace(sum, `"size":95`, `"size":99999`, 1)}, "unacceptable size"},
		"malformed digest": {[]string{strings.Replace(exe, `"state"`, `"digest":"sha1:abc","state"`, 1), sum}, "digest"},
	} {
		rels, _, err := parseReleases([]byte("[" + rel(tag, true, false, tc.assets...) + "]"))
		if err != nil || len(rels) != 1 {
			t.Fatalf("%s: %v %v", name, rels, err)
		}
		if tc.want == "" && rels[0].Problem != "" || tc.want != "" && !strings.Contains(rels[0].Problem, tc.want) {
			t.Errorf("%s: problem %q, want %q", name, rels[0].Problem, tc.want)
		}
		if (rels[0].Problem != "") && (Candidate{Release: rels[0], Relation: RelNewer}).Downloadable() {
			t.Errorf("%s: a release with a problem is downloadable", name)
		}
	}
	// A digest GitHub reports is kept for cross-checking.
	d := strings.Repeat("ab", 32)
	rels, _, _ := parseReleases([]byte("[" + rel(tag, true, false, strings.Replace(exe, `"state"`, `"digest":"sha256:`+strings.ToUpper(d)+`","state"`, 1), sum) + "]"))
	if rels[0].Exe.SHA256 != d {
		t.Errorf("digest %q", rels[0].Exe.SHA256)
	}
}

func TestParseReleasesRejectsNonReleaseLists(t *testing.T) {
	for _, body := range []string{"", "{}", `{"message":"API rate limit exceeded"}`, "<html>", `["x"]`, `[{"tag_name":5}]`} {
		if _, _, err := parseReleases([]byte(body)); err == nil {
			t.Errorf("accepted %q", body)
		}
	}
	if rels, n, err := parseReleases([]byte("[]")); err != nil || len(rels) != 0 || n != 0 {
		t.Errorf("empty list: %v %d %v", rels, n, err)
	}
}

func TestChannelParsing(t *testing.T) {
	for _, c := range Channels {
		if got, err := ParseChannel(string(c)); err != nil || got != c {
			t.Errorf("%s: %v %v", c, got, err)
		}
	}
	for _, bad := range []string{"", "Stable", "beta", "nightly"} {
		if _, err := ParseChannel(bad); err == nil {
			t.Errorf("accepted %q", bad)
		}
	}
}
