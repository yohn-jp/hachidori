// Package update is the manual, verified update of the Windows executable.
//
// Hachidori stays local-first: nothing in this package runs on its own. There
// is no timer, no startup check and no background refresh. Status, SetChannel
// and every other read are local. Only Service.Check retrieves release
// metadata and only Service.StartDownload retrieves release assets, each on an
// explicit operator action.
//
// The release authority is fixed in this file: the official yohn-jp/hachidori
// GitHub Releases. There is no configurable repository or URL. Tags, assets
// and the checksum file follow the shared Windows release workflow the
// repository's .github/workflows/release.yml calls:
//
//	tag     0.2.N-dev (development line, GitHub pre-release) or X.Y.Z (stable)
//	assets  hachidori-windows-amd64.exe
//	        hachidori-windows-amd64.exe.sha256   "<sha256-hex>  <asset name>"
//
// A downloaded executable becomes ready to install only when its SHA-256
// equals the digest in that checksum asset of the same release. The running
// executable is never overwritten in-process: replacement is a bounded helper
// (Apply) that runs after the application exited.
package update

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
)

// The fixed release authority and the release workflow's asset contract.
const (
	Owner = "yohn-jp"
	Repo  = "hachidori"

	// ExeAsset is the Windows amd64 executable of a release; SumAsset is its
	// checksum file.
	ExeAsset = "hachidori-windows-amd64.exe"
	SumAsset = ExeAsset + ".sha256"

	apiHost      = "api.github.com"
	downloadHost = "github.com"

	// Bounds on what release metadata may claim and the updater will accept.
	maxExeSize  = 512 << 20
	maxSumSize  = 1 << 10
	maxPageSize = 8 << 20
	perPage     = 100
	maxPages    = 5
	// MaxShown bounds the releases an explicit check presents.
	MaxShown = 10
)

// releasesURL is the one metadata endpoint, page n of the repository's
// releases. It is built here from constants; no response can name another.
func releasesURL(page int) string {
	return fmt.Sprintf("https://%s/repos/%s/%s/releases?per_page=%d&page=%d", apiHost, Owner, Repo, perPage, page)
}

// assetURL is the canonical download location of one asset of one release.
// Metadata is never trusted to name a download URL.
func assetURL(tag, name string) string {
	return fmt.Sprintf("https://%s/%s/%s/releases/download/%s/%s", downloadHost, Owner, Repo, tag, name)
}

// Channel is update-selection policy: which releases an explicit check offers.
// It is not a runtime or product identity.
type Channel string

const (
	// Stable offers stable releases (X.Y.Z, not flagged pre-release).
	Stable Channel = "stable"
	// Development offers development releases (X.Y.Z-dev) and stable releases.
	Development Channel = "development"
)

// Channels lists the selectable channels in display order.
var Channels = []Channel{Stable, Development}

// ParseChannel accepts exactly a supported channel name.
func ParseChannel(s string) (Channel, error) {
	for _, c := range Channels {
		if string(c) == s {
			return c, nil
		}
	}
	return "", fmt.Errorf("update channel %q is not supported (stable or development)", s)
}

// Eligible reports whether a release is offered on the channel. The tag decides
// what a release is; GitHub's pre-release flag can only withdraw a stable tag.
func (c Channel) Eligible(r Release) bool {
	stable := r.Version.IsStable() && !r.Prerelease
	switch c {
	case Stable:
		return stable
	case Development:
		return stable || r.Version.IsDevelopment()
	}
	return false
}

// Asset is one release asset the updater uses, with the location it may be
// downloaded from (always the canonical one).
type Asset struct {
	Name   string
	Size   int64
	URL    string
	SHA256 string // digest GitHub reports for the asset, when it reports one
}

// Release is one release that carries a well-formed version tag.
type Release struct {
	Tag        string
	Version    Version
	Prerelease bool // GitHub's pre-release flag
	Published  time.Time
	Exe, Sum   Asset
	// Problem is why this release cannot be downloaded: its Windows assets are
	// missing, ambiguous or inconsistent. Empty for an installable release.
	Problem string
}

// Relation of a release to the installed executable.
const (
	RelNewer   = "newer"
	RelCurrent = "current"
	RelOlder   = "older"
	// RelUnknown: the installed version is not known, so the release cannot be
	// ordered against it.
	RelUnknown = "unknown"
)

// Candidate is an eligible release as an explicit check presented it.
type Candidate struct {
	Release
	Relation string
}

// Downloadable reports whether the operator may download this candidate: it is
// installable and not older than, or equal to, what is installed. An update
// never goes backwards.
func (c Candidate) Downloadable() bool {
	return c.Problem == "" && (c.Relation == RelNewer || c.Relation == RelUnknown)
}

// rawRelease and rawAsset are the parts of GitHub's release JSON the updater
// reads. Everything else is ignored.
type rawRelease struct {
	Tag        string     `json:"tag_name"`
	Draft      bool       `json:"draft"`
	Prerelease bool       `json:"prerelease"`
	Published  *time.Time `json:"published_at"`
	Assets     []rawAsset `json:"assets"`
}

type rawAsset struct {
	Name   string `json:"name"`
	Size   int64  `json:"size"`
	State  string `json:"state"`
	URL    string `json:"browser_download_url"`
	Digest string `json:"digest"`
}

var hex64 = regexp.MustCompile(`^[0-9a-f]{64}$`)

// parseReleases decodes one page of GitHub's release list. Drafts, tags that
// are not versions and pre-release kinds the repository does not publish are
// not releases for the updater and are dropped; a body that is not a JSON array
// of releases is malformed. listed is how many entries the page held, so a
// caller can tell a full page from the last one.
func parseReleases(body []byte) (rels []Release, listed int, err error) {
	var raws []rawRelease
	if err := json.Unmarshal(body, &raws); err != nil {
		return nil, 0, fmt.Errorf("release list is not a JSON array of releases: %w", err)
	}
	var out []Release
	for _, rr := range raws {
		if rr.Draft {
			continue
		}
		v, err := ParseVersion(rr.Tag)
		if err != nil || !(v.IsStable() || v.IsDevelopment()) {
			continue // legacy dev-N tags, release candidates, anything else
		}
		r := Release{Tag: rr.Tag, Version: v, Prerelease: rr.Prerelease}
		if rr.Published != nil {
			r.Published = *rr.Published
		}
		r.Exe, r.Sum, r.Problem = bindAssets(rr)
		out = append(out, r)
	}
	return out, len(raws), nil
}

// bindAssets selects the release's Windows executable and checksum asset by
// exact name. Zero or several of either, an asset that is not fully uploaded,
// a size that is not plausible, a URL that is not the canonical download
// location of that exact asset of that exact tag, or a malformed digest make
// the release not installable.
func bindAssets(rr rawRelease) (exe, sum Asset, problem string) {
	find := func(name string, maxSize int64) (Asset, string) {
		var found []rawAsset
		for _, a := range rr.Assets {
			if a.Name == name {
				found = append(found, a)
			}
		}
		switch {
		case len(found) == 0:
			return Asset{}, "asset not found: " + name
		case len(found) > 1:
			return Asset{}, "asset ambiguous: " + name + " is listed more than once"
		}
		a := found[0]
		switch {
		case a.State != "uploaded":
			return Asset{}, "asset " + name + " is not fully uploaded"
		case a.Size <= 0 || a.Size > maxSize:
			return Asset{}, fmt.Sprintf("asset %s has an unacceptable size (%d bytes)", name, a.Size)
		case a.URL != "" && a.URL != assetURL(rr.Tag, name):
			return Asset{}, "asset " + name + " does not point at the official release location"
		}
		out := Asset{Name: name, Size: a.Size, URL: assetURL(rr.Tag, name)}
		if a.Digest != "" {
			d, ok := strings.CutPrefix(strings.ToLower(a.Digest), "sha256:")
			if !ok || !hex64.MatchString(d) {
				return Asset{}, "asset " + name + " reports a digest the updater cannot read"
			}
			out.SHA256 = d
		}
		return out, ""
	}
	exe, problem = find(ExeAsset, maxExeSize)
	if problem != "" {
		return Asset{}, Asset{}, problem
	}
	sum, problem = find(SumAsset, maxSumSize)
	if problem != "" {
		return Asset{}, Asset{}, problem
	}
	return exe, sum, ""
}

// candidates is the channel's eligible releases, newest first, each related to
// the installed version (zero Installed: unknown).
func candidates(rs []Release, c Channel, installed *Version) []Candidate {
	var out []Candidate
	for _, r := range rs {
		if !c.Eligible(r) {
			continue
		}
		rel := RelUnknown
		if installed != nil {
			switch r.Version.Compare(*installed) {
			case 1:
				rel = RelNewer
			case 0:
				rel = RelCurrent
			default:
				rel = RelOlder
			}
		}
		out = append(out, Candidate{Release: r, Relation: rel})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Version.Compare(out[j].Version) > 0 })
	return out
}

// ParseChecksum reads the release workflow's checksum asset for the named
// executable: exactly one line "<64 hex>  <name>" (sha256sum text or binary
// mode, LF or CRLF). Missing or different asset name, a malformed digest,
// extra lines or any other shape is rejected: the file must bind one digest to
// exactly this asset.
func ParseChecksum(data []byte, asset string) (string, error) {
	text := strings.TrimRight(string(data), "\r\n")
	if text == "" || strings.ContainsAny(text, "\r\n") {
		return "", errors.New("checksum file must contain exactly one line")
	}
	sum, rest, ok := strings.Cut(text, " ")
	if !ok || !hex64.MatchString(strings.ToLower(sum)) {
		return "", errors.New("checksum file does not start with a SHA-256 digest")
	}
	name, ok := strings.CutPrefix(rest, " ") // text mode: two spaces
	if !ok {
		name, ok = strings.CutPrefix(rest, "*") // binary mode: " *"
	}
	if !ok || name != asset {
		return "", fmt.Errorf("checksum file names %q, not %s", strings.TrimSpace(rest), asset)
	}
	if _, err := hex.DecodeString(sum); err != nil {
		return "", errors.New("checksum file digest is not hexadecimal")
	}
	return strings.ToLower(sum), nil
}
