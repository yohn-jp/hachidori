package update

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// Version is a Semantic Versioning 2.0.0 version as the repository's release
// workflow tags it: MAJOR.MINOR.PATCH with an optional pre-release part and no
// "v" prefix and no build metadata (stable "0.2.0", development "0.2.5-dev").
// Tags that are not exactly this shape (the legacy "dev-24" tags, "v1.0.0",
// "1.0.0+build") are not releases the updater knows how to order.
type Version struct {
	Major, Minor, Patch uint64
	Pre                 []string // dot-separated pre-release identifiers
}

var versionRe = regexp.MustCompile(`^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$`)

// maxTag bounds a tag the updater is willing to parse (and to use in a path).
const maxTag = 64

// ParseVersion parses a release tag. It is strict: anything outside the
// repository's tag shape is an error.
func ParseVersion(tag string) (Version, error) {
	if len(tag) == 0 || len(tag) > maxTag {
		return Version{}, fmt.Errorf("tag %q is not a release version", tag)
	}
	m := versionRe.FindStringSubmatch(tag)
	if m == nil {
		return Version{}, fmt.Errorf("tag %q is not a release version (MAJOR.MINOR.PATCH[-pre-release])", tag)
	}
	var v Version
	for i, dst := range []*uint64{&v.Major, &v.Minor, &v.Patch} {
		n, err := strconv.ParseUint(m[i+1], 10, 64)
		if err != nil {
			return Version{}, fmt.Errorf("tag %q: %w", tag, err)
		}
		*dst = n
	}
	if m[4] != "" {
		v.Pre = strings.Split(m[4], ".")
		for _, id := range v.Pre {
			if isNumeric(id) && len(id) > 1 && id[0] == '0' {
				return Version{}, fmt.Errorf("tag %q: numeric pre-release identifier %q has a leading zero", tag, id)
			}
		}
	}
	return v, nil
}

func isNumeric(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return s != ""
}

// String is the canonical tag form.
func (v Version) String() string {
	s := fmt.Sprintf("%d.%d.%d", v.Major, v.Minor, v.Patch)
	if len(v.Pre) > 0 {
		s += "-" + strings.Join(v.Pre, ".")
	}
	return s
}

// IsStable reports a version without a pre-release part.
func (v Version) IsStable() bool { return len(v.Pre) == 0 }

// IsDevelopment reports the repository's development tag: pre-release exactly
// "dev" (0.2.N-dev, published by the shared development-release workflow).
func (v Version) IsDevelopment() bool { return len(v.Pre) == 1 && v.Pre[0] == "dev" }

// Compare orders versions by SemVer precedence: -1, 0 or 1. Numeric parts and
// numeric pre-release identifiers compare as numbers (0.2.10-dev is newer than
// 0.2.9-dev), a pre-release is older than its release (0.2.5-dev < 0.2.5), and
// a longer identifier list is newer when the shared prefix is equal.
func (v Version) Compare(o Version) int {
	for _, p := range [][2]uint64{{v.Major, o.Major}, {v.Minor, o.Minor}, {v.Patch, o.Patch}} {
		if c := cmpUint(p[0], p[1]); c != 0 {
			return c
		}
	}
	switch {
	case len(v.Pre) == 0 && len(o.Pre) == 0:
		return 0
	case len(v.Pre) == 0:
		return 1
	case len(o.Pre) == 0:
		return -1
	}
	for i := 0; i < len(v.Pre) && i < len(o.Pre); i++ {
		if c := cmpIdent(v.Pre[i], o.Pre[i]); c != 0 {
			return c
		}
	}
	return cmpUint(uint64(len(v.Pre)), uint64(len(o.Pre)))
}

func cmpUint(a, b uint64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

// cmpIdent orders pre-release identifiers: numeric ones as numbers and below
// alphanumeric ones, alphanumeric ones in ASCII order.
func cmpIdent(a, b string) int {
	an, bn := isNumeric(a), isNumeric(b)
	switch {
	case an && bn:
		// No leading zeros, so a longer digit string is the larger number.
		if c := cmpUint(uint64(len(a)), uint64(len(b))); c != 0 {
			return c
		}
		return strings.Compare(a, b)
	case an:
		return -1
	case bn:
		return 1
	}
	return strings.Compare(a, b)
}
