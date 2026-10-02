// Package question implements caller-side, versioned Question Definitions.
//
// A Definition is a repository-owned development artifact: a stable question
// id, an explicit version, and the content of one v1 choice question. It
// compiles deterministically to the existing api.Question and carries a
// canonical content digest. Definitions, references to them and the files
// they live in never reach the HTTP server; only the compiled api.Question
// does.
package question

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/yohn-jp/hachidori/internal/api"
)

// Schema identifies the checked-in Question Definition file format.
const Schema = "hachidori.question.v1"

// Definition is one versioned question. The file form is a single JSON
// object with exactly these fields; unknown fields are rejected.
type Definition struct {
	Schema       string            `json:"schema"`
	ID           string            `json:"id"`
	Version      int               `json:"version"`
	Type         string            `json:"type"`
	Instructions string            `json:"instructions"`
	Choices      []string          `json:"choices"`
	Descriptions map[string]string `json:"descriptions,omitempty"`
}

// Identity names the exact definition used: id, explicit version and the
// canonical content digest.
type Identity struct {
	ID      string `json:"id"`
	Version int    `json:"version"`
	Digest  string `json:"digest"`
}

func (i Identity) String() string { return fmt.Sprintf("%s@%d", i.ID, i.Version) }

// Validate checks the definition with the same rules and limits the v1 API
// applies to a question, plus an explicit version >= 1.
func (d Definition) Validate() error {
	if d.Schema != Schema {
		return fmt.Errorf("schema must be %q, got %q", Schema, d.Schema)
	}
	if strings.TrimSpace(d.ID) == "" {
		return fmt.Errorf("id is required")
	}
	if d.Version < 1 {
		return fmt.Errorf("question %q: version must be an integer >= 1", d.ID)
	}
	return d.Compile().Validate()
}

// Compile projects the definition to the v1 transport question. Choice order
// is preserved; an empty descriptions object compiles to none.
func (d Definition) Compile() api.Question {
	q := api.Question{ID: d.ID, Type: d.Type, Instructions: d.Instructions, Choices: append([]string(nil), d.Choices...)}
	if len(d.Descriptions) > 0 {
		q.Descriptions = make(map[string]string, len(d.Descriptions))
		for k, v := range d.Descriptions {
			q.Descriptions[k] = v
		}
	}
	return q
}

// Canonical returns the JSON encoding the digest is computed over: fixed
// field order, descriptions with sorted keys (encoding/json sorts map keys),
// empty descriptions omitted, no insignificant whitespace. Source path, file
// formatting and field order in the file do not participate.
func (d Definition) Canonical() []byte {
	q := d.Compile()
	b, err := json.Marshal(Definition{Schema: Schema, ID: d.ID, Version: d.Version, Type: q.Type,
		Instructions: q.Instructions, Choices: q.Choices, Descriptions: q.Descriptions})
	if err != nil {
		panic(err) // only strings, ints and string maps: cannot fail
	}
	return b
}

// Digest is "sha256:" + hex SHA-256 of Canonical.
func (d Definition) Digest() string {
	s := sha256.Sum256(d.Canonical())
	return "sha256:" + hex.EncodeToString(s[:])
}

// Identity returns id, version and digest.
func (d Definition) Identity() Identity {
	return Identity{ID: d.ID, Version: d.Version, Digest: d.Digest()}
}

// Parse decodes and validates one definition document.
func Parse(data []byte) (Definition, error) {
	var d Definition
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&d); err != nil {
		return Definition{}, err
	}
	// More is false before a stray closing bracket or brace too; only the
	// end of the input ends the document.
	if _, err := dec.Token(); err != io.EOF {
		return Definition{}, fmt.Errorf("trailing data after definition")
	}
	if err := d.Validate(); err != nil {
		return Definition{}, err
	}
	return d, nil
}

// Ref references a definition by id and version. Digest, when set, pins the
// exact content and must match the resolved definition.
type Ref struct {
	ID      string `json:"id"`
	Version int    `json:"version"`
	Digest  string `json:"digest,omitempty"`
}

func (r Ref) String() string { return fmt.Sprintf("%s@%d", r.ID, r.Version) }

type key struct {
	id      string
	version int
}

// Set is a collection of definitions unique by (id, version).
type Set struct {
	defs map[key]Definition
	src  map[key]string
}

// Add inserts d; source is used only in error messages.
func (s *Set) Add(d Definition, source string) error {
	if s.defs == nil {
		s.defs, s.src = map[key]Definition{}, map[key]string{}
	}
	k := key{d.ID, d.Version}
	if prev, ok := s.src[k]; ok {
		return fmt.Errorf("%s: duplicate definition %s@%d (also in %s)", source, d.ID, d.Version, prev)
	}
	s.defs[k], s.src[k] = d, source
	return nil
}

// Len is the number of definitions.
func (s *Set) Len() int { return len(s.defs) }

// Definitions returns every definition sorted by id, then version.
func (s *Set) Definitions() []Definition {
	out := make([]Definition, 0, len(s.defs))
	for _, d := range s.defs {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].ID != out[j].ID {
			return out[i].ID < out[j].ID
		}
		return out[i].Version < out[j].Version
	})
	return out
}

// Resolve finds the definition for r and checks an optional digest pin.
func (s *Set) Resolve(r Ref) (Definition, error) {
	if s == nil || s.defs == nil {
		return Definition{}, fmt.Errorf("unresolved question definition %s: no definitions loaded", r)
	}
	d, ok := s.defs[key{r.ID, r.Version}]
	if !ok {
		return Definition{}, fmt.Errorf("unresolved question definition %s", r)
	}
	if r.Digest != "" && r.Digest != d.Digest() {
		return Definition{}, fmt.Errorf("question definition %s: digest %s does not match pinned %s", r, d.Digest(), r.Digest)
	}
	return d, nil
}

// Load reads definitions from files and directories (non-recursive, *.json)
// into one Set. Duplicate (id, version) pairs fail.
func Load(paths ...string) (*Set, error) {
	s := &Set{}
	for _, p := range paths {
		fi, err := os.Stat(p)
		if err != nil {
			return nil, err
		}
		files := []string{p}
		if fi.IsDir() {
			// The directory is listed, never used as a glob pattern: its
			// name may contain [, * or ?.
			ents, err := os.ReadDir(p)
			if err != nil {
				return nil, err
			}
			files = nil
			for _, e := range ents {
				if !e.IsDir() && strings.HasSuffix(e.Name(), ".json") {
					files = append(files, filepath.Join(p, e.Name()))
				}
			}
			sort.Strings(files)
		}
		for _, f := range files {
			data, err := os.ReadFile(f)
			if err != nil {
				return nil, err
			}
			d, err := Parse(data)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", f, err)
			}
			if err := s.Add(d, f); err != nil {
				return nil, err
			}
		}
	}
	return s, nil
}
