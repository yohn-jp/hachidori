package update

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
)

// Class names why an update step failed. The set follows the failure semantics
// of the update contract: each class has its own operator hint.
type Class string

const (
	ClassNetwork          Class = "network"               // release metadata or network failure
	ClassMalformed        Class = "malformed_metadata"    // release metadata is not what the release workflow publishes
	ClassAsset            Class = "asset"                 // the Windows asset is missing or ambiguous
	ClassDownload         Class = "download"              // the executable could not be downloaded
	ClassVerification     Class = "verification_artifact" // the checksum asset is missing, unreadable or malformed
	ClassChecksumMismatch Class = "checksum_mismatch"     // the executable does not match its checksum
	ClassReplace          Class = "replacement"           // the executable could not be replaced
	ClassRestart          Class = "restart"               // the updated executable could not be restarted
	ClassRefused          Class = "refused"               // the action is not allowed in the current state
)

// Error is a classified update failure.
type Error struct {
	Class Class
	Msg   string
	Err   error
}

func (e *Error) Error() string {
	if e.Err != nil {
		return e.Msg + ": " + e.Err.Error()
	}
	return e.Msg
}

func (e *Error) Unwrap() error { return e.Err }

func fail(c Class, err error, format string, a ...any) *Error {
	return &Error{Class: c, Msg: fmt.Sprintf(format, a...), Err: err}
}

// Hints are the operator's next step for each failure class.
var Hints = map[Class]string{
	ClassNetwork:          "Hachidori could not read the release list from GitHub. Check the network connection and try Check for updates again. The installed Hachidori is unchanged.",
	ClassMalformed:        "GitHub's release data is not in the shape the release workflow publishes, so nothing was offered. The installed Hachidori is unchanged.",
	ClassAsset:            "This release does not carry exactly one Windows executable and one checksum file. Choose another release or try again later.",
	ClassDownload:         "The executable could not be downloaded. Nothing was installed; Download can be retried.",
	ClassVerification:     "The release's checksum file is missing or not usable, so the download was discarded. Nothing was installed.",
	ClassChecksumMismatch: "The downloaded file does not match the release's checksum, so it was discarded. Nothing was installed; Download can be retried.",
	ClassReplace:          "The executable could not be replaced. The previous executable was kept or restored; Hachidori keeps running from it.",
	ClassRestart:          "The executable was replaced but could not be restarted. Start Hachidori again from the same path.",
	ClassRefused:          "",
}

// Schema versions the ready and result records.
const (
	readySchema  = "hachidori.update-ready/1"
	resultSchema = "hachidori.update-result/1"
)

// Dir is the update area beneath HACHIDORI_HOME. It holds only what an
// explicit Download produced and what the helper reported: staged executables,
// the ready record, the result record and copies of the helper.
func Dir(root string) string { return filepath.Join(root, "state", "updates") }

func readyPath(root string) string  { return filepath.Join(Dir(root), "ready.json") }
func resultPath(root string) string { return filepath.Join(Dir(root), "result.json") }
func helperDir(root string) string  { return filepath.Join(Dir(root), "helper") }

// StagedPath is where the verified executable of a tag is kept. It is derived
// from the tag and the fixed asset name, never taken from input.
func StagedPath(root string, v Version) string {
	return filepath.Join(Dir(root), v.String(), ExeAsset)
}

// Ready is the record of a downloaded and verified executable: the only state
// from which Restart & update is possible.
type Ready struct {
	Schema     string    `json:"schema"`
	Tag        string    `json:"tag"`
	Prerelease bool      `json:"prerelease,omitempty"`
	Asset      string    `json:"asset"`
	SHA256     string    `json:"sha256"`
	Size       int64     `json:"size"`
	Target     string    `json:"target"` // the executable this update was prepared to replace
	Created    time.Time `json:"created"`
}

// validate checks the record is self-consistent. It does not look at disk.
func (r Ready) validate() (Version, error) {
	v, err := ParseVersion(r.Tag)
	switch {
	case r.Schema != readySchema:
		return v, fmt.Errorf("unknown ready record schema %q", r.Schema)
	case err != nil:
		return v, err
	case !(v.IsStable() || v.IsDevelopment()):
		return v, fmt.Errorf("tag %q is not a release the updater installs", r.Tag)
	case r.Asset != ExeAsset:
		return v, fmt.Errorf("ready record names asset %q, not %s", r.Asset, ExeAsset)
	case !hex64.MatchString(r.SHA256):
		return v, errors.New("ready record has no valid SHA-256")
	case r.Size <= 0 || r.Size > maxExeSize:
		return v, fmt.Errorf("ready record size %d is not acceptable", r.Size)
	case !filepath.IsAbs(r.Target) || filepath.Clean(r.Target) != r.Target:
		return v, errors.New("ready record has no clean absolute destination")
	}
	return v, nil
}

// LoadReady reads and validates state/updates/ready.json; a missing record is
// fs.ErrNotExist.
func LoadReady(root string) (Ready, Version, error) {
	var r Ready
	if err := home.ReadJSON(readyPath(root), &r); err != nil {
		return Ready{}, Version{}, err
	}
	v, err := r.validate()
	if err != nil {
		return Ready{}, Version{}, err
	}
	return r, v, nil
}

// Result outcomes the helper reports.
const (
	OutcomeApplied  = "applied"
	OutcomeFailed   = "replacement_failed"
	OutcomeRestart  = "restart_failed"
	OutcomeRefused  = "refused"
	maxResultLength = 1 << 16
)

// Result is the helper's report of the last replacement attempt.
type Result struct {
	Schema  string    `json:"schema"`
	Tag     string    `json:"tag"`
	SHA256  string    `json:"sha256"`
	Outcome string    `json:"outcome"`
	Message string    `json:"message,omitempty"`
	Target  string    `json:"target"`
	Time    time.Time `json:"time"`
}

// LoadResult reads state/updates/result.json; a missing record is
// fs.ErrNotExist.
func LoadResult(root string) (Result, error) {
	var r Result
	b, err := os.ReadFile(resultPath(root))
	if err != nil {
		return r, err
	}
	if len(b) > maxResultLength {
		return r, errors.New("update result record is too large")
	}
	if err := home.ReadJSON(resultPath(root), &r); err != nil {
		return Result{}, err
	}
	if r.Schema != resultSchema {
		return Result{}, fmt.Errorf("unknown update result schema %q", r.Schema)
	}
	return r, nil
}

// writeResult records a replacement attempt. It only writes into an update
// area that already exists beneath an absolute home: a refused or malformed
// plan never creates directories anywhere.
func writeResult(root string, r Result) error {
	if !filepath.IsAbs(root) {
		return errors.New("home is not an absolute path")
	}
	if fi, err := os.Stat(Dir(root)); err != nil || !fi.IsDir() {
		return errors.New("no update area to report into")
	}
	r.Schema = resultSchema
	return home.WriteJSON(resultPath(root), r)
}

// FileSHA256 hashes a file.
func FileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// peMagic starts every Windows executable.
const peMagic = "MZ"

// checkExecutable reports an error unless path is a regular file (not a
// symbolic link or directory) that starts like a Windows executable.
func checkExecutable(path string) (fs.FileInfo, error) {
	fi, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !fi.Mode().IsRegular() {
		return nil, fmt.Errorf("%s is not a regular file", path)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	head := make([]byte, len(peMagic))
	if _, err := io.ReadFull(f, head); err != nil || string(head) != peMagic {
		return nil, fmt.Errorf("%s is not a Windows executable", path)
	}
	return fi, nil
}

func samePath(a, b string) bool {
	if os.PathSeparator == '\\' {
		return strings.EqualFold(a, b)
	}
	return a == b
}

// Settings is the update subsystem's slice of settings.json: the selected
// channel, what the last explicit check found and which release the installed
// executable is known to be. It is stored by the settings authority.
type Settings struct {
	Channel   Channel    `json:"channel,omitempty"`
	LastCheck *LastCheck `json:"last_check,omitempty"`
	Installed *Installed `json:"installed,omitempty"`
}

// LastCheck is the locally remembered outcome of the last explicit check.
type LastCheck struct {
	Time    time.Time `json:"time"`
	Channel Channel   `json:"channel"`
	OK      bool      `json:"ok"`
	Class   Class     `json:"class,omitempty"` // failure class when not OK
	Message string    `json:"message,omitempty"`
	Latest  string    `json:"latest,omitempty"` // newest eligible tag, when there was one
}

// Installed binds a release version to the digest of the executable that is
// that release, so a version is only believed for exactly that executable.
type Installed struct {
	Version string `json:"version"`
	SHA256  string `json:"sha256"`
}

// Clone returns an independent copy.
func (s Settings) Clone() Settings {
	if s.LastCheck != nil {
		c := *s.LastCheck
		s.LastCheck = &c
	}
	if s.Installed != nil {
		i := *s.Installed
		s.Installed = &i
	}
	return s
}

// Validate accepts only known channels, a well-formed version and digest.
func (s Settings) Validate() error {
	if s.Channel != "" {
		if _, err := ParseChannel(string(s.Channel)); err != nil {
			return err
		}
	}
	if c := s.LastCheck; c != nil && c.Channel != "" {
		if _, err := ParseChannel(string(c.Channel)); err != nil {
			return err
		}
	}
	if i := s.Installed; i != nil {
		if _, err := ParseVersion(i.Version); err != nil {
			return err
		}
		if !hex64.MatchString(i.SHA256) {
			return errors.New("installed record has no valid SHA-256")
		}
	}
	return nil
}

// Store persists Settings. internal/settings.Store is the production one; the
// update package owns no file of its own for them.
type Store interface {
	UpdateSettings() (Settings, error)
	ModifyUpdateSettings(func(*Settings) error) error
}
