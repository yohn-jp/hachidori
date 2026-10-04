// Package e2e is the shared foundation of the Windows appliance E2E
// certification: the immutable release-candidate identity every shard
// consumes, and the bounded, secret-safe evidence every shard produces.
//
// The candidate is built exactly once per certification run by the
// workflow's candidate job. A shard never builds hachidori.exe; it verifies
// the downloaded artifact against its manifest and against identity values
// the workflow passes out of band (the source commit and the SHA-256 reported
// by the candidate job), then certifies those exact bytes.
//
// Hosted-runner evidence produced through this package is CI evidence only.
// It is never a physical Windows PASS; see docs/windows-certification-checklist.md.
package e2e

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Environment variables that connect a shard to the workflow.
const (
	// EnvEnable switches a shard from portable mode (scenarios skip) to
	// certification mode (every identity precondition is a hard failure).
	EnvEnable = "HACHIDORI_WINDOWS_E2E"
	// EnvCandidateDir is the directory holding candidate.json and the
	// executable it names.
	EnvCandidateDir = "HACHIDORI_E2E_CANDIDATE_DIR"
	// EnvEvidenceDir is the root under which each shard writes its evidence.
	EnvEvidenceDir = "HACHIDORI_E2E_EVIDENCE_DIR"
	// EnvExpectedSHA256 is the executable SHA-256 the candidate job reported
	// as a job output, independent of the artifact transport.
	EnvExpectedSHA256 = "HACHIDORI_E2E_CANDIDATE_SHA256"
	// EnvExpectedCommit is the source commit the run certifies.
	EnvExpectedCommit = "HACHIDORI_E2E_SOURCE_SHA"
)

// Schema identifiers of the two documents this package defines.
const (
	CandidateSchema = "hachidori.windows-e2e.candidate/v1"
	ResultSchema    = "hachidori.windows-e2e.result/v2"
)

// ManifestName is the candidate manifest's file name inside the candidate
// directory.
const ManifestName = "candidate.json"

var (
	shaPattern    = regexp.MustCompile(`^[0-9a-f]{64}$`)
	commitPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)
)

// Runner is the runner identity, taken only from the fixed allow-list of
// GitHub Actions variables below. No other environment value is recorded.
type Runner struct {
	OS           string `json:"os,omitempty"`
	Arch         string `json:"arch,omitempty"`
	Name         string `json:"name,omitempty"`
	ImageOS      string `json:"image_os,omitempty"`
	ImageVersion string `json:"image_version,omitempty"`
}

// RunnerFromEnv reads the runner identity where the runner provides it.
func RunnerFromEnv() Runner {
	return Runner{
		OS:           os.Getenv("RUNNER_OS"),
		Arch:         os.Getenv("RUNNER_ARCH"),
		Name:         os.Getenv("RUNNER_NAME"),
		ImageOS:      os.Getenv("ImageOS"),
		ImageVersion: os.Getenv("ImageVersion"),
	}
}

// Run identifies the workflow run that produced a document.
type Run struct {
	Workflow string `json:"workflow,omitempty"`
	ID       string `json:"id,omitempty"`
	Attempt  string `json:"attempt,omitempty"`
}

// RunFromEnv reads the workflow run identity where available.
func RunFromEnv() Run {
	return Run{
		Workflow: os.Getenv("GITHUB_WORKFLOW"),
		ID:       os.Getenv("GITHUB_RUN_ID"),
		Attempt:  os.Getenv("GITHUB_RUN_ATTEMPT"),
	}
}

// Manifest is candidate.json: the immutable identity of the one executable a
// certification run certifies.
type Manifest struct {
	Schema       string `json:"schema"`
	SourceCommit string `json:"source_commit"`
	File         string `json:"file"`
	SHA256       string `json:"sha256"`
	Size         int64  `json:"size"`
	GOOS         string `json:"goos"`
	GOARCH       string `json:"goarch"`
	GoVersion    string `json:"go_version,omitempty"`
	Runner       Runner `json:"runner"`
	Run          Run    `json:"run"`
}

// Candidate is a verified manifest together with the local path of the
// executable it describes.
type Candidate struct {
	Manifest
	Path string `json:"-"`
}

// Expect carries identity values supplied independently of the artifact. An
// empty field is not checked.
type Expect struct {
	SourceCommit string
	SHA256       string
}

// ExpectFromEnv reads the out-of-band identity from the environment.
func ExpectFromEnv() Expect {
	return Expect{
		SourceCommit: os.Getenv(EnvExpectedCommit),
		SHA256:       os.Getenv(EnvExpectedSHA256),
	}
}

// HashFile returns the lower-case hex SHA-256 and the size of the file.
func HashFile(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

// WriteCandidate computes the identity of the executable file inside dir and
// writes candidate.json next to it. file must be a bare file name that already
// exists in dir.
func WriteCandidate(dir, file, sourceCommit, goos, goarch, goVersion string) (Manifest, error) {
	if err := validateFileName(file); err != nil {
		return Manifest{}, err
	}
	if !commitPattern.MatchString(sourceCommit) {
		return Manifest{}, fmt.Errorf("source commit %q is not a 40-character lower-case hex SHA", sourceCommit)
	}
	sum, size, err := HashFile(filepath.Join(dir, file))
	if err != nil {
		return Manifest{}, fmt.Errorf("hash candidate: %w", err)
	}
	if size == 0 {
		return Manifest{}, errors.New("candidate executable is empty")
	}
	m := Manifest{
		Schema:       CandidateSchema,
		SourceCommit: sourceCommit,
		File:         file,
		SHA256:       sum,
		Size:         size,
		GOOS:         goos,
		GOARCH:       goarch,
		GoVersion:    goVersion,
		Runner:       RunnerFromEnv(),
		Run:          RunFromEnv(),
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return Manifest{}, err
	}
	if err := os.WriteFile(filepath.Join(dir, ManifestName), append(data, '\n'), 0o644); err != nil {
		return Manifest{}, err
	}
	return m, nil
}

// LoadCandidate verifies the candidate in dir. It fails closed: a missing or
// malformed manifest, a file name that is not a bare name, a file whose size or
// SHA-256 differs from the manifest, or any mismatch with the out-of-band
// expectation is an error. It never rebuilds or repairs anything.
func LoadCandidate(dir string, expect Expect) (Candidate, error) {
	data, err := os.ReadFile(filepath.Join(dir, ManifestName))
	if err != nil {
		return Candidate{}, fmt.Errorf("read candidate manifest: %w", err)
	}
	var m Manifest
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return Candidate{}, fmt.Errorf("parse candidate manifest: %w", err)
	}
	if m.Schema != CandidateSchema {
		return Candidate{}, fmt.Errorf("candidate manifest schema %q, want %q", m.Schema, CandidateSchema)
	}
	if err := validateFileName(m.File); err != nil {
		return Candidate{}, err
	}
	if !shaPattern.MatchString(m.SHA256) {
		return Candidate{}, fmt.Errorf("candidate manifest sha256 %q is not 64 lower-case hex characters", m.SHA256)
	}
	if !commitPattern.MatchString(m.SourceCommit) {
		return Candidate{}, fmt.Errorf("candidate manifest source_commit %q is not a 40-character lower-case hex SHA", m.SourceCommit)
	}
	path := filepath.Join(dir, m.File)
	sum, size, err := HashFile(path)
	if err != nil {
		return Candidate{}, fmt.Errorf("hash candidate: %w", err)
	}
	if size != m.Size || sum != m.SHA256 {
		return Candidate{}, fmt.Errorf("candidate %s does not match its manifest: sha256 %s size %d, manifest sha256 %s size %d",
			m.File, sum, size, m.SHA256, m.Size)
	}
	if expect.SHA256 != "" && !strings.EqualFold(expect.SHA256, sum) {
		return Candidate{}, fmt.Errorf("candidate sha256 %s differs from the expected %s", sum, expect.SHA256)
	}
	if expect.SourceCommit != "" && !strings.EqualFold(expect.SourceCommit, m.SourceCommit) {
		return Candidate{}, fmt.Errorf("candidate source commit %s differs from the expected %s", m.SourceCommit, expect.SourceCommit)
	}
	return Candidate{Manifest: m, Path: path}, nil
}

func validateFileName(name string) error {
	if name == "" || name == "." || name == ".." || name != filepath.Base(name) ||
		strings.ContainsAny(name, `/\`) {
		return fmt.Errorf("candidate file name %q is not a bare file name", name)
	}
	return nil
}
