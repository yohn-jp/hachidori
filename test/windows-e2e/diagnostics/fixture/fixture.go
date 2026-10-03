// Package fixture builds the disposable Hachidori home and process
// environment the diagnostics E2E exports a bundle from, and plants the canary
// secrets the bundle must never contain.
//
// The home is shaped exactly like an activated CPU home (activation record,
// Runtime Spec manifest of this build, catalog model manifest), so the real
// executable resolves it and serves status for it, but it holds no real
// runtime: the "python" file is a text placeholder that cannot be executed, so
// the worker deterministically fails to start and the bundle has a real failure
// to describe. It never downloads or runs a model.
//
// Canaries are unique random strings generated per run. Each is planted where
// a leak could come from: the process environment, an SSH key and known_hosts
// in the user profile, semantic request/question/state text in history and in
// the worker log, a dataset, a credentials file, and a model binary. A
// scenario then proves each canary exists at its source and is absent from the
// bundle, so the absence is not vacuous.
package fixture

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// Info describes the materialized home.
type Info struct {
	Root     string
	Runtime  string // runtime identity (= directory name)
	ModelID  string
	ModelDir string // models/<owner>--<repo>/<revision>, slash separated
	Device   string
	Python   string // the placeholder interpreter
	// ModelFile is the model file that carries a canary payload.
	ModelFile string
}

// Materialize writes an activated CPU home under root.
func Materialize(root string) (Info, error) {
	h := home.Home{Root: root}
	if err := h.Ensure(); err != nil {
		return Info{}, err
	}
	spec, err := setup.Desired("cpu")
	if err != nil {
		return Info{}, err
	}
	model, err := setup.LookupModel(setup.DefaultModel)
	if err != nil {
		return Info{}, err
	}
	name := spec.ID()
	active := home.Active{Runtime: name, ModelID: model.ID, Model: setup.ModelDirName(model), Device: "cpu"}
	rm := home.RuntimeManifest{
		Identity: name, Spec: spec, PythonVersion: spec.Python,
		PythonRelPath: "env/Scripts/python.exe", Installed: strings.Split(spec.Provider, ","),
	}
	if err := writeJSON(h.Path("runtime", name, "manifest.json"), rm); err != nil {
		return Info{}, err
	}
	python := h.PythonExe(active, rm)
	if err := writeFile(python, []byte("this is a placeholder, not an interpreter\n")); err != nil {
		return Info{}, err
	}
	modelDir := h.ModelDir(active)
	if err := writeJSON(filepath.Join(modelDir, "hachidori-model.json"), model); err != nil {
		return Info{}, err
	}
	if err := writeJSON(h.Path("state", "active-runtime.json"), active); err != nil {
		return Info{}, err
	}
	return Info{Root: root, Runtime: name, ModelID: model.ID, ModelDir: active.Model, Device: "cpu", Python: python}, nil
}

func writeJSON(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return home.WriteJSON(path, v)
}

func writeFile(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// Canary is one planted secret or private value.
type Canary struct {
	Label  string // what it stands for
	Needle string // the unique text that must never appear in a bundle
	Where  string // where it was planted, for the evidence
}

// Canaries is everything planted for one run.
type Canaries struct {
	List []Canary
	// Env are the KEY=value entries to add to the process environment.
	Env []string
	// Sources are files, relative to the home or profile as named by Where, that
	// must still hold their canary after the export (the export reads, never
	// removes, them).
	Sources []Source
	// LogSeqFirst and LogSeqLast bound the sequence numbers of the planted
	// prefixed worker-log lines.
	LogSeqFirst, LogSeqLast int
}

// Source is a planted file and the canary text it must contain.
type Source struct {
	Path   string
	Needle string
}

// Needles returns every canary needle.
func (c Canaries) Needles() []Canary { return c.List }

func token(prefix string) string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return prefix + hex.EncodeToString(b)
}

// Planted log shape. Lines with the worker prefix are the only ones a bundle may
// carry; the rest simulate raw worker stderr and library output.
const (
	LogPrefix = "[worker] "
	logLines  = 320
)

// LogSeq is the format of the planted prefixed lines.
func LogSeq(n int) string { return fmt.Sprintf("%sseq=%04d heartbeat", LogPrefix, n) }

// Plant writes the canaries into the disposable home and profile and returns
// them. profile is the throwaway user profile folder the process runs with.
func Plant(info Info, profile string) (Canaries, error) {
	var c Canaries
	add := func(label, needle, where string) { c.List = append(c.List, Canary{label, needle, where}) }
	plant := func(path, label, needle, body string) error {
		if err := writeFile(path, []byte(body)); err != nil {
			return err
		}
		c.Sources = append(c.Sources, Source{Path: path, Needle: needle})
		add(label, needle, path)
		return nil
	}

	// Environment: credentials a host environment commonly carries.
	for _, e := range []struct{ key, prefix, label string }{
		{"HF_TOKEN", "hf_", "Hugging Face token (environment)"},
		{"GITHUB_TOKEN", "ghp_", "GitHub token (environment)"},
		{"AWS_SECRET_ACCESS_KEY", "aws-secret-", "cloud secret (environment)"},
		{"HACHIDORI_E2E_CANARY_ENV", "env-canary-", "arbitrary environment value"},
	} {
		v := token(e.prefix)
		c.Env = append(c.Env, e.key+"="+v)
		add(e.label, v, "process environment "+e.key)
	}

	// SSH credentials and known_hosts of the user profile.
	keyBody := token("ssh-key-")
	if err := plant(filepath.Join(profile, ".ssh", "id_ed25519"), "SSH private key", keyBody,
		"-----BEGIN OPENSSH PRIVATE KEY-----\n"+keyBody+"\n-----END OPENSSH PRIVATE KEY-----\n"); err != nil {
		return c, err
	}
	hostCanary := token("known-host-") + ".example"
	if err := plant(filepath.Join(profile, ".ssh", "known_hosts"), "known_hosts entry", hostCanary,
		hostCanary+" ssh-ed25519 AAAAC3NzaC1lZDI1NTE5"+token("")+"\n"); err != nil {
		return c, err
	}

	// Semantic content: request/question/state text in history and a dataset.
	req, q, st := token("request-text-"), token("question-text-"), token("state-text-")
	if err := plant(filepath.Join(info.Root, "state", "history", "experiment-canary.json"), "semantic request text", req,
		fmt.Sprintf(`{"schema":"fixture","request":%q,"question":%q,"state":%q}`+"\n", req, q, st)); err != nil {
		return c, err
	}
	add("semantic question text", q, "state/history/experiment-canary.json")
	add("semantic state text", st, "state/history/experiment-canary.json")
	ds := token("dataset-row-")
	if err := plant(filepath.Join(info.Root, "state", "datasets", "canary.jsonl"), "dataset content", ds,
		fmt.Sprintf(`{"input":%q}`+"\n", ds)); err != nil {
		return c, err
	}

	// Credentials file under the home's cache and a model binary.
	pw := token("password-")
	if err := plant(filepath.Join(info.Root, "cache", "home", ".netrc"), "credentials file", pw,
		"machine canary.example login canary-user password "+pw+"\n"); err != nil {
		return c, err
	}
	mb := token("model-binary-")
	info.ModelFile = filepath.Join(info.Root, "models", filepath.FromSlash(info.ModelDir), "model.safetensors")
	if err := plant(info.ModelFile, "model binary", mb, strings.Repeat(mb+"\n", 2048)); err != nil {
		return c, err
	}

	// The worker log: Hachidori's own prefixed lines (kept, bounded) mixed with
	// raw stderr and library output that quote request text and secrets (never
	// kept), a prefixed line that names the home, one that quotes a credential
	// in a URL, and one that is far longer than a line may be.
	raw := token("raw-stderr-request-")
	urlPass := token("url-password-")
	queryTok := token("query-token-")
	var log strings.Builder
	root := info.Root
	for i := 1; i <= logLines; i++ {
		log.WriteString(LogSeq(i) + "\n")
		switch i {
		case 40:
			log.WriteString("Traceback (most recent call last):\n  File \"worker.py\", line 7\nValueError: request=" + raw + "\n")
		case 150:
			log.WriteString(LogPrefix + "model directory " + filepath.Join(root, "models", "x") + " ready\n")
		case 200:
			log.WriteString(LogPrefix + "fetch https://user:" + urlPass + "@host.example/p?token=" + queryTok + " done\n")
		case 250:
			log.WriteString(LogPrefix + "overlong " + strings.Repeat("x", 4000) + "\n")
		}
	}
	logPath := filepath.Join(root, "logs", "worker.log")
	if err := plant(logPath, "raw worker stderr quoting request text", raw, log.String()); err != nil {
		return c, err
	}
	add("credential in a prefixed log line (URL password)", urlPass, "logs/worker.log")
	add("credential in a prefixed log line (token query)", queryTok, "logs/worker.log")
	c.LogSeqFirst, c.LogSeqLast = 1, logLines
	return c, nil
}

// Environment is the complete environment the executable under test runs with:
// the OS essentials, an isolated profile and temp folder, and the canaries.
// Nothing else is inherited, so a real user's variables cannot leak into or out
// of the run.
func Environment(profile, tmp string, canaries Canaries) []string {
	var env []string
	for _, k := range []string{"SystemRoot", "SYSTEMROOT", "windir", "ComSpec", "PATHEXT", "PATH", "SystemDrive"} {
		if v := os.Getenv(k); v != "" {
			env = append(env, k+"="+v)
		}
	}
	env = append(env,
		"USERPROFILE="+profile, "HOME="+profile,
		"LOCALAPPDATA="+filepath.Join(profile, "AppData", "Local"), "APPDATA="+filepath.Join(profile, "AppData", "Roaming"),
		"TEMP="+tmp, "TMP="+tmp, "TMPDIR="+tmp,
	)
	return append(env, canaries.Env...)
}
