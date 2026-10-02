package diagnostics

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/redact"
)

var fakeSecrets = []string{
	"hf_AbCdEf0123456789secrettoken",
	"ghp_0123456789abcdefSECRETvalue",
	"s3cr3t-passw0rd",
	"Bearer eyJhbGciOiJIUzI1NiJ9.payloadpart.sigpart",
	"session=cookie-secret-value",
}

func noSecrets(t *testing.T, where string, b []byte) {
	t.Helper()
	for _, s := range []string{"hf_AbCdEf0123456789secrettoken", "ghp_0123456789abcdefSECRETvalue", "s3cr3t-passw0rd", "eyJhbGciOiJIUzI1NiJ9", "cookie-secret-value"} {
		if strings.Contains(string(b), s) {
			t.Fatalf("%s: the secret %q survived into the diagnostic", where, s)
		}
	}
}

func input(root string) ForgeInput {
	cause := fmt.Errorf("GET https://user:s3cr3t-passw0rd@huggingface.co/x?token=hf_AbCdEf0123456789secrettoken: reset (Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payloadpart.sigpart)")
	return ForgeInput{
		Operation: ForgeOperation{ID: "op-1", Kind: "optimize", Phase: "quantizing", Started: "2026-10-02T00:00:00Z"},
		Identity:  ForgeIdentity{Model: "clef-flash", Provider: "clef", SourceRevision: strings.Repeat("ab", 20), Variant: "clef-flash--w4--0123456789ab", Runtime: "rt-1"},
		Optimization: &ForgeOptimization{Recipe: "clef-flash-w4a16-rtn-g128", RecipeSHA256: "d1", Backend: "llmcompressor", BackendVersion: "0.14.0", Scheme: "W4A16",
			Preserved: []string{"lm_head", `re:model\.visual.*`}},
		Certification: &ForgeCertification{PolicyID: "p", DatasetSHA256: "d2"},
		Runtime:       ForgeRuntime{Device: "cuda", DType: "bfloat16", Torch: "2.11.0+cu128", DeviceName: "RTX 3060 " + fakeSecrets[0]},
		Resources:     ForgeResources{RAMTotal: 64 << 30, VRAMTotal: 12 << 30},
		Preflight: &ForgePreflight{Kind: "optimize", Outcome: "attention", Findings: []ForgeFinding{
			{ID: "memory.fit", Status: "unknown", Summary: "host RAM is 64 GiB under " + root + " password=" + fakeSecrets[2]}}},
		Err:        fmt.Errorf("optimizer exited: %w", fmt.Errorf("worker: %w", cause)),
		Phase:      "quantizing",
		StderrTail: []string{"Traceback (most recent call last):", "HF_TOKEN=" + fakeSecrets[0], "Authorization: " + fakeSecrets[3], "Cookie: " + fakeSecrets[4], "file " + root + "/models/x"},
		LogTail:    []string{"downloading https://huggingface.co/a?api_key=" + fakeSecrets[1], "ok"},
		Secondary:  []string{"the log could not be read: " + fakeSecrets[0] + " and " + fakeSecrets[1]},
	}
}

func TestForgeDiagnosticRedactsSecretsEverywhere(t *testing.T) {
	root := t.TempDir()
	d := BuildForge(input(root), root, time.Unix(1_700_000_000, 0))
	b, _ := marshalForge(d)
	noSecrets(t, "document", b)
	if strings.Contains(string(b), root) {
		t.Fatal("the HACHIDORI_HOME path survived")
	}
	if !strings.Contains(string(b), "<HACHIDORI_HOME>") || !strings.Contains(string(b), "<redacted>") {
		t.Fatalf("placeholders missing:\n%s", b)
	}
	// Identities and the useful failure evidence are kept.
	for _, want := range []string{"clef-flash--w4--0123456789ab", "clef-flash-w4a16-rtn-g128", "llmcompressor", "quantizing", "optimizer exited", "Traceback (most recent call last):", "memory.fit"} {
		if !strings.Contains(string(b), want) {
			t.Errorf("%q missing from the diagnostic", want)
		}
	}
	if len(d.Failure.Chain) != 3 || d.Failure.Chain[0].Type == "" {
		t.Fatalf("error chain %+v", d.Failure.Chain)
	}
	if d.Schema != ForgeSchema || !d.LocalOnly || len(d.Excluded) == 0 || !ValidForgeID(d.ID) {
		t.Fatalf("envelope %+v", d)
	}
	// Saved and read back, still clean.
	if _, err := SaveForge(root, d); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(filepath.Join(root, "state", "forge", "diagnostics", d.ID+".json"))
	noSecrets(t, "stored file", raw)
}

// The document is allowlist-shaped: its top-level fields are exactly the typed
// sections, so nothing else can be carried (no free-form map, no environment).
func TestForgeDiagnosticHasOnlyAllowlistedSections(t *testing.T) {
	d := BuildForge(input(t.TempDir()), "", time.Now())
	b, _ := json.Marshal(d)
	var top map[string]json.RawMessage
	json.Unmarshal(b, &top)
	allowed := map[string]bool{"schema": true, "id": true, "created_utc": true, "operation": true, "identity": true, "optimization": true, "certification": true,
		"runtime": true, "resources": true, "preflight": true, "failure": true, "secondary": true, "local_only": true, "excluded": true, "truncated": true}
	for k := range top {
		if !allowed[k] {
			t.Errorf("unexpected top-level field %q", k)
		}
	}
	for _, banned := range []string{"environment", "env", "weights", "dataset_contents", "questions", "state", "headers"} {
		if _, ok := top[banned]; ok {
			t.Errorf("banned field %q present", banned)
		}
	}
}

func TestForgeDiagnosticIsBounded(t *testing.T) {
	root := t.TempDir()
	in := input(root)
	huge := strings.Repeat("x", 8<<10)
	in.StderrTail, in.LogTail = nil, nil
	for i := 0; i < 300; i++ {
		in.StderrTail = append(in.StderrTail, fmt.Sprintf("stderr line %d %s", i, huge))
		in.LogTail = append(in.LogTail, fmt.Sprintf("log line %d %s", i, huge))
	}
	in.Err = errors.New(huge)
	in.Secondary = nil
	for i := 0; i < 100; i++ {
		in.Secondary = append(in.Secondary, huge)
	}
	for i := 0; i < 200; i++ {
		in.Preflight.Findings = append(in.Preflight.Findings, ForgeFinding{ID: "f", Status: "unknown", Summary: huge})
	}
	d := BuildForge(in, root, time.Now())
	b, _ := json.Marshal(d)
	if len(b) > MaxForgeBytes {
		t.Fatalf("diagnostic is %d bytes, bound %d", len(b), MaxForgeBytes)
	}
	if len(d.Failure.StderrTail) > MaxForgeStderrLines || len(d.Failure.LogTail) > MaxForgeLogLines {
		t.Fatalf("tails %d/%d", len(d.Failure.StderrTail), len(d.Failure.LogTail))
	}
	for _, l := range append(append([]string{}, d.Failure.StderrTail...), d.Failure.LogTail...) {
		if len(l) > MaxForgeLineBytes+len("...[truncated]") {
			t.Fatalf("a line of %d bytes", len(l))
		}
	}
	if len(d.Failure.Error) > 2048+20 || len(d.Preflight.Findings) > maxForgeFindings {
		t.Fatalf("error %d findings %d", len(d.Failure.Error), len(d.Preflight.Findings))
	}
}

func TestForgeDiagnosticStoreListsLoadsAndPrunes(t *testing.T) {
	root := t.TempDir()
	var ids []string
	for i := 0; i < KeepForge+5; i++ {
		in := input(root)
		in.Operation.Kind = []string{"optimize", "probe"}[i%2]
		in.Err = fmt.Errorf("failure %d", i)
		d := BuildForge(in, root, time.Unix(1_700_000_000+int64(i), 0))
		if _, err := SaveForge(root, d); err != nil {
			t.Fatal(err)
		}
		ids = append(ids, d.ID)
	}
	all := ListForge(root, ForgeFilter{})
	if len(all) != KeepForge || all[0].ID != ids[len(ids)-1] || all[len(all)-1].ID != ids[5] {
		t.Fatalf("kept %d diagnostics, newest %s oldest %s", len(all), all[0].ID, all[len(all)-1].ID)
	}
	if latest, ok := LatestForge(root, ForgeFilter{Kind: "probe"}); !ok || latest.ID != ids[len(ids)-2] {
		t.Fatalf("latest probe %s, want %s", latest.ID, ids[len(ids)-2])
	}
	if _, ok := LatestForge(root, ForgeFilter{Model: "nope"}); ok {
		t.Fatal("a filter that matches nothing found a diagnostic")
	}
	d, err := LoadForge(root, ids[10])
	if err != nil || d.ID != ids[10] {
		t.Fatalf("load %v %v", d.ID, err)
	}
	for _, bad := range []string{"../../etc/passwd", "optimize-1-2", ids[0] + ".json", ""} {
		if _, err := LoadForge(root, bad); err == nil {
			t.Errorf("%q loaded", bad)
		}
	}
	if _, err := LoadForge(root, ids[0]); err == nil {
		t.Fatal("a pruned diagnostic still loads")
	}
	// Export: to a directory and to a file; nothing leaves the machine.
	out := t.TempDir()
	p, err := ExportForge(d, out)
	if err != nil || filepath.Base(p) != d.ID+".json" {
		t.Fatalf("export %s %v", p, err)
	}
	if _, err := ExportForge(d, filepath.Join(out, "x", "copy.json")); err != nil {
		t.Fatal(err)
	}
}

// storedRoundTrip saves d, reads the stored bytes back as an independent
// reader would and returns the decoded document.
func storedRoundTrip(t *testing.T, root string, d ForgeDiagnostic) ForgeDiagnostic {
	t.Helper()
	path, err := SaveForge(root, d)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var back ForgeDiagnostic
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	return back
}

// The ID of a stored diagnostic is the content identity of the stored document:
// recomputing it from what was read back reproduces it, for a normal document
// and for one the bound truncated.
func TestForgeDiagnosticIDIsTheStoredContentIdentity(t *testing.T) {
	for name, build := range map[string]func(root string) ForgeInput{
		"normal": input,
		"oversized, truncated": func(root string) ForgeInput {
			in := input(root)
			for i := 0; i < 400; i++ {
				in.StderrTail = append(in.StderrTail, fmt.Sprintf("stderr %d %s", i, strings.Repeat("s", 600)))
				in.LogTail = append(in.LogTail, fmt.Sprintf("log %d %s", i, strings.Repeat("l", 600)))
			}
			for i := 0; i < 60; i++ {
				in.Preflight.Findings = append(in.Preflight.Findings, ForgeFinding{ID: "f", Status: "unknown", Summary: strings.Repeat("p", 600)})
			}
			return in
		},
	} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			d := BuildForge(build(root), root, time.Unix(1_700_000_000, 0))
			if want := name != "normal"; d.Truncated != want {
				t.Fatalf("truncated = %v, want %v", d.Truncated, want)
			}
			if b, _ := json.Marshal(d); len(b) > MaxForgeBytes {
				t.Fatalf("%d bytes over the bound", len(b))
			}
			if err := VerifyForge(d); err != nil {
				t.Fatal(err)
			}
			back := storedRoundTrip(t, root, d)
			if back.ID != d.ID || ForgeContentID(back) != d.ID {
				t.Fatalf("stored ID %s, recomputed %s", back.ID, ForgeContentID(back))
			}
			if err := VerifyForge(back); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadForge(root, d.ID); err != nil {
				t.Fatalf("load: %v", err)
			}
			// Any change of the stored content no longer verifies, and the
			// reader refuses it.
			back.Failure.Error += "!"
			if VerifyForge(back) == nil {
				t.Fatal("an edited document verified")
			}
			b, _ := marshalForge(back)
			os.WriteFile(filepath.Join(root, "state", "forge", "diagnostics", d.ID+".json"), b, 0o600)
			if _, err := LoadForge(root, d.ID); err == nil || !strings.Contains(err.Error(), "content identity") {
				t.Fatalf("an edited stored document loaded: %v", err)
			}
		})
	}
}

// Truncation happens before the identity is taken: a truncated document's ID
// is not the identity of the document before truncation.
func TestForgeDiagnosticIDIsTakenAfterTruncation(t *testing.T) {
	root := t.TempDir()
	in := input(root)
	for i := 0; i < 400; i++ {
		in.StderrTail = append(in.StderrTail, fmt.Sprintf("stderr %d %s", i, strings.Repeat("s", 600)))
		in.LogTail = append(in.LogTail, fmt.Sprintf("log %d %s", i, strings.Repeat("l", 600)))
	}
	for i := 0; i < 60; i++ {
		in.Preflight.Findings = append(in.Preflight.Findings, ForgeFinding{ID: "f", Status: "unknown", Summary: strings.Repeat("p", 600)})
	}
	d := BuildForge(in, root, time.Unix(1_700_000_000, 0))
	if !d.Truncated {
		t.Fatal("fixture was not truncated")
	}
	full := d
	full.Failure.StderrTail = tail(in.StderrTail, redact.New(root), MaxForgeStderrLines)
	full.Failure.LogTail = tail(in.LogTail, redact.New(root), MaxForgeLogLines)
	full.Truncated = false
	if len(full.Failure.LogTail) == len(d.Failure.LogTail) && len(full.Failure.StderrTail) == len(d.Failure.StderrTail) {
		t.Fatal("nothing was truncated from the tails")
	}
	if ForgeContentID(full) == d.ID {
		t.Fatal("the ID describes the document before truncation")
	}
}

// A document from before content identities stays readable without
// verification; it is a legacy document.
func TestLegacyForgeDiagnosticStaysReadable(t *testing.T) {
	root := t.TempDir()
	d := BuildForge(input(root), root, time.Unix(1_700_000_000, 0))
	d.Schema = LegacyForgeSchema
	d.Failure.Error = "changed after the ID was taken, as a v1 truncation did"
	b, _ := marshalForge(d)
	dir := filepath.Join(root, "state", "forge", "diagnostics")
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, d.ID+".json"), b, 0o600)
	got, err := LoadForge(root, d.ID)
	if err != nil || got.Schema != LegacyForgeSchema || VerifyForge(got) == nil {
		t.Fatalf("legacy load %v; verify must not pass for its pre-truncation ID", err)
	}
}
