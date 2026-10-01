package dashboard

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/history"
)

// The benchmarks below are the reproducible server-side half of the desktop
// performance budgets in docs/desktop.md: the cost of producing each
// workspace document and the shell's periodic live fragment, including the
// representative 100 / 1,000 / 10,000-record evidence and history views.
// They run only with -bench and never in the canonical test run.

// benchPage measures one GET and reports the response size.
func benchPage(b *testing.B, e *env, path string) {
	b.Helper()
	var n int
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec := e.get(b, path)
		if rec.Code != 200 {
			b.Fatalf("GET %s: %d", path, rec.Code)
		}
		n = rec.Body.Len()
	}
	b.ReportMetric(float64(n), "bytes/op")
}

func BenchmarkWorkspaces(b *testing.B) {
	for _, p := range []string{"/", "/live", "/diagnostics", "/workbench", "/experiments", "/errors"} {
		b.Run(url.PathEscape(p), func(b *testing.B) {
			benchPage(b, newEnv(b), p)
		})
	}
}

// BenchmarkEvidenceRecords is the Evidence workspace with an open report of n
// observations, unfiltered (the default view) and inspecting one row.
func BenchmarkEvidenceRecords(b *testing.B) {
	for _, n := range []int{100, 1000, 10000} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			e, rep := scaledExperiment(b, n)
			path := filepath.Join(b.TempDir(), "report.json")
			if err := os.WriteFile(path, rep, 0o644); err != nil {
				b.Fatal(err)
			}
			e.post(b, "/errors/open", url.Values{"path": {path}})
			if body := e.get(b, "/errors").Body.String(); !strings.Contains(body, `id="obs-h"`) {
				b.Fatal("report did not open")
			}
			b.Run("default", func(b *testing.B) { benchPage(b, e, "/errors") })
			b.Run("inspect", func(b *testing.B) { benchPage(b, e, fmt.Sprintf("/errors?obs=%d", n/2)) })
		})
	}
}

// BenchmarkHistoryRecords is the Experiments workspace with n saved
// experiments in history. One entry is saved through the store; the other
// n-1 are its index rows under distinct entry ids, which is exactly what the
// workspace lists (seeding n real saves is quadratic, see docs/desktop.md).
func BenchmarkHistoryRecords(b *testing.B) {
	for _, n := range []int{100, 1000, 10000} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			e, rep := scaledExperiment(b, 10)
			cfg := e.d.cfg
			cfg.HistoryDir = filepath.Join(e.home, "state", "history")
			e.d = New(cfg)
			if _, err := e.d.hist.Save(rep, "run", ""); err != nil {
				b.Fatal(err)
			}
			list, _, err := e.d.hist.List()
			if err != nil || len(list) != 1 {
				b.Fatal(list, err)
			}
			entries := make([]history.Summary, n)
			for i := range entries {
				sm := list[0]
				if i > 0 {
					sm.ID = fmt.Sprintf("20260101T000000Z-%012x", i)
					sm.Label = fmt.Sprintf("run %d", i)
					if err := os.Mkdir(filepath.Join(cfg.HistoryDir, "entries", sm.ID), 0o755); err != nil {
						b.Fatal(err)
					}
				}
				entries[i] = sm
			}
			ix, _ := json.Marshal(map[string]any{"schema": history.IndexSchema, "entries": entries})
			if err := os.WriteFile(filepath.Join(cfg.HistoryDir, "index.json"), ix, 0o644); err != nil {
				b.Fatal(err)
			}
			if got, _, _ := e.d.hist.List(); len(got) != n {
				b.Fatalf("seeded %d entries, want %d", len(got), n)
			}
			benchPage(b, e, "/experiments")
		})
	}
}

// scaledExperiment runs a real experiment of n single-question cases against
// the fake inference API and returns the env and its canonical report bytes.
func scaledExperiment(b *testing.B, n int) (*env, []byte) {
	b.Helper()
	e, _ := newWorkbenchEnv(b)
	dir := b.TempDir()
	var ds strings.Builder
	for i := 0; i < n; i++ {
		exp := []string{"yes", "no"}[i%2]
		fmt.Fprintf(&ds, `{"id":"c%d","state":"s%d","questions":[{"id":"q%d","type":"choice","instructions":"X?","choices":["yes","no"]}],"expected":{"q%d":"%s"}}`+"\n", i, i, i%7, i%7, exp)
	}
	dataset := writeDef(b, dir, "dataset.jsonl", ds.String())
	e.post(b, "/experiments/run", url.Values{"dataset": {dataset}, "warmup": {"0"}, "passes": {"1"}})
	deadline := time.Now().Add(2 * time.Minute)
	for {
		if x := e.d.exp.snapshot(); x != nil && x.State != ExpRunning {
			if x.State != ExpSucceeded || x.Report == nil {
				b.Fatalf("experiment %+v", x)
			}
			break
		}
		if time.Now().After(deadline) {
			b.Fatal("experiment did not finish")
		}
		time.Sleep(10 * time.Millisecond)
	}
	out := filepath.Join(dir, "report.json")
	if body := e.post(b, "/experiments/export", url.Values{"seq": {"1"}, "export_path": {out}}).Body.String(); !strings.Contains(body, "wrote") {
		b.Fatal("export failed")
	}
	data, err := os.ReadFile(out)
	if err != nil {
		b.Fatal(err)
	}
	return e, data
}
