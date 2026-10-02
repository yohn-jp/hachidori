package setup

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/yohn-jp/hachidori/internal/home"
)

// rangeServer is a local object server with HTTP range semantics and the
// failure modes a multi-GB download meets: an interrupted body, a server that
// ignores Range, a changed validator or length. It never leaves the machine.
type rangeServer struct {
	*httptest.Server
	mu          sync.Mutex
	body        []byte
	etag        string
	lastMod     string
	ignoreRange bool // answer every request with the full object (200)
	ignoreIf    bool // honour Range but never evaluate If-Range
	chunked     bool // no Content-Length
	cutAfter    int  // abort the first n bytes into the next full or ranged body (0: no cut); reset after use
	blockAfter  int  // send this many bytes, then wait for the client to go away (0: off); reset after use
	reqs        []rangeReq
}

type rangeReq struct{ Range, IfRange string }

func newRangeServer(t *testing.T, body []byte) *rangeServer {
	s := &rangeServer{body: body, etag: `"v1"`}
	s.Server = httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(s.Close)
	return s
}

func (s *rangeServer) set(f func(*rangeServer)) { s.mu.Lock(); f(s); s.mu.Unlock() }

func (s *rangeServer) requests() []rangeReq {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]rangeReq(nil), s.reqs...)
}

func (s *rangeServer) serve(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	body, etag, lastMod := s.body, s.etag, s.lastMod
	ignore, ignoreIf, chunked := s.ignoreRange, s.ignoreIf, s.chunked
	cut, block := s.cutAfter, s.blockAfter
	s.cutAfter, s.blockAfter = 0, 0
	s.reqs = append(s.reqs, rangeReq{r.Header.Get("Range"), r.Header.Get("If-Range")})
	s.mu.Unlock()

	h := w.Header()
	if etag != "" {
		h.Set("ETag", etag)
	}
	if lastMod != "" {
		h.Set("Last-Modified", lastMod)
	}
	start, status := 0, http.StatusOK
	if rg := r.Header.Get("Range"); rg != "" && !ignore {
		validator := r.Header.Get("If-Range")
		if ignoreIf || validator == "" || validator == etag || (etag == "" && validator == lastMod) {
			n, err := strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(rg, "bytes="), "-"))
			if err != nil || n >= len(body) {
				h.Set("Content-Range", fmt.Sprintf("bytes */%d", len(body)))
				w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
				return
			}
			start, status = n, http.StatusPartialContent
			h.Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", n, len(body)-1, len(body)))
		}
	}
	if !chunked {
		h.Set("Content-Length", strconv.Itoa(len(body)-start))
	}
	w.WriteHeader(status)
	rest := body[start:]
	switch {
	case cut > 0 && cut < len(rest):
		w.Write(rest[:cut])
		w.(http.Flusher).Flush()
		panic(http.ErrAbortHandler) // the connection dies mid-body
	case block > 0 && block < len(rest):
		w.Write(rest[:block])
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	default:
		w.Write(rest)
	}
}

func sumOf(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

func pattern(n int, seed byte) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*31+i/251) ^ seed
	}
	return b
}

// run is one resumable download attempt; it returns the log and the last
// reported progress.
func (s *rangeServer) run(t *testing.T, ctx context.Context, dst, want string) (string, Progress, error) {
	t.Helper()
	var log strings.Builder
	var last Progress
	obs := &Observer{OnProgress: func(p Progress) {
		if p.Step == StepDownload {
			last = p
		}
	}}
	err := fetchResumable(ctx, s.URL+"/obj", dst, want, &log, obs, Progress{Step: StepDownload, Detail: "obj"})
	return log.String(), last, err
}

func present(p string) bool { _, err := os.Stat(p); return err == nil }

func TestResumeContinuesAnInterruptedDownloadAfterProvingTheObject(t *testing.T) {
	body := pattern(1<<20, 1)
	s := newRangeServer(t, body)
	dst := filepath.Join(t.TempDir(), "m", "f.safetensors")
	s.set(func(s *rangeServer) { s.cutAfter = 300_000 })
	if _, _, err := s.run(t, context.Background(), dst, sumOf(body)); err == nil {
		t.Fatal("an interrupted body was reported complete")
	}
	if present(dst) {
		t.Fatal("a partial download was published as the object")
	}
	st, ok := InspectPartial(dst, s.URL+"/obj", sumOf(body))
	if !ok || st.Bytes != 300_000 || st.Total != int64(len(body)) {
		t.Fatalf("partial = %+v ok=%v, want 300000 of %d kept for resume", st, ok, len(body))
	}

	log, last, err := s.run(t, context.Background(), dst, sumOf(body))
	if err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(dst)
	if sumOf(got) != sumOf(body) {
		t.Fatal("resumed object differs from the pinned bytes")
	}
	reqs := s.requests()
	if len(reqs) != 2 || reqs[1].Range != "bytes=300000-" || reqs[1].IfRange != `"v1"` {
		t.Fatalf("requests %+v: the resume must send Range and the recorded validator", reqs)
	}
	if !strings.Contains(log, "resuming") {
		t.Fatalf("log does not say it resumed:\n%s", log)
	}
	if last.Resumed != 300_000 || last.Done != int64(len(body)) || last.Total != int64(len(body)) {
		t.Fatalf("progress = %+v: it must count the 300000 bytes already held and end at the full object", last)
	}
	if present(dst+".part") || present(dst+".part.json") {
		t.Fatal("partial files remain after publication")
	}
}

func TestResumeFallsBackWhenTheServerIgnoresRange(t *testing.T) {
	body := pattern(400_000, 2)
	s := newRangeServer(t, body)
	dst := filepath.Join(t.TempDir(), "f")
	s.set(func(s *rangeServer) { s.cutAfter = 100_000 })
	if _, _, err := s.run(t, context.Background(), dst, sumOf(body)); err == nil {
		t.Fatal("interrupted download reported complete")
	}
	s.set(func(s *rangeServer) { s.ignoreRange = true })
	log, last, err := s.run(t, context.Background(), dst, sumOf(body))
	if err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(dst)
	if sumOf(got) != sumOf(body) || len(got) != len(body) {
		t.Fatalf("a full body appended to the old partial corrupted the object (%d bytes)", len(got))
	}
	if !strings.Contains(log, "full object") || last.Resumed != 0 {
		t.Fatalf("the restart was not explicit (resumed=%d):\n%s", last.Resumed, log)
	}
}

func TestResumeRestartsWhenTheValidatorChanges(t *testing.T) {
	body := pattern(400_000, 3)
	for name, ignoreIf := range map[string]bool{
		"If-Range failure answered with the full object": false,
		"server that ignores If-Range answers 206":       true,
	} {
		t.Run(name, func(t *testing.T) {
			s := newRangeServer(t, body)
			dst := filepath.Join(t.TempDir(), "f")
			s.set(func(s *rangeServer) { s.cutAfter = 150_000 })
			if _, _, err := s.run(t, context.Background(), dst, sumOf(body)); err == nil {
				t.Fatal("interrupted download reported complete")
			}
			s.set(func(s *rangeServer) { s.etag = `"v2"`; s.ignoreIf = ignoreIf })
			log, last, err := s.run(t, context.Background(), dst, sumOf(body))
			if err != nil {
				t.Fatal(err)
			}
			got, _ := os.ReadFile(dst)
			if sumOf(got) != sumOf(body) {
				t.Fatal("object differs from the pinned bytes")
			}
			if last.Resumed != 0 {
				t.Fatalf("the old prefix was kept (resumed=%d) although the object's validator changed:\n%s", last.Resumed, log)
			}
		})
	}
}

func TestResumeRestartsWhenTheObjectLengthChanges(t *testing.T) {
	old := pattern(400_000, 4)
	s := newRangeServer(t, old)
	dst := filepath.Join(t.TempDir(), "f")
	s.set(func(s *rangeServer) { s.cutAfter = 150_000 })
	if _, _, err := s.run(t, context.Background(), dst, sumOf(old)); err == nil {
		t.Fatal("interrupted download reported complete")
	}
	// The same validator over a different length: a misbehaving server. The
	// recorded length no longer matches, so the prefix is not continued; the
	// new bytes are not the pinned object, so nothing is published.
	grown := append(append([]byte(nil), old...), pattern(1000, 9)...)
	s.set(func(s *rangeServer) { s.body, s.ignoreIf = grown, true })
	log, last, err := s.run(t, context.Background(), dst, sumOf(old))
	if err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("err = %v, want the digest to refuse the changed object", err)
	}
	if last.Resumed != 0 || !strings.Contains(log, "length changed") {
		t.Fatalf("length change was not detected (resumed=%d):\n%s", last.Resumed, log)
	}
	for _, p := range []string{dst, dst + ".part", dst + ".part.json"} {
		if present(p) {
			t.Fatalf("%s left after a changed object", p)
		}
	}
}

func TestResumeNeverPublishesACorruptPartial(t *testing.T) {
	body := pattern(400_000, 5)
	s := newRangeServer(t, body)
	dst := filepath.Join(t.TempDir(), "f")
	s.set(func(s *rangeServer) { s.cutAfter = 200_000 })
	if _, _, err := s.run(t, context.Background(), dst, sumOf(body)); err == nil {
		t.Fatal("interrupted download reported complete")
	}
	// The kept prefix is damaged (a torn write after a crash): same length,
	// wrong bytes. The record still claims the right object.
	b, _ := os.ReadFile(dst + ".part")
	b[1234] ^= 0xff
	if err := os.WriteFile(dst+".part", b, 0o644); err != nil {
		t.Fatal(err)
	}
	log, last, err := s.run(t, context.Background(), dst, sumOf(body))
	if err != nil {
		t.Fatalf("the corrupt partial must be discarded and the object fetched again: %v", err)
	}
	got, _ := os.ReadFile(dst)
	if sumOf(got) != sumOf(body) {
		t.Fatal("a corrupt partial reached the published object")
	}
	if !strings.Contains(log, "does not match the pinned digest") || last.Resumed != 0 {
		t.Fatalf("the digest failure and the restart were not reported (resumed=%d):\n%s", last.Resumed, log)
	}
	reqs := s.requests()
	if last := reqs[len(reqs)-1]; last.Range != "" {
		t.Fatalf("the restart must be a plain request, got %+v", last)
	}
}

func TestResumeDiscardsAPartialThatCannotBeTheObject(t *testing.T) {
	body := pattern(100_000, 6)
	url := func(s *rangeServer) string { return s.URL + "/obj" }
	for name, setup := range map[string]func(t *testing.T, s *rangeServer, dst string){
		"longer than the remote object": func(t *testing.T, s *rangeServer, dst string) {
			writePartial(t, dst, pattern(150_000, 6), url(s), sumOf(body), `"v1"`, int64(len(body)))
		},
		"no resume record": func(t *testing.T, s *rangeServer, dst string) {
			if err := os.WriteFile(dst+".part", body[:5000], 0o644); err != nil {
				t.Fatal(err)
			}
		},
		"record of another object": func(t *testing.T, s *rangeServer, dst string) {
			writePartial(t, dst, body[:5000], url(s), strings.Repeat("0", 64), `"v1"`, int64(len(body)))
		},
		"record without a validator": func(t *testing.T, s *rangeServer, dst string) {
			writePartial(t, dst, body[:5000], url(s), sumOf(body), "", int64(len(body)))
		},
		"complete length but wrong bytes": func(t *testing.T, s *rangeServer, dst string) {
			writePartial(t, dst, pattern(len(body), 7), url(s), sumOf(body), `"v1"`, int64(len(body)))
		},
	} {
		t.Run(name, func(t *testing.T) {
			s := newRangeServer(t, body)
			dst := filepath.Join(t.TempDir(), "f")
			setup(t, s, dst)
			_, last, err := s.run(t, context.Background(), dst, sumOf(body))
			if err != nil {
				t.Fatal(err)
			}
			got, _ := os.ReadFile(dst)
			if sumOf(got) != sumOf(body) || last.Resumed != 0 {
				t.Fatalf("an unprovable partial was continued (resumed=%d)", last.Resumed)
			}
			for _, r := range s.requests() {
				if r.Range != "" {
					t.Fatalf("a partial that cannot be the object was resumed: %+v", r)
				}
			}
		})
	}
}

func writePartial(t *testing.T, dst string, data []byte, url, want, etag string, total int64) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dst+".part", data, 0o644); err != nil {
		t.Fatal(err)
	}
	m := partialMeta{Schema: partialSchema, URL: url, SHA256: want, ETag: etag, Total: total}
	if err := home.WriteJSON(dst+".part.json", m); err != nil {
		t.Fatal(err)
	}
}

func TestResumeCancellationKeepsOnlyASafePartial(t *testing.T) {
	oldEvery := progressInterval
	progressInterval = 0 // report every write, so the test can cancel at an exact byte
	t.Cleanup(func() { progressInterval = oldEvery })
	body := pattern(600_000, 8)
	s := newRangeServer(t, body)
	dst := filepath.Join(t.TempDir(), "f")
	s.set(func(s *rangeServer) { s.blockAfter = 250_000 })
	ctx, cancel := context.WithCancel(context.Background())
	var log strings.Builder
	obs := &Observer{OnProgress: func(p Progress) {
		if p.Step == StepDownload && p.Done >= 250_000 {
			cancel()
		}
	}}
	err := fetchResumable(ctx, s.URL+"/obj", dst, sumOf(body), &log, obs, Progress{Step: StepDownload})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want a cancellation", err)
	}
	if present(dst) {
		t.Fatal("a cancelled download was published")
	}
	st, ok := InspectPartial(dst, s.URL+"/obj", sumOf(body))
	if !ok || st.Bytes < 250_000 || st.Bytes >= int64(len(body)) {
		t.Fatalf("partial after cancel = %+v ok=%v", st, ok)
	}
	// The kept partial completes, and a cancelled download of an object that
	// states no validator leaves nothing resumable.
	if _, last, err := s.run(t, context.Background(), dst, sumOf(body)); err != nil || last.Resumed != st.Bytes {
		t.Fatalf("resume after cancel: err=%v resumed=%d want %d", err, last.Resumed, st.Bytes)
	}

	s2 := newRangeServer(t, body)
	s2.set(func(s *rangeServer) { s.etag = ""; s.blockAfter = 250_000 })
	dst2 := filepath.Join(t.TempDir(), "g")
	ctx2, cancel2 := context.WithCancel(context.Background())
	obs2 := &Observer{OnProgress: func(p Progress) {
		if p.Done >= 250_000 {
			cancel2()
		}
	}}
	if err := fetchResumable(ctx2, s2.URL+"/obj", dst2, sumOf(body), io.Discard, obs2, Progress{Step: StepDownload}); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if present(dst2+".part") || present(dst2+".part.json") {
		t.Fatal("a download whose remote identity is unknown left an ambiguous partial")
	}
}

func TestResumeFinalChecksumMismatchPublishesNothing(t *testing.T) {
	body := pattern(100_000, 10)
	s := newRangeServer(t, body)
	dst := filepath.Join(t.TempDir(), "f")
	_, _, err := s.run(t, context.Background(), dst, strings.Repeat("0", 64))
	if err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("err = %v, want a digest mismatch", err)
	}
	for _, p := range []string{dst, dst + ".part", dst + ".part.json"} {
		if present(p) {
			t.Fatalf("%s left after a digest mismatch", p)
		}
	}
}

func TestResumeFreshAndAlreadyCompleteDownloads(t *testing.T) {
	body := pattern(100_000, 11)
	s := newRangeServer(t, body)
	dst := filepath.Join(t.TempDir(), "d", "f")
	if _, last, err := s.run(t, context.Background(), dst, sumOf(body)); err != nil || last.Resumed != 0 || last.Total != int64(len(body)) {
		t.Fatalf("fresh download: err=%v progress=%+v", err, last)
	}
	// Already complete and valid: no request, and a stale partial is dropped.
	before := len(s.requests())
	writePartial(t, dst, body[:10], s.URL+"/obj", sumOf(body), `"v1"`, int64(len(body)))
	if _, _, err := s.run(t, context.Background(), dst, sumOf(body)); err != nil {
		t.Fatal(err)
	}
	if len(s.requests()) != before || present(dst+".part") {
		t.Fatal("a valid object was fetched again, or its stale partial kept")
	}
}

func TestResumeDoesNotInventATotal(t *testing.T) {
	body := pattern(100_000, 12)
	s := newRangeServer(t, body)
	s.set(func(s *rangeServer) { s.chunked = true })
	dst := filepath.Join(t.TempDir(), "f")
	var totals []int64
	obs := &Observer{OnProgress: func(p Progress) {
		if p.Step == StepDownload {
			totals = append(totals, p.Total)
		}
	}}
	if err := fetchResumable(context.Background(), s.URL+"/obj", dst, sumOf(body), io.Discard, obs, Progress{Step: StepDownload}); err != nil {
		t.Fatal(err)
	}
	for _, tot := range totals {
		if tot != 0 {
			t.Fatalf("progress total %d for a body of unknown length", tot)
		}
	}
}

// The legacy whole-object download is unchanged: it never resumes and leaves
// nothing behind on failure.
func TestFetchRemainsAWholeObjectDownload(t *testing.T) {
	body := pattern(100_000, 13)
	s := newRangeServer(t, body)
	dst := filepath.Join(t.TempDir(), "f")
	s.set(func(s *rangeServer) { s.cutAfter = 30_000 })
	if err := fetch(s.URL+"/obj", dst, sumOf(body), io.Discard, nil, Progress{}); err == nil {
		t.Fatal("interrupted download reported complete")
	}
	if present(dst+".part") || present(dst+".part.json") {
		t.Fatal("the legacy download left a partial")
	}
	if err := fetch(s.URL+"/obj", dst, sumOf(body), io.Discard, nil, Progress{}); err != nil {
		t.Fatal(err)
	}
	for _, r := range s.requests() {
		if r.Range != "" {
			t.Fatalf("legacy fetch sent %+v", r)
		}
	}
}

// Model materialization continues an interrupted model file, prunes what is
// not part of the pinned set, and publishes only verified bytes.
func TestMaterializeModelResumesAnInterruptedFile(t *testing.T) {
	big := pattern(500_000, 14)
	small := []byte(`{"k":1}`)
	s := newRangeServer(t, nil)
	var mu sync.Mutex
	files := map[string][]byte{"big.safetensors": big, "config.json": small}
	cuts := map[string]int{"big.safetensors": 200_000}
	reqs := map[string][]string{}
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		rel := r.URL.Path[strings.LastIndex(r.URL.Path, "/")+1:]
		mu.Lock()
		body, cut := files[rel], cuts[rel]
		delete(cuts, rel)
		reqs[rel] = append(reqs[rel], r.Header.Get("Range"))
		mu.Unlock()
		w.Header().Set("ETag", `"`+rel+`"`)
		start := 0
		if rg := r.Header.Get("Range"); rg != "" {
			start, _ = strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(rg, "bytes="), "-"))
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, len(body)-1, len(body)))
			w.Header().Set("Content-Length", strconv.Itoa(len(body)-start))
			w.WriteHeader(http.StatusPartialContent)
		} else {
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		}
		if cut > 0 {
			w.Write(body[start : start+cut])
			w.(http.Flusher).Flush()
			panic(http.ErrAbortHandler)
		}
		w.Write(body[start:])
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	old := modelBaseURL
	modelBaseURL = srv.URL + "/"
	t.Cleanup(func() { modelBaseURL = old })

	m := home.ModelManifest{ID: "test-big", Provider: "laya", Repo: "test/big", Revision: strings.Repeat("ab", 20),
		Files: map[string]string{"big.safetensors": sumOf(big), "config.json": sumOf(small)}}
	h := home.Home{Root: t.TempDir()}
	if err := h.Ensure(); err != nil {
		t.Fatal(err)
	}
	final := h.Path("models", filepath.FromSlash(ModelDirName(m)))
	if err := materializeModel(context.Background(), h, m, io.Discard, nil); err == nil {
		t.Fatal("interrupted materialization reported complete")
	}
	if present(filepath.Join(final, "hachidori-model.json")) {
		t.Fatal("an incomplete model is materialized")
	}
	stage := final + ".staging"
	if err := os.WriteFile(filepath.Join(stage, "stray.bin"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	var resumed int64
	obs := &Observer{OnProgress: func(p Progress) {
		if p.Step == StepDownload && p.Detail == "big.safetensors" {
			resumed = max(resumed, p.Resumed)
		}
	}}
	if err := materializeModel(context.Background(), h, m, io.Discard, obs); err != nil {
		t.Fatal(err)
	}
	if resumed != 200_000 {
		t.Fatalf("resumed = %d, want the 200000 bytes already held", resumed)
	}
	if err := VerifyModel(final, m); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{"stray.bin", "big.safetensors.part", "big.safetensors.part.json"} {
		if present(filepath.Join(final, p)) {
			t.Fatalf("%s was published with the model", p)
		}
	}
	if present(stage) {
		t.Fatal("staging remains after publication")
	}
	_ = s
}
