package setup

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
)

// Resumable acquisition of large pinned artifacts.
//
// An interrupted download leaves a partial: "<dst>.part" holding a prefix of
// the object and "<dst>.part.json" recording which remote object that prefix
// came from. A partial is never a materialized artifact: nothing reads it, it
// has no digest status, and only a complete, digest-verified object is
// renamed to dst.
//
// A partial is continued only when the remote object is proven to be the one
// the prefix came from: the request carries Range and If-Range with the strong
// validator (ETag, else Last-Modified) recorded with the prefix, and the 206
// answer must start at the partial's length, state the recorded total length
// and carry no different validator. Anything else (no validator recorded, a
// server that ignores Range, a changed validator or length, a partial longer
// than the object) restarts the object from byte 0 explicitly; uncertain bytes
// are never concatenated. The pinned SHA-256 stays the only authority over
// what is published: a resumed object that fails it is discarded and fetched
// once more from the start.

// partialSchema identifies the partial-download record.
const partialSchema = "hachidori.partial-download.v1"

// partialMeta is "<dst>.part.json".
type partialMeta struct {
	Schema       string `json:"schema"`
	URL          string `json:"url"`
	SHA256       string `json:"sha256"` // the pinned digest of the complete object
	ETag         string `json:"etag,omitempty"`
	LastModified string `json:"last_modified,omitempty"`
	Total        int64  `json:"total"` // length of the complete object
}

// validator is the strongest identity of the remote object the server stated:
// a strong ETag, else Last-Modified. A weak ETag proves nothing byte-wise and
// is not a validator.
func validator(h http.Header) (etag, lastModified string) {
	if e := strings.TrimSpace(h.Get("ETag")); e != "" && !strings.HasPrefix(e, "W/") {
		etag = e
	}
	return etag, strings.TrimSpace(h.Get("Last-Modified"))
}

func (m partialMeta) ifRange() string {
	if m.ETag != "" {
		return m.ETag
	}
	return m.LastModified
}

// sameObject reports whether a response's validators do not contradict the
// recorded ones. The recorded validator must be present in the response when
// the response states one of that kind.
func (m partialMeta) sameObject(h http.Header) bool {
	etag, lm := validator(h)
	if m.ETag != "" {
		return etag == m.ETag
	}
	return m.LastModified != "" && lm == m.LastModified
}

var contentRangeRe = regexp.MustCompile(`^bytes (\d+)-(\d+)/(\d+)$`)

// parseContentRange reads "bytes first-last/total".
func parseContentRange(v string) (first, last, total int64, ok bool) {
	m := contentRangeRe.FindStringSubmatch(strings.TrimSpace(v))
	if m == nil {
		return 0, 0, 0, false
	}
	first, _ = strconv.ParseInt(m[1], 10, 64)
	last, _ = strconv.ParseInt(m[2], 10, 64)
	total, _ = strconv.ParseInt(m[3], 10, 64)
	return first, last, total, last >= first && total > last
}

// PartialPaths are the files of the partial of dst.
func partialPaths(dst string) (part, meta string) { return dst + ".part", dst + ".part.json" }

func discardPartial(dst string) {
	part, meta := partialPaths(dst)
	_ = os.Remove(part)
	_ = os.Remove(meta)
}

// PartialState is what an interrupted download of one file left behind.
type PartialState struct {
	Bytes int64 // bytes held
	Total int64 // length of the complete object (0 when unknown)
}

// loadPartial reads the partial of dst for url and want. It returns the
// record and the held length when the partial may be continued; otherwise the
// reason it may not ("" reason with ok=false means there is none).
func loadPartial(dst, url, want string) (meta partialMeta, size int64, reason string, ok bool) {
	part, metaPath := partialPaths(dst)
	fi, err := os.Stat(part)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			_ = os.Remove(metaPath)
			return meta, 0, "", false
		}
		return meta, 0, err.Error(), false
	}
	if !fi.Mode().IsRegular() {
		return meta, 0, "the partial is not a regular file", false
	}
	if fi.Size() == 0 {
		return meta, 0, "", false
	}
	if err := home.ReadJSON(metaPath, &meta); err != nil {
		return meta, 0, "no readable resume record", false
	}
	switch {
	case meta.Schema != partialSchema:
		return meta, 0, "unknown resume record schema " + meta.Schema, false
	case meta.URL != url || meta.SHA256 != want:
		return meta, 0, "the resume record is for another object", false
	case meta.Total <= 0 || (meta.ETag == "" && meta.LastModified == ""):
		return meta, 0, "the remote object's identity was not recorded", false
	case fi.Size() > meta.Total:
		return meta, 0, fmt.Sprintf("the partial (%d bytes) is longer than the object (%d bytes)", fi.Size(), meta.Total), false
	}
	return meta, fi.Size(), "", true
}

// InspectPartial reports the resumable partial of dst for the pinned object,
// if a continuable one exists. It reads only the partial's record.
func InspectPartial(dst, url, want string) (PartialState, bool) {
	meta, size, _, ok := loadPartial(dst, url, want)
	if !ok {
		return PartialState{}, false
	}
	return PartialState{Bytes: size, Total: meta.Total}, true
}

// errRestart makes the caller start the object again without a partial.
type errRestart struct{ why string }

func (e errRestart) Error() string { return e.why }

// fetch downloads url to dst and verifies its SHA-256. A present file with
// the right digest is reused. The digest stays the only authority over what
// is accepted; a download that stalls or fails leaves nothing behind. It is
// the whole-object download of small artifacts; the large model files use
// fetchResumable.
func fetch(url, dst, want string, log io.Writer, obs *Observer, at Progress) error {
	return fetchWith(context.Background(), url, dst, want, log, obs, at, false)
}

// fetchResumable is fetch for large objects: an interrupted download keeps a
// partial that a later call continues once the remote object is proven to be
// the same. ctx cancels the transfer; a cancelled transfer keeps its partial
// exactly as an interrupted one does.
func fetchResumable(ctx context.Context, url, dst, want string, log io.Writer, obs *Observer, at Progress) error {
	return fetchWith(ctx, url, dst, want, log, obs, at, true)
}

func fetchWith(ctx context.Context, url, dst, want string, log io.Writer, obs *Observer, at Progress, resumable bool) error {
	if got, err := fileSHA256Observed(dst, obs, Progress{Step: StepVerify, Detail: at.Detail, Item: at.Item, Items: at.Items}); err == nil && got == want {
		discardPartial(dst)
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	usePartial := resumable
	for {
		err := downloadOnce(ctx, url, dst, want, log, obs, at, resumable, usePartial)
		var r errRestart
		if errors.As(err, &r) && usePartial {
			fmt.Fprintf(log, "restarting %s from byte 0: %s\n", url, r.why)
			usePartial = false
			continue
		}
		return err
	}
}

// downloadOnce runs one transfer. With usePartial it first tries to continue
// the partial of dst. With keep, an interruption leaves a resumable partial
// when the remote identity is known; without it nothing is left behind.
func downloadOnce(ctx context.Context, url, dst, want string, log io.Writer, obs *Observer, at Progress, keep, usePartial bool) error {
	part, metaPath := partialPaths(dst)
	var (
		meta    partialMeta
		offset  int64
		resumed bool
	)
	if usePartial {
		m, size, reason, ok := loadPartial(dst, url, want)
		switch {
		case ok:
			meta, offset, resumed = m, size, true
		case reason != "":
			fmt.Fprintf(log, "discarding the partial download of %s: %s\n", url, reason)
			discardPartial(dst)
		}
	}
	if resumed && offset == meta.Total {
		// Everything is already here: only the digest decides.
		got, err := fileSHA256Observed(part, obs, Progress{Step: StepVerify, Detail: at.Detail, Item: at.Item, Items: at.Items})
		if err == nil && got == want {
			discardPartialMeta(dst)
			return os.Rename(part, dst)
		}
		discardPartial(dst)
		return errRestart{"the complete partial does not match the pinned digest"}
	}

	// The held prefix is hashed before any request is made: it can take a
	// while for a multi-GB partial, and neither the stall watchdog nor an idle
	// connection should be waiting on it.
	hsh := sha256.New()
	if resumed {
		if err := hashPrefix(hsh, part, offset, obs, Progress{Step: StepVerify, Detail: at.Detail + " (partial)", Item: at.Item, Items: at.Items}); err != nil {
			discardPartial(dst)
			return errRestart{"the partial could not be read back: " + err.Error()}
		}
	}

	cctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var stalledFlag atomic.Bool
	watchdog := time.AfterFunc(downloadStall, func() { stalledFlag.Store(true); cancel() })
	defer watchdog.Stop()
	interrupted := func(err error) error {
		switch {
		case stalledFlag.Load():
			return fmt.Errorf("GET %s: stalled, no data received for %s", url, downloadStall)
		case ctx.Err() != nil:
			return fmt.Errorf("GET %s: %w", url, ctx.Err())
		}
		return err
	}

	req, err := http.NewRequestWithContext(cctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if keep {
		// Byte offsets are only meaningful over the identity encoding.
		req.Header.Set("Accept-Encoding", "identity")
	}
	if resumed {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
		req.Header.Set("If-Range", meta.ifRange())
		fmt.Fprintf(log, "resuming %s at byte %d of %d\n", url, offset, meta.Total)
	} else {
		fmt.Fprintf(log, "downloading %s\n", url)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return interrupted(err)
	}
	defer resp.Body.Close()

	switch {
	case resumed && resp.StatusCode == http.StatusPartialContent:
		first, last, total, ok := parseContentRange(resp.Header.Get("Content-Range"))
		switch {
		case !ok:
			discardPartial(dst)
			return errRestart{"the server's Content-Range is not a byte range with a total length"}
		case first != offset:
			discardPartial(dst)
			return errRestart{fmt.Sprintf("the server answered from byte %d, the partial ends at %d", first, offset)}
		case total != meta.Total:
			discardPartial(dst)
			return errRestart{fmt.Sprintf("the object length changed (%d bytes recorded, %d now)", meta.Total, total)}
		case last != total-1:
			discardPartial(dst)
			return errRestart{"the server answered a partial range, not the rest of the object"}
		case !meta.sameObject(resp.Header):
			discardPartial(dst)
			return errRestart{"the object's validator (ETag/Last-Modified) changed"}
		}
	case resumed && resp.StatusCode == http.StatusRequestedRangeNotSatisfiable:
		discardPartial(dst)
		return errRestart{"the server cannot satisfy the range (the partial no longer fits the object)"}
	case resp.StatusCode == http.StatusOK:
		if resumed {
			// The server ignored Range, or If-Range failed because the object
			// changed: the body is the whole object. The old prefix is not
			// appended to; the object starts over.
			fmt.Fprintf(log, "the server returned the full object instead of the range; restarting %s from byte 0\n", url)
			discardPartial(dst)
		}
		resumed, offset = false, 0
		meta = partialMeta{}
		hsh.Reset()
	default:
		// The partial, if any, is untouched: nothing was learned about it.
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}

	total := int64(0)
	if resumed {
		total = meta.Total
	} else {
		meta = partialMeta{Schema: partialSchema, URL: url, SHA256: want, Total: max(resp.ContentLength, 0)}
		meta.ETag, meta.LastModified = validator(resp.Header)
		total = meta.Total
	}
	// A partial is worth keeping only if the object can be identified again.
	persist := keep && meta.Total > 0 && (meta.ETag != "" || meta.LastModified != "")
	flags := os.O_CREATE | os.O_WRONLY
	if resumed {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
		_ = os.Remove(metaPath)
	}
	f, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return err
	}
	if persist && !resumed {
		if err := home.WriteFileAtomic(metaPath, mustJSON(meta), 0o644); err != nil {
			f.Close()
			os.Remove(part)
			return err
		}
	}
	// Done counts the bytes already held, so a resumed object reports its
	// real position; Total is the whole object when the server stated it.
	at.Total, at.Resumed = total, offset
	counter := newByteCounter(obs, at)
	counter.done = offset
	counter.flush()
	_, err = io.Copy(io.MultiWriter(f, hsh, counter), progressReader{resp.Body, watchdog})
	if err == nil {
		counter.flush()
	}
	if err == nil && persist {
		err = f.Sync()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	held := counter.done
	if err == nil && total > 0 && held != total {
		err = fmt.Errorf("GET %s: the body ended after %d of %d bytes", url, held, total)
	}
	if err != nil {
		if persist {
			// Every byte in the file was received in order: it is a safe
			// prefix of the object. It is continued only after the remote
			// identity is proven again.
			fmt.Fprintf(log, "download of %s interrupted at byte %d; the partial is kept for resume\n", url, held)
		} else {
			discardPartial(dst)
		}
		return interrupted(err)
	}
	if got := hex.EncodeToString(hsh.Sum(nil)); got != want {
		discardPartial(dst)
		if offset > 0 {
			return errRestart{fmt.Sprintf("the resumed object does not match the pinned digest (sha256 %s, want %s)", got, want)}
		}
		return fmt.Errorf("%s: sha256 mismatch: got %s want %s", url, got, want)
	}
	discardPartialMeta(dst)
	return os.Rename(part, dst)
}

func discardPartialMeta(dst string) { _, m := partialPaths(dst); _ = os.Remove(m) }

func mustJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

// hashPrefix feeds the first n bytes of path into h, reporting determinate
// byte progress.
func hashPrefix(h hash.Hash, path string, n int64, obs *Observer, at Progress) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	at.Total = n
	c := newByteCounter(obs, at)
	if copied, err := io.Copy(io.MultiWriter(h, c), io.LimitReader(f, n)); err != nil {
		return err
	} else if copied != n {
		return fmt.Errorf("the partial holds %d bytes, expected %d", copied, n)
	}
	c.flush()
	return nil
}
