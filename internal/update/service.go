package update

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
)

// Phases and steps of a download, in the vocabulary of the long-running
// operation presentation (internal/dashboard/ops.go).
const (
	PhaseReleases = "releases"
	PhaseChecksum = "checksum"
	PhaseDownload = "download"
	PhaseVerify   = "verify"

	StepFetching    = "fetching"
	StepDownloading = "downloading"
	StepVerifying   = "verifying"

	KindCheck    = "check"
	KindDownload = "update"
)

var (
	checkPlan    = []string{PhaseReleases}
	downloadPlan = []string{PhaseChecksum, PhaseDownload, PhaseVerify}
)

// Time bounds. The download bound is on progress, never on total time.
const (
	checkTimeout  = 45 * time.Second
	sumTimeout    = 30 * time.Second
	downloadStall = 2 * time.Minute
	userAgent     = "hachidori-update"
)

// Failure is why an operation ended unsuccessfully.
type Failure struct {
	Class   Class
	Phase   string
	Step    string
	Message string
}

// Operation is the check or download in flight, or the last download
// finished. It carries the same fields as app.Operation's phases and
// progress, and the dashboard presents it with the same long-running operation
// view. At most one exists at a time: it is the single authority for whether
// an update action is already in progress.
type Operation struct {
	Kind              string
	Tag               string
	Plan, Phases      []string
	Phase             string
	Step, Detail      string
	Done, Total       int64
	Started, Finished time.Time
	Failure           *Failure
}

func (o *Operation) clone() *Operation {
	if o == nil {
		return nil
	}
	c := *o
	c.Plan = slices.Clone(o.Plan)
	c.Phases = slices.Clone(o.Phases)
	if o.Failure != nil {
		f := *o.Failure
		c.Failure = &f
	}
	return &c
}

// CheckResult is what one explicit check found.
type CheckResult struct {
	Time      time.Time
	Channel   Channel
	Releases  []Candidate // eligible releases, newest first
	Installed InstalledView
}

// InstalledView is the installed executable's identity as far as it is known.
type InstalledView struct {
	Version string // "" when unknown
	Known   bool
	SHA256  string // digest of the running executable, "" when it could not be read
}

// Status is the update subsystem's local state. Reading it never uses the
// network.
type Status struct {
	// Supported is false when Restart & update is not possible here.
	Supported bool
	Channel   Channel
	Installed InstalledView
	LastCheck *LastCheck
	// Check is the result of the last explicit check in this process; nil until
	// one was made, and cleared by a channel change or a failed check.
	Check *CheckResult
	// Busy is the check or download in flight (nil when idle); every other
	// update action is refused while it is set. Last is the last download
	// finished.
	Busy *Operation
	Last *Operation
	// Ready is a downloaded, verified executable; ReadyProblem is why it cannot
	// be installed right now ("" when it can).
	Ready        *Ready
	ReadyProblem string
	// Result is the helper's report of the last replacement attempt.
	Result *Result
	Err    string
}

// Service is the update subsystem. It is created once by the composition; it
// does nothing until an explicit action calls it.
type Service struct {
	// Home reports the current HACHIDORI_HOME ("" while none is selected).
	Home func() string
	// Exe is the running executable, the destination of an update.
	Exe string
	// Settings stores the channel, the last check and the installed identity.
	Settings Store
	// Transport carries every request; nil is http.DefaultTransport. Redirects
	// are policed by the service whatever the transport is. The authority URLs
	// are fixed in this package; the transport only decides how they are
	// reached (tests route them to a local fixture).
	Transport http.RoundTripper
	// StartHelper starts the replacement helper detached; nil means Restart &
	// update is unavailable.
	StartHelper func(exe string, args []string) error
	// RestartHome, when set, is passed so the update restarts as
	// "desktop --home <it>" (the home was explicit, not discoverable).
	RestartHome string
	// Now is the clock; nil is time.Now.
	Now func() time.Time

	mu       sync.Mutex
	check    *CheckResult
	op, last *Operation
	exeSum   string
	exeRead  bool
	persist  string // last settings write error
}

func (s *Service) now() time.Time {
	if s.Now != nil {
		return s.Now()
	}
	return time.Now()
}

func (s *Service) root() string {
	if s.Home == nil {
		return ""
	}
	return s.Home()
}

// Supported reports whether Restart & update can run.
func (s *Service) Supported() bool {
	return s.StartHelper != nil && s.Exe != "" && s.root() != ""
}

// ---- local state -----------------------------------------------------------

func (s *Service) settings() (Settings, error) {
	if s.Settings == nil {
		return Settings{}, nil
	}
	return s.Settings.UpdateSettings()
}

func (s *Service) modify(fn func(*Settings) error) {
	if s.Settings == nil {
		return
	}
	err := s.Settings.ModifyUpdateSettings(fn)
	s.mu.Lock()
	s.persist = ""
	if err != nil {
		s.persist = err.Error()
	}
	s.mu.Unlock()
}

// Channel is the selected channel; Stable until one is saved.
func (s *Service) channel() (Channel, error) {
	st, err := s.settings()
	if st.Channel == "" {
		return Stable, err
	}
	return st.Channel, err
}

// SetChannel saves the channel. It is local: it contacts nothing, downloads
// nothing and installs nothing; it only discards the displayed results of an
// earlier check, which were made for another channel.
func (s *Service) SetChannel(c Channel) error {
	if _, err := ParseChannel(string(c)); err != nil {
		return err
	}
	if s.Settings == nil {
		return errors.New("update settings are not available")
	}
	// The channel selects what a check offers and what may be installed, so it
	// does not change under a check or download that is running.
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.op != nil {
		return errInProgress()
	}
	if err := s.Settings.ModifyUpdateSettings(func(st *Settings) error { st.Channel = c; return nil }); err != nil {
		return err
	}
	s.check = nil
	return nil
}

// executableSum is the SHA-256 of the running executable, read once.
func (s *Service) executableSum() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.exeRead {
		s.exeRead = true
		if s.Exe != "" {
			s.exeSum, _ = FileSHA256(s.Exe)
		}
	}
	return s.exeSum
}

// installed resolves the installed release. The version is believed only for
// an executable whose digest matches the record it came from: the helper's
// result of an applied update, or an earlier explicit check that matched the
// executable to a release asset.
func (s *Service) installed() (InstalledView, *Version) {
	sum := s.executableSum()
	view := InstalledView{SHA256: sum}
	if sum == "" {
		return view, nil
	}
	known := func(tag string) (InstalledView, *Version) {
		v, err := ParseVersion(tag)
		if err != nil {
			return view, nil
		}
		view.Version, view.Known = v.String(), true
		return view, &v
	}
	if root := s.root(); root != "" {
		if r, err := LoadResult(root); err == nil && (r.Outcome == OutcomeApplied || r.Outcome == OutcomeRestart) && r.SHA256 == sum {
			return known(r.Tag)
		}
	}
	if st, _ := s.settings(); st.Installed != nil && st.Installed.SHA256 == sum {
		return known(st.Installed.Version)
	}
	return view, nil
}

// Status is the local state. It performs no network access.
func (s *Service) Status() Status {
	ch, serr := s.channel()
	st := Status{Supported: s.Supported(), Channel: ch}
	st.Installed, _ = s.installed()
	if set, _ := s.settings(); set.LastCheck != nil {
		lc := *set.LastCheck
		st.LastCheck = &lc
	}
	s.mu.Lock()
	if s.check != nil {
		c := *s.check
		c.Releases = slices.Clone(s.check.Releases)
		st.Check = &c
	}
	st.Busy, st.Last, st.Err = s.op.clone(), s.last.clone(), s.persist
	s.mu.Unlock()
	if serr != nil {
		st.Err = serr.Error()
	}
	if root := s.root(); root != "" {
		if r, _, err := s.installable(root, ch); err == nil {
			st.Ready = &r
		} else if r2, _, lerr := LoadReady(root); lerr == nil {
			st.Ready, st.ReadyProblem = &r2, err.Error()
		}
		if res, err := LoadResult(root); err == nil {
			st.Result = &res
		}
	}
	return st
}

// installable is the one rule for whether the ready update may be installed
// now: it exists, its staged file is still there, it was prepared for this
// executable, the selected channel offers it, and it is newer than the
// installed release.
func (s *Service) installable(root string, ch Channel) (Ready, Version, error) {
	r, v, err := LoadReady(root)
	if err != nil {
		return Ready{}, Version{}, err
	}
	if fi, err := os.Lstat(StagedPath(root, v)); err != nil || !fi.Mode().IsRegular() || fi.Size() != r.Size {
		return Ready{}, Version{}, errors.New("the downloaded file is missing or changed; download it again")
	}
	if s.Exe == "" || !samePath(r.Target, s.Exe) {
		return Ready{}, Version{}, fmt.Errorf("the update was prepared for %s, not this executable", r.Target)
	}
	if !ch.Eligible(Release{Version: v, Prerelease: r.Prerelease}) {
		return Ready{}, Version{}, fmt.Errorf("release %s is not offered on the %s channel; check again or switch back", r.Tag, ch)
	}
	if _, iv := s.installed(); iv != nil && v.Compare(*iv) <= 0 {
		return Ready{}, Version{}, fmt.Errorf("release %s is not newer than the installed %s; an update never goes backwards", r.Tag, iv)
	}
	return r, v, nil
}

// ---- HTTP ------------------------------------------------------------------

// downloadHosts are where a release asset download may redirect to: GitHub's
// own release-asset storage. Nothing else is followed.
var downloadHosts = []string{
	"objects.githubusercontent.com",
	"release-assets.githubusercontent.com",
	"github-releases.githubusercontent.com",
}

const maxRedirects = 3

// redirectPolicy limits redirects of an asset download to https, no
// credentials, the standard port and GitHub's release-asset hosts.
func redirectPolicy(req *http.Request, via []*http.Request) error {
	if len(via) > maxRedirects {
		return fmt.Errorf("stopped after %d redirects", maxRedirects)
	}
	u := req.URL
	switch {
	case u.Scheme != "https":
		return fmt.Errorf("redirect to a non-https location refused")
	case u.User != nil:
		return fmt.Errorf("redirect with credentials refused")
	case u.Port() != "" && u.Port() != "443":
		return fmt.Errorf("redirect to port %s refused", u.Port())
	case !slices.Contains(downloadHosts, strings.ToLower(u.Hostname())):
		return fmt.Errorf("redirect to %s refused: not a GitHub release-asset host", u.Hostname())
	}
	return nil
}

func noRedirect(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }

// get requests one authority URL. follow permits the asset-host redirects a
// release download needs; metadata requests follow none.
func (s *Service) get(ctx context.Context, rawurl, accept string, follow bool) (*http.Response, error) {
	u, err := url.Parse(rawurl)
	if err != nil || u.Scheme != "https" || (u.Host != apiHost && u.Host != downloadHost) {
		return nil, fmt.Errorf("refusing to request %q: not the official release authority", rawurl)
	}
	t := s.Transport
	if t == nil {
		t = http.DefaultTransport
	}
	c := &http.Client{Transport: t, CheckRedirect: noRedirect}
	if follow {
		c.CheckRedirect = redirectPolicy
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawurl, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", accept)
	req.Header.Set("User-Agent", userAgent)
	if u.Host == apiHost {
		req.Header.Set("X-GitHub-Api-Version", "2022-11-28")
	}
	return c.Do(req)
}

// readAll reads at most max bytes; a longer body is an error.
func readAll(r io.Reader, max int64) ([]byte, error) {
	b, err := io.ReadAll(io.LimitReader(r, max+1))
	if err == nil && int64(len(b)) > max {
		err = fmt.Errorf("response is larger than %d bytes", max)
	}
	return b, err
}

// ---- Check -----------------------------------------------------------------

// errInProgress is the refusal of every action while a check or download runs.
func errInProgress() error {
	return &Error{Class: ClassRefused, Msg: "another update action is in progress"}
}

// beginCheck accepts a check: it becomes the operation in flight, visible in
// Status from this moment, and discards the results of the previous check. It
// refuses while a check or download is running.
func (s *Service) beginCheck() (*Operation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.op != nil {
		return nil, errInProgress()
	}
	op := &Operation{Kind: KindCheck, Plan: slices.Clone(checkPlan), Started: s.now()}
	s.op, s.check = op, nil
	return op, nil
}

// StartCheck accepts a check and returns at once; the check runs in the
// background and its progress and outcome are read from Status. It is how the
// desktop starts one, so the operator sees the action accepted before the
// network answers.
func (s *Service) StartCheck() error {
	op, err := s.beginCheck()
	if err != nil {
		return err
	}
	go func() { _ = s.runCheck(context.Background(), op) }()
	return nil
}

// Check retrieves the repository's release list from the fixed authority and
// presents the releases the selected channel offers, and returns when it has
// finished. It is the only action that retrieves release metadata, and it runs
// only when called.
func (s *Service) Check(ctx context.Context) error {
	op, err := s.beginCheck()
	if err != nil {
		return err
	}
	return s.runCheck(ctx, op)
}

func (s *Service) runCheck(ctx context.Context, op *Operation) error {
	defer func() {
		s.mu.Lock()
		op.Finished, s.op = s.now(), nil
		s.mu.Unlock()
	}()
	s.phase(op, PhaseReleases)
	s.progress(op, StepFetching, "release list", 0, 0)

	ch, _ := s.channel()
	ctx, cancel := context.WithTimeout(ctx, checkTimeout)
	defer cancel()
	rels, err := s.fetchReleases(ctx, op)
	if err != nil {
		lc := LastCheck{Time: s.now(), Channel: ch, Class: ClassNetwork, Message: err.Error()}
		var e *Error
		if errors.As(err, &e) {
			lc.Class = e.Class
		}
		s.record(lc)
		return err
	}
	view, iv := s.installed()
	if iv == nil && view.SHA256 != "" {
		// The executable carries no version of its own: identify it by the
		// digest GitHub reports for the release assets, when one matches.
		for _, r := range rels {
			if r.Problem == "" && r.Exe.SHA256 == view.SHA256 {
				v := r.Version
				iv, view.Version, view.Known = &v, v.String(), true
				s.modify(func(st *Settings) error {
					st.Installed = &Installed{Version: v.String(), SHA256: view.SHA256}
					return nil
				})
				break
			}
		}
	}
	cs := candidates(rels, ch, iv)
	if len(cs) > MaxShown {
		cs = cs[:MaxShown]
	}
	res := &CheckResult{Time: s.now(), Channel: ch, Releases: cs, Installed: view}
	lc := LastCheck{Time: res.Time, Channel: ch, OK: true}
	if len(cs) == 0 {
		lc.Message = "no eligible releases on the " + string(ch) + " channel"
	} else {
		lc.Latest = cs[0].Tag
	}
	s.record(lc)
	s.mu.Lock()
	s.check = res
	s.mu.Unlock()
	return nil
}

func (s *Service) record(lc LastCheck) {
	s.modify(func(st *Settings) error { st.LastCheck = &lc; return nil })
}

// fetchReleases reads the release list page by page. Every request is built
// from constants; a response can never name another URL.
func (s *Service) fetchReleases(ctx context.Context, op *Operation) ([]Release, error) {
	var all []Release
	seen := map[string]bool{}
	for page := 1; page <= maxPages; page++ {
		s.progress(op, StepFetching, fmt.Sprintf("release list, page %d", page), 0, 0)
		resp, err := s.get(ctx, releasesURL(page), "application/vnd.github+json", false)
		if err != nil {
			return nil, fail(ClassNetwork, err, "release metadata could not be retrieved")
		}
		body, rerr := readAll(resp.Body, maxPageSize)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			msg := "GitHub answered " + resp.Status
			if resp.StatusCode == http.StatusForbidden || resp.StatusCode == http.StatusTooManyRequests {
				msg += " (GitHub's unauthenticated request limit may be reached; try again later)"
			}
			return nil, fail(ClassNetwork, nil, "release metadata could not be retrieved: %s", msg)
		}
		if rerr != nil {
			return nil, fail(ClassNetwork, rerr, "release metadata could not be read")
		}
		rels, n, err := parseReleases(body)
		if err != nil {
			return nil, fail(ClassMalformed, err, "release metadata is malformed")
		}
		for _, r := range rels {
			if seen[r.Tag] {
				return nil, fail(ClassMalformed, nil, "release metadata is malformed: tag %s is listed twice", r.Tag)
			}
			seen[r.Tag] = true
		}
		all = append(all, rels...)
		if n < perPage {
			break
		}
	}
	return all, nil
}

// ---- Download --------------------------------------------------------------

// StartDownload begins downloading the executable of one release of the most
// recent explicit check. It returns once the download is accepted; progress
// and the outcome are read from Status. It is the only action that retrieves
// release assets.
func (s *Service) StartDownload(tag string) error {
	root := s.root()
	if root == "" {
		return &Error{Class: ClassRefused, Msg: "no Hachidori home is selected"}
	}
	s.mu.Lock()
	if s.op != nil {
		s.mu.Unlock()
		return errInProgress()
	}
	var cand *Candidate
	if s.check != nil {
		for i := range s.check.Releases {
			if s.check.Releases[i].Tag == tag {
				cand = &s.check.Releases[i]
			}
		}
	}
	switch {
	case cand == nil:
		s.mu.Unlock()
		return &Error{Class: ClassRefused, Msg: fmt.Sprintf("release %q is not in the most recent check; run Check for updates", tag)}
	case !cand.Downloadable():
		s.mu.Unlock()
		msg := fmt.Sprintf("release %s cannot be downloaded", tag)
		if cand.Problem != "" {
			msg += ": " + cand.Problem
		} else {
			msg += ": it is not newer than the installed release"
		}
		return &Error{Class: ClassRefused, Msg: msg}
	}
	c := *cand
	op := &Operation{Kind: KindDownload, Tag: tag, Plan: slices.Clone(downloadPlan), Started: s.now()}
	s.op, s.last = op, nil
	s.mu.Unlock()
	go s.download(op, root, c)
	return nil
}

// phase enters a download phase.
func (s *Service) phase(op *Operation, p string) {
	s.mu.Lock()
	op.Phase, op.Step, op.Detail, op.Done, op.Total = p, "", "", 0, 0
	op.Phases = append(op.Phases, p)
	s.mu.Unlock()
}

func (s *Service) progress(op *Operation, step, detail string, done, total int64) {
	s.mu.Lock()
	op.Step, op.Detail, op.Done, op.Total = step, detail, done, total
	s.mu.Unlock()
}

func (s *Service) endOp(op *Operation, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	op.Finished = s.now()
	if err != nil {
		f := &Failure{Class: ClassDownload, Phase: op.Phase, Step: op.Step, Message: err.Error()}
		var e *Error
		if errors.As(err, &e) {
			f.Class = e.Class
		}
		op.Failure = f
	}
	s.last, s.op = op, nil
}

func (s *Service) download(op *Operation, root string, c Candidate) {
	s.endOp(op, s.runDownload(op, root, c))
}

func (s *Service) runDownload(op *Operation, root string, c Candidate) error {
	// Phase 1: the release's checksum asset, bound to this exact tag and asset.
	s.phase(op, PhaseChecksum)
	s.progress(op, StepDownloading, SumAsset, 0, 0)
	want, err := s.fetchChecksum(c)
	if err != nil {
		return err
	}

	// Phase 2: the executable, into a staging file that is not an executable
	// path anyone launches.
	s.phase(op, PhaseDownload)
	dir := filepath.Dir(StagedPath(root, c.Version))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fail(ClassDownload, err, "the update area could not be created")
	}
	part := StagedPath(root, c.Version) + ".part"
	sum, size, err := s.fetchExecutable(op, c, part)
	if err != nil {
		os.Remove(part)
		return err
	}

	// Phase 3: verification. The digest of what was received must equal the
	// release's checksum before the file can be ready.
	s.phase(op, PhaseVerify)
	s.progress(op, StepVerifying, ExeAsset, size, size)
	if sum != want {
		os.Remove(part)
		return fail(ClassChecksumMismatch, nil, "SHA-256 mismatch: downloaded %s, release checksum %s", sum, want)
	}
	if _, err := checkExecutable(part); err != nil {
		os.Remove(part)
		return fail(ClassAsset, err, "the downloaded file is not a Windows executable")
	}
	return s.stageReady(root, c, sum, size)
}

// stageReady publishes the verified file: the previous ready record and other
// staged versions go first, then the file takes its final name and the ready
// record is written last. Only the ready record makes an update installable.
func (s *Service) stageReady(root string, c Candidate, sum string, size int64) error {
	_ = os.Remove(readyPath(root))
	clearHelpers(root) // copies left by an earlier install; a running one stays
	if ents, err := os.ReadDir(Dir(root)); err == nil {
		for _, e := range ents {
			if _, perr := ParseVersion(e.Name()); e.IsDir() && perr == nil && e.Name() != c.Version.String() {
				_ = os.RemoveAll(filepath.Join(Dir(root), e.Name()))
			}
		}
	}
	final := StagedPath(root, c.Version)
	if err := os.Rename(final+".part", final); err != nil {
		return fail(ClassDownload, err, "the verified file could not be put in place")
	}
	rec := Ready{Schema: readySchema, Tag: c.Tag, Prerelease: c.Prerelease, Asset: ExeAsset, SHA256: sum, Size: size, Target: s.Exe, Created: s.now()}
	if err := home.WriteJSON(readyPath(root), rec); err != nil {
		_ = os.Remove(final)
		return fail(ClassDownload, err, "the ready record could not be written")
	}
	return nil
}

// fetchChecksum downloads the release's checksum asset and returns the digest
// it binds to the executable asset. Metadata's own digest for the executable,
// when GitHub reports one, must agree.
func (s *Service) fetchChecksum(c Candidate) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), sumTimeout)
	defer cancel()
	resp, err := s.get(ctx, c.Sum.URL, "application/octet-stream", true)
	if err != nil {
		return "", fail(ClassVerification, err, "the checksum file of %s could not be retrieved", c.Tag)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fail(ClassVerification, nil, "the checksum file of %s could not be retrieved: %s", c.Tag, resp.Status)
	}
	body, err := readAll(resp.Body, maxSumSize)
	if err != nil {
		return "", fail(ClassVerification, err, "the checksum file of %s could not be read", c.Tag)
	}
	want, err := ParseChecksum(body, ExeAsset)
	if err != nil {
		return "", fail(ClassVerification, err, "the checksum file of %s is unusable", c.Tag)
	}
	if c.Exe.SHA256 != "" && c.Exe.SHA256 != want {
		return "", fail(ClassVerification, nil, "the checksum file of %s disagrees with the digest GitHub reports for the executable", c.Tag)
	}
	return want, nil
}

// fetchExecutable downloads the executable asset to dst, hashing it as it
// arrives, and returns its digest and size. The size must be exactly what the
// release metadata states. A transfer that receives no data for downloadStall
// is abandoned.
func (s *Service) fetchExecutable(op *Operation, c Candidate, dst string) (string, int64, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	watchdog := time.AfterFunc(downloadStall, cancel)
	defer watchdog.Stop()
	stalled := func(err error) error {
		if ctx.Err() != nil {
			return fail(ClassDownload, nil, "the download stalled: no data received for %s", downloadStall)
		}
		return fail(ClassDownload, err, "the executable of %s could not be downloaded", c.Tag)
	}
	resp, err := s.get(ctx, c.Exe.URL, "application/octet-stream", true)
	if err != nil {
		return "", 0, stalled(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", 0, fail(ClassDownload, nil, "the executable of %s could not be downloaded: %s", c.Tag, resp.Status)
	}
	if resp.ContentLength >= 0 && resp.ContentLength != c.Exe.Size {
		return "", 0, fail(ClassDownload, nil, "the server announced %d bytes, the release lists %d", resp.ContentLength, c.Exe.Size)
	}
	f, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
	if err != nil {
		return "", 0, fail(ClassDownload, err, "the staging file could not be created")
	}
	h := sha256.New()
	pr := &progressWriter{s: s, op: op, total: c.Exe.Size}
	n, err := io.Copy(io.MultiWriter(f, h, pr), stallReader{resp.Body, watchdog})
	if err == nil {
		err = f.Sync()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	switch {
	case err != nil:
		return "", 0, stalled(err)
	case n != c.Exe.Size:
		return "", 0, fail(ClassDownload, nil, "received %d bytes, the release lists %d", n, c.Exe.Size)
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

type stallReader struct {
	r io.Reader
	w *time.Timer
}

func (p stallReader) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	if n > 0 {
		p.w.Reset(downloadStall)
	}
	return n, err
}

// progressWriter reports determinate byte progress: the release metadata
// states the asset's size.
type progressWriter struct {
	s     *Service
	op    *Operation
	total int64
	done  int64
}

func (p *progressWriter) Write(b []byte) (int, error) {
	p.done += int64(len(b))
	p.s.progress(p.op, StepDownloading, ExeAsset, p.done, p.total)
	return len(b), nil
}

// ---- Install ---------------------------------------------------------------

// Install hands the verified update to a separate helper process. It returns
// as soon as the helper is started; the caller must then end the application
// so the helper can replace the executable. Nothing is replaced here: the
// running executable is not touched, and the helper acts only after the
// application has exited.
func (s *Service) Install() error {
	root := s.root()
	switch {
	case s.StartHelper == nil:
		return &Error{Class: ClassRefused, Msg: "Restart & update is available only in the Windows desktop application"}
	case root == "" || s.Exe == "":
		return &Error{Class: ClassRefused, Msg: "no Hachidori home or executable is known"}
	}
	s.mu.Lock()
	busy := s.op != nil
	s.mu.Unlock()
	if busy {
		return errInProgress()
	}
	ch, _ := s.channel()
	ready, ver, err := s.installable(root, ch)
	if err != nil {
		return &Error{Class: ClassRefused, Msg: "no update can be installed: " + err.Error()}
	}
	if err := verifyFile(StagedPath(root, ver), ready); err != nil {
		return &Error{Class: ClassRefused, Msg: "the downloaded update failed re-verification, so it was not installed: " + err.Error()}
	}
	if _, err := checkExecutable(s.Exe); err != nil {
		return &Error{Class: ClassRefused, Msg: "the running executable is not a replaceable file: " + err.Error()}
	}
	helper, err := s.prepareHelper(root)
	if err != nil {
		return fail(ClassReplace, err, "the update helper could not be prepared")
	}
	args := []string{"apply-update", "--home", root, "--pid", fmt.Sprint(os.Getpid()), "--target", s.Exe}
	if s.RestartHome != "" {
		args = append(args, "--restart-home", s.RestartHome)
	}
	if err := s.StartHelper(helper, args); err != nil {
		return fail(ClassReplace, err, "the update helper could not be started")
	}
	return nil
}

// clearHelpers removes earlier helper copies; one that is still running is
// locked by the OS and stays until a later call.
func clearHelpers(root string) {
	if ents, err := os.ReadDir(helperDir(root)); err == nil {
		for _, e := range ents {
			_ = os.Remove(filepath.Join(helperDir(root), e.Name()))
		}
	}
}

// prepareHelper copies the running executable into the update area. The copy
// is the helper: it can be started while the application runs and keeps
// running after the application's own file has been replaced.
func (s *Service) prepareHelper(root string) (string, error) {
	dir := helperDir(root)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	clearHelpers(root)
	dst := filepath.Join(dir, fmt.Sprintf("hachidori-update-%d.exe", s.now().UnixNano()))
	in, err := os.Open(s.Exe)
	if err != nil {
		return "", err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o755)
	if err != nil {
		return "", err
	}
	_, err = io.Copy(out, in)
	if err == nil {
		err = out.Sync()
	}
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(dst)
		return "", err
	}
	return dst, nil
}
