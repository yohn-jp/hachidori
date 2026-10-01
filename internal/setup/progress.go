package setup

import (
	"io"
	"sync"
	"time"
)

// Step is the kind of work a setup operation is doing inside a Phase. The
// phase says where in the operation it is; the step says what it is busy with.
type Step string

const (
	StepDownload    Step = "downloading"   // fetching a pinned artifact
	StepVerify      Step = "verifying"     // checking digests or the private interpreter
	StepMaterialize Step = "materializing" // the private uv building the runtime
	StepPublish     Step = "publishing"    // atomically publishing a verified artifact
	StepActivate    Step = "activating"    // replacing the activation record
	StepRemove      Step = "removing"      // deleting an unused artifact
)

// Progress is a real report of work inside the current phase. Total is zero
// when the amount of work is not known: the step is then indeterminate and no
// percentage may be derived from it. Done and Total count bytes when set.
type Progress struct {
	Step   Step   `json:"step"`
	Detail string `json:"detail,omitempty"` // what is being worked on: a file, a uv subcommand
	Done   int64  `json:"done,omitempty"`
	Total  int64  `json:"total,omitempty"`
	// Item and Items place Detail among a known number of files (1-based);
	// both are zero when the step is not part of a sequence.
	Item  int `json:"item,omitempty"`
	Items int `json:"items,omitempty"`
}

// Determinate reports whether Done/Total is a meaningful fraction.
func (p Progress) Determinate() bool { return p.Total > 0 && p.Done >= 0 }

// Fraction is Done/Total clamped to [0, 1]; it is 0 for an indeterminate step.
func (p Progress) Fraction() float64 {
	if !p.Determinate() {
		return 0
	}
	return max(0, min(1, float64(p.Done)/float64(p.Total)))
}

// Observer receives the real boundaries and progress of a setup operation.
// A nil *Observer, and nil callbacks, observe nothing. Callbacks run on the
// goroutine doing the work and must not block.
type Observer struct {
	OnPhase    func(Phase)
	OnProgress func(Progress)
}

func (o *Observer) phase(p Phase) {
	if o != nil && o.OnPhase != nil {
		o.OnPhase(p)
	}
}

func (o *Observer) progress(p Progress) {
	if o != nil && o.OnProgress != nil {
		o.OnProgress(p)
	}
}

// step reports an indeterminate step: work whose total is not measurable.
func (o *Observer) step(s Step, detail string) { o.progress(Progress{Step: s, Detail: detail}) }

// progressInterval bounds how often byte progress is reported. Transfers
// read in small chunks; the observer sees a steady, bounded stream and always
// the final position.
var progressInterval = 150 * time.Millisecond

// byteCounter is an io.Writer that reports bytes written as progress.
type byteCounter struct {
	obs   *Observer
	base  Progress // Step, Detail, Total, Item, Items; Done is overwritten
	mu    sync.Mutex
	done  int64
	last  time.Time
	every time.Duration
}

func newByteCounter(obs *Observer, base Progress) *byteCounter {
	return &byteCounter{obs: obs, base: base, every: progressInterval}
}

func (c *byteCounter) Write(b []byte) (int, error) {
	c.mu.Lock()
	c.done += int64(len(b))
	report := c.every <= 0 || time.Since(c.last) >= c.every
	if report {
		c.last = time.Now()
	}
	p := c.base
	p.Done = c.done
	c.mu.Unlock()
	if report {
		c.obs.progress(p)
	}
	return len(b), nil
}

// flush reports the final position.
func (c *byteCounter) flush() {
	c.mu.Lock()
	p := c.base
	p.Done = c.done
	c.mu.Unlock()
	c.obs.progress(p)
}

var _ io.Writer = (*byteCounter)(nil)
