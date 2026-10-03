package app

import (
	"context"
	"errors"
	"path/filepath"
	"time"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/setup"
)

// The device capacity Tuning and Forge plan against must not depend on a model
// being resident. While a serving worker runs it reports the accelerator
// itself; otherwise (serving stopped, or paused for model engineering) the
// controller observes the device once with the active runtime's private torch:
// no worker, no model load. The observation is host/device capacity, never a
// measurement of a candidate: a candidate's memory stays MEASURED only when a
// serving worker or a probe actually measured it.

// ErrNoAccelerator is the observation of a home whose active runtime serves on
// the cpu: there is no accelerator capacity to observe.
var ErrNoAccelerator = errors.New("the active runtime does not use an accelerator")

// accelRetry is how long a failed observation stands before it is repeated.
const accelRetry = time.Minute

// AcceleratorObservation is the device capacity observed without a worker.
type AcceleratorObservation struct {
	// Pending: the observation is being made now (or has not yet been made).
	Pending bool
	// Name and TotalBytes are the device as torch reports it; zero when the
	// device could not be observed (Err says why) or there is none.
	Name       string
	TotalBytes uint64
	Err        string
}

type accelCache struct {
	root    string
	running bool
	have    bool
	at      time.Time
	obs     AcceleratorObservation
}

// Accelerator is the device capacity under the selected home, observed without
// a serving worker. It never starts a worker, never loads a model and never
// observes while a worker serves (the worker reports the accelerator itself) or
// while model engineering owns the device. The observation runs in the
// background; until it finishes the result is Pending.
func (c *Controller) Accelerator() AcceleratorObservation {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.observeAcceleratorLocked()
	if c.accel.have && c.accel.root == c.home {
		return c.accel.obs
	}
	return AcceleratorObservation{Pending: true}
}

// observeAcceleratorLocked starts the background observation unless one is
// running, a current one exists, or the accelerator is in use by serving or by
// model engineering; c.mu must be held.
func (c *Controller) observeAcceleratorLocked() {
	if c.home == "" || c.closed || c.accel.running || c.eng != nil || (c.rt != nil && c.rt.Running()) {
		return
	}
	if c.accel.have && c.accel.root == c.home && (c.accel.obs.Err == "" || time.Since(c.accel.at) < accelRetry) {
		return
	}
	root := c.home
	c.accel.running = true
	go func() {
		f, err := c.cfg.Accelerator(c.bg, root)
		obs := AcceleratorObservation{}
		switch {
		case err != nil:
			obs.Err = err.Error()
		case f.Error != "":
			obs.Err = f.Error
		case !f.CUDAAvailable:
			obs.Err = "torch reports no usable CUDA device"
		default:
			obs.Name, obs.TotalBytes = f.DeviceName, f.VRAMTotal
		}
		c.mu.Lock()
		defer c.mu.Unlock()
		c.accel.running = false
		if c.home != root {
			return
		}
		c.accel.root, c.accel.have, c.accel.at, c.accel.obs = root, true, time.Now(), obs
		c.notify()
	}()
}

// observeAccelerator is the default observation: the active runtime's private
// interpreter asks torch for the device. A cpu activation has nothing to
// observe.
func observeAccelerator(ctx context.Context, root string) (setup.AcceleratorFacts, error) {
	h := home.Home{Root: root}
	a, rm, _, err := h.LoadActive()
	if err != nil {
		return setup.AcceleratorFacts{}, err
	}
	if a.Device != "cuda" {
		return setup.AcceleratorFacts{}, ErrNoAccelerator
	}
	return setup.ProbeAccelerator(ctx, h, filepath.Join(h.Path("runtime", a.Runtime), filepath.FromSlash(rm.PythonRelPath)))
}
