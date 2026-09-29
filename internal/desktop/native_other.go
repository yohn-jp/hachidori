//go:build !windows

package desktop

import "context"

// Native returns this OS's desktop platform. Outside Windows every method
// fails with ErrUnsupported; no windowing dependency is linked.
func Native() Platform { return unsupported{} }

type unsupported struct{}

func (unsupported) RuntimeVersion() (string, error)    { return "", ErrUnsupported }
func (unsupported) AcquireInstance() (func(), error)   { return nil, ErrUnsupported }
func (unsupported) Open(context.Context, Window) error { return ErrUnsupported }
func (unsupported) Activate() error                    { return ErrUnsupported }
func (unsupported) ReportError(string, string)         {}
