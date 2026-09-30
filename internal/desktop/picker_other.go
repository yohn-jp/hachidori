//go:build !windows

package desktop

import "context"

// NativePicker returns this OS's folder picker. Outside Windows it always
// fails with ErrUnsupported; no windowing dependency is linked.
func NativePicker() FolderPicker { return unsupportedPicker{} }

type unsupportedPicker struct{}

func (unsupportedPicker) PickFolder(context.Context, string) (string, error) {
	return "", ErrUnsupported
}

func (unsupportedPicker) PickOpen(context.Context, string) (string, error) {
	return "", ErrUnsupported
}

func (unsupportedPicker) PickSave(context.Context, string) (string, error) {
	return "", ErrUnsupported
}
