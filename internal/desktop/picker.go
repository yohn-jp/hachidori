package desktop

import (
	"context"
	"errors"
)

// ErrPickCancelled is returned when the user dismisses a native picker without
// choosing a path.
var ErrPickCancelled = errors.New("folder selection cancelled")

// FolderPicker is the native folder-selection surface. It returns a
// filesystem path and nothing else: the caller validates the path, and the
// picker grants no read or write authority beyond that one string.
type FolderPicker interface {
	// PickFolder shows the operating system's folder chooser and blocks until
	// the user chooses or cancels (ErrPickCancelled), or ctx is done.
	PickFolder(ctx context.Context, title string) (string, error)
}

// PathPicker extends the first-run folder picker with the bounded file
// choices used by the workstation. It returns one path and grants no general
// filesystem authority to the dashboard.
type PathPicker interface {
	FolderPicker
	PickOpen(ctx context.Context, title string) (string, error)
	PickSave(ctx context.Context, title string) (string, error)
}
