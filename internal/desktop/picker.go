package desktop

import (
	"context"
	"errors"
)

// ErrPickCancelled is returned by a FolderPicker when the user dismissed the
// dialog without choosing a folder.
var ErrPickCancelled = errors.New("folder selection cancelled")

// FolderPicker is the native folder-selection surface. It returns a
// filesystem path and nothing else: the caller validates the path, and the
// picker grants no read or write authority beyond that one string.
type FolderPicker interface {
	// PickFolder shows the operating system's folder chooser and blocks until
	// the user chooses or cancels (ErrPickCancelled), or ctx is done.
	PickFolder(ctx context.Context, title string) (string, error)
}
