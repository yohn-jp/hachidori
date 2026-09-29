package desktop

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

const instancePrefix = `Local\Hachidori.Desktop.`

// InstanceName is the name of the per-user single-instance object for the
// user identified by userID (on Windows, the user's SID string).
//
// It lives in the session-local ("Local\") kernel namespace, which needs no
// privilege, and embeds a digest of the user identity so two users (or a
// "run as" of another account in the same session) never share it.
func InstanceName(userID string) string {
	sum := sha256.Sum256([]byte("hachidori-desktop\x00" + userID))
	return instancePrefix + hex.EncodeToString(sum[:12])
}

// WindowClassName is the per-user window class of the desktop shell's main
// window. A second launch finds the running shell's window by this name, so it
// must embed the same per-user digest as InstanceName: another user's shell in
// the same session is never activated.
func WindowClassName(userID string) string {
	return "Hachidori.Desktop.Window." + strings.TrimPrefix(InstanceName(userID), instancePrefix)
}

// ActivateMessageName is the registered window message a second launch posts
// to the running shell's window to make it show and focus itself.
func ActivateMessageName(userID string) string {
	return "Hachidori.Desktop.Activate." + strings.TrimPrefix(InstanceName(userID), instancePrefix)
}
