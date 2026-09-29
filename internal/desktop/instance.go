package desktop

import (
	"crypto/sha256"
	"encoding/hex"
)

// InstanceName is the name of the per-user single-instance object for the
// user identified by userID (on Windows, the user's SID string).
//
// It lives in the session-local ("Local\") kernel namespace, which needs no
// privilege, and embeds a digest of the user identity so two users (or a
// "run as" of another account in the same session) never share it.
func InstanceName(userID string) string {
	sum := sha256.Sum256([]byte("hachidori-desktop\x00" + userID))
	return `Local\Hachidori.Desktop.` + hex.EncodeToString(sum[:12])
}
