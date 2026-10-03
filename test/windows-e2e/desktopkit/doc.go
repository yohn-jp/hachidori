// Package desktopkit is the helper layer the bootstrap and single-instance
// Windows E2E shards share to drive the real, packaged hachidori.exe as a
// subprocess inside a disposable user profile.
//
// Everything here is deliberately small, deterministic and portable where it
// can be: path shapes, disposable profile environments, bounded process output,
// observable-state polling, bootstrap-locator fixtures, the loopback first-run
// and dashboard clients, netstat parsing and the installed-home fixture are
// ordinary Go that its own tests exercise on any OS. Only the process and
// socket observation that needs Windows (process snapshots, process-tree
// termination, netstat) lives behind build tags.
//
// The package never builds hachidori.exe and never touches a real Hachidori
// home or user profile: every child process runs with LOCALAPPDATA, APPDATA,
// USERPROFILE and TEMP pointed into a directory the scenario created.
package desktopkit
