//go:build windows

package subprocess

import (
	"os/exec"
	"syscall"
	"testing"
)

func TestConfigureHidesOnlyTheChildWindow(t *testing.T) {
	cmd := exec.Command("child")
	cmd.Env = []string{"HACHIDORI_TEST=1"}
	cmd.Dir = `C:\Hachidori`
	Configure(cmd)
	if cmd.SysProcAttr == nil || !cmd.SysProcAttr.HideWindow {
		t.Fatal("Configure did not enable HideWindow")
	}
	if len(cmd.Env) != 1 || cmd.Env[0] != "HACHIDORI_TEST=1" || cmd.Dir != `C:\Hachidori` {
		t.Fatalf("Configure changed command settings: env=%v dir=%q", cmd.Env, cmd.Dir)
	}

	cmd.SysProcAttr = &syscall.SysProcAttr{}
	Configure(cmd)
	if !cmd.SysProcAttr.HideWindow {
		t.Fatal("Configure did not update an existing process policy")
	}
}
