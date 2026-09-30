//go:build windows

package subprocess

import (
	"errors"
	"os/exec"
	"strings"
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

	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP}
	Configure(cmd)
	if !cmd.SysProcAttr.HideWindow || cmd.SysProcAttr.CreationFlags != syscall.CREATE_NEW_PROCESS_GROUP {
		t.Fatalf("Configure did not preserve an existing process policy: %+v", cmd.SysProcAttr)
	}
}

func TestConfiguredChildKeepsPipesEnvironmentAndExitStatus(t *testing.T) {
	cmd := exec.Command("cmd.exe", "/d", "/c", "echo %HACHIDORI_TEST%& echo err 1>&2& exit /b 7")
	cmd.Env = []string{"HACHIDORI_TEST=piped", `SystemRoot=C:\Windows`}
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	Configure(cmd)
	err := cmd.Run()
	var exit *exec.ExitError
	if !errors.As(err, &exit) || exit.ExitCode() != 7 {
		t.Fatalf("exit = %v, want status 7", err)
	}
	if got := strings.TrimSpace(stdout.String()); got != "piped" {
		t.Errorf("stdout = %q", got)
	}
	if got := strings.TrimSpace(stderr.String()); got != "err" {
		t.Errorf("stderr = %q", got)
	}
}
