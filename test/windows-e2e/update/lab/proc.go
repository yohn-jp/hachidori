package lab

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

func psQuote(s string) string { return "'" + strings.ReplaceAll(s, "'", "''") + "'" }

func powershell(ctx context.Context, script string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script)
	return cmd.Output()
}

// killImage ends every process that is running the executable at path. The
// replacement helper restarts the executable it kept, so a scenario ends what
// it started this way before it removes its scratch folder.
func killImage(path string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	_, err := powershell(ctx, "Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq "+psQuote(path)+" } | Stop-Process -Force -ErrorAction SilentlyContinue")
	return err
}

// imageProcesses counts the processes running the executable at path.
func imageProcesses(path string) (int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	out, err := powershell(ctx, "@(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq "+psQuote(path)+" }).Count")
	if err != nil {
		return 0, err
	}
	n, err := strconv.Atoi(strings.TrimSpace(string(out)))
	if err != nil {
		return 0, fmt.Errorf("unexpected process count %q", strings.TrimSpace(string(out)))
	}
	return n, nil
}
