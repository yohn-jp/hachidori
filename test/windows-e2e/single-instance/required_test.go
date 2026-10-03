package singleinstance

import (
	"testing"

	"github.com/yohn-jp/hachidori/test/windows-e2e/desktopkit"
)

// TestRequiredScenariosMatchTheTests keeps required.json and the scenarios the
// package begins in step. It runs in portable mode too.
func TestRequiredScenariosMatchTheTests(t *testing.T) { desktopkit.CheckScenarioCoverage(t, ".") }
