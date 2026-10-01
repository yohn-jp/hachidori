package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/update"
)

// cmdApplyUpdate is the replacement helper. It is not a user command: the
// desktop starts it from a copy of the running executable after a verified
// update was downloaded, and it exits when it is done. It accepts exactly the
// four flags below, opens no network connection and replaces only the
// destination the ready record under HACHIDORI_HOME/state/updates names.
func cmdApplyUpdate(args []string) error {
	return runApplyUpdate(args, update.DefaultApplyEnv(), os.Stderr)
}

func runApplyUpdate(args []string, env update.ApplyEnv, out io.Writer) error {
	fs := flag.NewFlagSet("apply-update", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	var p update.Plan
	fs.StringVar(&p.Home, "home", "", "HACHIDORI_HOME that holds the ready update")
	fs.IntVar(&p.PID, "pid", 0, "process id of the application to wait for")
	fs.StringVar(&p.Target, "target", "", "the executable to replace")
	fs.StringVar(&p.RestartHome, "restart-home", "", "restart as `desktop --home <path>`")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return errors.New("apply-update takes no arguments")
	}
	res := update.Apply(p, env)
	fmt.Fprintf(out, "hachidori: update %s: %s\n", res.Outcome, res.Message)
	if res.Outcome != update.OutcomeApplied {
		return errors.New(res.Message)
	}
	return nil
}

// updateManager adapts the update subsystem to the dashboard's Updates
// surface and adds the one application-level rule it cannot know: an update
// is not installed while the application is busy with setup or maintenance.
// It adds no behavior of its own: nothing here runs until the operator posts
// an Updates action.
type updateManager struct {
	svc  *update.Service
	ctl  func() *app.Controller
	quit func() // ends the application so the helper can replace the executable
}

// quitDelay lets the response to the Restart & update request reach the page
// before the window closes.
const quitDelay = 750 * time.Millisecond

func (m updateManager) Status() update.Status             { return m.svc.Status() }
func (m updateManager) SetChannel(c update.Channel) error { return m.svc.SetChannel(c) }
func (m updateManager) Check(ctx context.Context) error   { return m.svc.Check(ctx) }
func (m updateManager) Download(tag string) error         { return m.svc.StartDownload(tag) }

func (m updateManager) Install() error {
	if c := m.ctl(); c != nil && c.Snapshot().Operation != nil {
		return &update.Error{Class: update.ClassRefused, Msg: "another application action is in progress; wait for it to finish, then restart & update"}
	}
	if err := m.svc.Install(); err != nil {
		return err
	}
	time.AfterFunc(quitDelay, m.quit)
	return nil
}

// newUpdateManager composes the update subsystem for the desktop. The release
// authority is fixed in internal/update; nothing here names a repository or a
// URL. Channel, last check and installed identity are stored by the one
// settings authority, and the staging area lives under HACHIDORI_HOME.
func (a *desktopApp) newUpdateManager(settings update.Store, ctl func() *app.Controller, homeArg string, quit func()) updateManager {
	exe := a.Exe
	if exe == "" {
		exe = updateExe()
	}
	start := a.UpdateStart
	if start == nil {
		start = update.StartDetached
	}
	return updateManager{
		svc: &update.Service{
			Home: func() string {
				if c := ctl(); c != nil {
					return c.Snapshot().Home
				}
				return ""
			},
			Exe: exe, Settings: settings, Transport: a.UpdateTransport, StartHelper: start, RestartHome: homeArg,
		},
		ctl:  ctl,
		quit: quit,
	}
}

// updateExe is the running executable: the destination of an update.
func updateExe() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	return filepath.Clean(exe)
}

var _ dashboard.Updates = updateManager{}
