package dashboard

import (
	"errors"
	"net/http"
	"net/url"
	"strings"
	"testing"
)

type fakeDesktopPrefs struct {
	signIn, minimized bool
	err               error
	sets              int
}

func (f *fakeDesktopPrefs) Prefs() (bool, bool, error) { return f.signIn, f.minimized, nil }
func (f *fakeDesktopPrefs) Set(a, b bool) error {
	f.sets++
	if f.err != nil {
		return f.err
	}
	f.signIn, f.minimized = a, b
	return nil
}

func withDesktop(e *env, f *fakeDesktopPrefs) {
	cfg := e.d.cfg
	cfg.Desktop = f
	e.d = New(cfg)
}

func TestDesktopPanelAbsentUnlessHostedByDesktop(t *testing.T) {
	e := newEnv(t)
	body := e.get(t, "/").Body.String() + e.get(t, "/diagnostics").Body.String()
	if strings.Contains(body, "Start Hachidori when I sign in") || strings.Contains(body, "/desktop/prefs") {
		t.Fatal("serve/dashboard page shows the desktop panel")
	}
	if rec := e.post(t, "/desktop/prefs", nil); rec.Code != http.StatusNotFound && rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("desktop route exists without a desktop shell: %d", rec.Code)
	}
}

func TestDesktopPanelShowsOptInPreferencesAndTrayBehaviour(t *testing.T) {
	e := newEnv(t)
	withDesktop(e, &fakeDesktopPrefs{})
	body := e.get(t, "/diagnostics").Body.String()
	for _, want := range []string{"Start Hachidori when I sign in", "Start minimized", "hides Hachidori to the system tray",
		"Quit Hachidori", `action="/desktop/prefs"`} {
		if !strings.Contains(body, want) {
			t.Errorf("diagnostics lacks %q", want)
		}
	}
	// The tray opens the dashboard root at #diagnostics when the application
	// needs attention (desktop.DiagnosticsFragment); the root keeps that
	// anchor and forwards it to the Diagnostics workspace.
	if root := e.get(t, "/").Body.String(); !strings.Contains(root, `id="diagnostics"`) || !strings.Contains(root, `location.hash === "#diagnostics"`) {
		t.Error("runtime page does not keep the #diagnostics entry")
	}
	if strings.Contains(body, "checked") {
		t.Error("preferences are opt-in: nothing may be checked by default")
	}
}

func TestDesktopPrefsFormAppliesCompleteDesiredState(t *testing.T) {
	e := newEnv(t)
	f := &fakeDesktopPrefs{}
	withDesktop(e, f)
	if rec := e.post(t, "/desktop/prefs", url.Values{"start_at_sign_in": {"1"}}); rec.Code != http.StatusSeeOther {
		t.Fatalf("status %d", rec.Code)
	}
	if !f.signIn || f.minimized {
		t.Fatalf("after enabling sign-in only: %+v", f)
	}
	if !strings.Contains(e.get(t, "/diagnostics").Body.String(), `name="start_at_sign_in" value="1" checked`) {
		t.Error("enabled preference not shown checked")
	}
	// An empty form (both boxes cleared) turns both off: reversible.
	e.post(t, "/desktop/prefs", nil)
	if f.signIn || f.minimized {
		t.Fatalf("after clearing: %+v", f)
	}
	// Same-origin/token protections apply to the new route as to all POSTs.
	if rec := e.post(t, "/desktop/prefs", url.Values{"token": {"forged"}, "start_at_sign_in": {"1"}}); rec.Code != http.StatusForbidden || f.signIn {
		t.Fatalf("forged token: %d %+v", rec.Code, f)
	}
}

func TestDesktopPrefsFailureIsShown(t *testing.T) {
	e := newEnv(t)
	withDesktop(e, &fakeDesktopPrefs{err: errors.New("registry denied")})
	e.post(t, "/desktop/prefs", url.Values{"start_at_sign_in": {"1"}})
	if a := e.lastAction(t); a.OK || !strings.Contains(a.Message, "registry denied") {
		t.Fatalf("action %+v", a)
	}
}
