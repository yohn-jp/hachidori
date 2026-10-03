package desktopkit

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/yohn-jp/hachidori/internal/app"
	"github.com/yohn-jp/hachidori/internal/dashboard"
	"github.com/yohn-jp/hachidori/internal/desktop"
	"github.com/yohn-jp/hachidori/internal/firstrun"
	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/server"
	"github.com/yohn-jp/hachidori/internal/tunnel"
	"github.com/yohn-jp/hachidori/internal/worker"
)

func addrOf(ts *httptest.Server) string { return ts.Listener.Addr().String() }

type neverPicks struct{}

func (neverPicks) PickFolder(context.Context, string) (string, error) {
	return "", desktop.ErrPickCancelled
}

// The HTTP half of the scenarios (first-run state, dashboard token, Start, API
// status and health) is exercised here against the production handlers wired
// the way the desktop wires them, over the fixture home and its stand-in
// worker, so the clients are proven against the real pages and documents on
// any OS before they are pointed at the packaged executable.
func TestClientsAgainstTheProductionHandlers(t *testing.T) {
	inst, err := InstallFixture(filepath.Join(t.TempDir(), "Home ハチドリ"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var apiH, dashH http.Handler
	open := func(root string) (app.Runtime, error) {
		return app.WorkerRuntime(ctx, io.Discard, worker.DefaultPolicy, func(b *app.WorkerBinding) {
			dashH = dashboard.New(dashboard.Config{
				APIAddr: "127.0.0.1:7843", Status: b.Status, Lifecycle: b.Lifecycle,
				Doctor: func(io.Writer) bool { return true }, Tunnel: tunnel.NewManager("ssh"),
				PrefsPath: filepath.Join(root, "state", "dashboard.json"),
			})
			apiH = server.HandlerSince(b.Supervisor, b.Info, b.Started)
		})(root)
	}
	env := firstrun.Env{}
	ctl := app.New(app.Config{Home: inst.Home.Root, Open: open, Installed: env.IsInstalled})
	defer func() {
		c, cc := context.WithTimeout(context.Background(), 20*time.Second)
		defer cc()
		_ = ctl.Close(c)
	}()
	if err := ctl.Bind(); err != nil {
		t.Fatal(err)
	}
	flow := firstrun.New(firstrun.Config{Ctl: ctl, Plan: firstrun.Plan{Mode: firstrun.ModeLaunch, Home: inst.Home.Root},
		Picker: neverPicks{}, Env: env, Remember: home.Remember})
	apiSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { apiH.ServeHTTP(w, r) }))
	defer apiSrv.Close()
	wiz := firstrun.NewHandler(flow, func() http.Handler { return dashH })
	dashSrv := httptest.NewServer(wiz)
	defer dashSrv.Close()

	v, err := WizardState(addrOf(dashSrv))
	if err != nil {
		t.Fatal(err)
	}
	if v.Mode != firstrun.ModeLaunch || !v.Dashboard || v.Selection != nil {
		t.Fatalf("wizard state %+v", v)
	}
	if st, err := Status(addrOf(apiSrv)); err != nil || st.Runtime.Home != inst.Home.Root || st.Worker.PID != 0 || st.Worker.Ready {
		t.Fatalf("status before Start: %+v, %v", st, err)
	}
	if ok, err := Ready(addrOf(apiSrv)); err != nil || ok {
		t.Fatalf("READY before Start: %v, %v", ok, err)
	}

	tok, err := DashboardToken(addrOf(dashSrv))
	if err != nil {
		t.Fatal(err)
	}
	if err := PostRuntime(addrOf(dashSrv), "start", "stale-token"); err == nil {
		t.Fatal("a post with a wrong token must be refused")
	}
	if err := PostRuntime(addrOf(dashSrv), "start", tok); err != nil {
		t.Fatal(err)
	}
	if err := Poll(30*time.Second, "READY", func() (bool, error) { return Ready(addrOf(apiSrv)) }); err != nil {
		t.Fatal(err)
	}
	st, err := Status(addrOf(apiSrv))
	if err != nil || st.Worker.PID == 0 || !st.Worker.Ready || st.Worker.Starts != 1 {
		t.Fatalf("status after Start: %+v, %v", st.Worker, err)
	}
	if _, err := WizardState("127.0.0.1:1"); err == nil {
		t.Error("an unreachable dashboard must be an error")
	}
	if !errors.Is(Fatal{io.EOF}, io.EOF) {
		t.Error("Fatal must unwrap")
	}
}
