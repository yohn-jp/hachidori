package dashboard

// The Settings > Updates surface. Like the rest of the dashboard it decides
// nothing: it renders the update subsystem's local status (internal/update)
// and forwards the four explicit operator actions. Rendering the page, opening
// it and saving the channel never use the network; only the Check and Download
// actions do, and only inside the update subsystem.

import (
	"context"
	"net/http"
	"strings"

	"github.com/yohn-jp/hachidori/internal/update"
)

// Updates is the update subsystem as the dashboard uses it (update.Service,
// adapted by the desktop composition for the application-level rules).
type Updates interface {
	// Status is local: it reads no network.
	Status() update.Status
	// SetChannel saves the channel; it contacts, downloads and installs nothing.
	SetChannel(update.Channel) error
	// Check retrieves release metadata. Only this action does.
	Check(ctx context.Context) error
	// Download begins retrieving one release's executable and checksum. Only
	// this action does. It returns once accepted; progress is in Status.
	Download(tag string) error
	// Install hands the verified update to the replacement helper and ends the
	// application so it can run.
	Install() error
}

// UpdatesView is the Updates view model.
type UpdatesView struct {
	update.Status
	Channels []update.Channel
	// Op is the download in flight and LastOp the last one finished, in the
	// long-running operation presentation shared with Models & Runtimes.
	Op, LastOp *ModelOp
	// Hints are the operator's next step for a failure (update.Hints).
	OpHint, CheckHint, ResultHint string
}

// updateOp restates the update subsystem's operation as the shared
// long-running operation view.
func updateOp(o *update.Operation) *ModelOp {
	if o == nil {
		return nil
	}
	op := &ModelOp{Kind: o.Kind, Target: o.Tag, Plan: o.Plan, Phases: o.Phases, Phase: o.Phase,
		Step: o.Step, Detail: o.Detail, Done: o.Done, Total: o.Total, Started: o.Started, Finished: o.Finished}
	if f := o.Failure; f != nil {
		op.Failure, op.FailurePhase, op.FailureStep = f.Message, f.Phase, f.Step
	}
	return op
}

func (d *Dashboard) updatesView() *UpdatesView {
	st := d.cfg.Updates.Status()
	uv := &UpdatesView{Status: st, Channels: update.Channels, Op: updateOp(st.Busy), LastOp: updateOp(st.Last)}
	if f := st.Last; f != nil && f.Failure != nil {
		uv.OpHint = update.Hints[f.Failure.Class]
	}
	if lc := st.LastCheck; lc != nil && !lc.OK {
		uv.CheckHint = update.Hints[lc.Class]
	}
	if r := st.Result; r != nil {
		switch r.Outcome {
		case update.OutcomeFailed:
			uv.ResultHint = update.Hints[update.ClassReplace]
		case update.OutcomeRestart:
			uv.ResultHint = update.Hints[update.ClassRestart]
		}
	}
	return uv
}

// updatesPage renders Updates from local state only. It never checks.
func (d *Dashboard) updatesPage(w http.ResponseWriter, r *http.Request) {
	v := d.view("Updates", "settings")
	v.Live = false
	v.HasUpdates = true
	v.Upd = d.updatesView()
	d.renderView(w, "updates", v)
}

// updatesChannel stores the channel. It only stores.
func (d *Dashboard) updatesChannel(w http.ResponseWriter, r *http.Request) {
	c, err := update.ParseChannel(strings.TrimSpace(r.PostFormValue("channel")))
	if err == nil {
		err = d.cfg.Updates.SetChannel(c)
	}
	d.done(w, r, "update channel", err, "saved; nothing was checked, downloaded or installed")
}

// updatesCheck is the explicit Check for updates action.
func (d *Dashboard) updatesCheck(w http.ResponseWriter, r *http.Request) {
	d.done(w, r, "check for updates", d.cfg.Updates.Check(r.Context()), "check finished; the releases are listed below")
}

// updatesDownload is the explicit Download action for one release.
func (d *Dashboard) updatesDownload(w http.ResponseWriter, r *http.Request) {
	d.done(w, r, "download update", d.cfg.Updates.Download(strings.TrimSpace(r.PostFormValue("tag"))),
		"download started; the update is installable only after its checksum verifies")
}

// updatesInstall is the explicit Restart & update action.
func (d *Dashboard) updatesInstall(w http.ResponseWriter, r *http.Request) {
	d.done(w, r, "restart & update", d.cfg.Updates.Install(), "Hachidori is closing so the update can be installed; it reopens from the same path")
}
