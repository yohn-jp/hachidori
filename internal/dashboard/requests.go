package dashboard

import (
	"net/http"
)

func (d *Dashboard) requestHistory(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, d.cfg.RequestHistory.List())
}

func (d *Dashboard) requestHistoryDetail(w http.ResponseWriter, r *http.Request) {
	detail, ok := d.cfg.RequestHistory.Get(r.PathValue("id"))
	if !ok {
		http.NotFound(w, r)
		return
	}
	writeJSON(w, detail)
}
