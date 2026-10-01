package eval

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/yohn-jp/hachidori/internal/api"
)

// fakeResident is one resident of the fake endpoint: a fixed identity, a
// counter of what it answered and a scripted answer. Nothing in the fake can
// load, start, stop or activate a model: Decide and Status are its whole
// surface, so a test that passes proves the evaluation used nothing else.
type fakeResident struct {
	id, provider string
	running      bool
	pid, starts  int
	uptime       int
	loadMS       float64
	warmupMS     float64
	alloc, rsrv  int64
	dtype        string
	noAccel      bool
	decides      int
	// answer maps a state to (choice, confidence); the default answers the
	// first choice with 0.9.
	answer func(state string, q api.Question) (string, float64)
	// inferenceMS maps a state to the reported server inference time.
	inferenceMS func(state string) float64
}

type fakeResidents struct {
	mu        sync.Mutex
	residents []*fakeResident
	// misroute answers every request with the given resident instead of the
	// named one; noServed omits the served provenance.
	misroute string
	noServed bool
	// onDecide runs before each request is answered.
	onDecide func(f *fakeResidents, model string, n int)
	total    int
	statuses int
}

func (f *fakeResidents) find(id string) *fakeResident {
	for _, r := range f.residents {
		if r.id == id {
			return r
		}
	}
	return nil
}

func (f *fakeResidents) Decide(req api.DecideRequest) (api.DecideResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.total++
	if f.onDecide != nil {
		f.onDecide(f, req.Model, f.total)
	}
	id := req.Model
	if f.misroute != "" {
		id = f.misroute
	}
	r := f.find(id)
	if r == nil || !r.running {
		return api.DecideResponse{}, errors.New("not_ready")
	}
	r.decides++
	var rs []api.Result
	for _, q := range req.Questions {
		choice, conf := q.Choices[0], 0.9
		if r.answer != nil {
			choice, conf = r.answer(req.State, q)
		}
		probs := map[string]float64{}
		for _, c := range q.Choices {
			probs[c] = (1 - conf) / float64(len(q.Choices)-1)
		}
		probs[choice] = conf
		rs = append(rs, api.Result{ID: q.ID, Choice: choice, Confidence: conf, Probabilities: probs})
	}
	inf := 5.0
	if r.inferenceMS != nil {
		inf = r.inferenceMS(req.State)
	}
	resp := api.DecideResponse{Schema: api.SchemaV1, Results: rs, Timing: &api.Timing{InferenceMS: inf}}
	if !f.noServed {
		resp.Served = &api.Served{Model: r.id, Provider: r.provider}
	}
	return resp, nil
}

func (r *fakeResident) status() map[string]any {
	accel := map[string]any{}
	if !r.noAccel {
		accel = map[string]any{"memory_allocated": float64(r.alloc), "memory_reserved": float64(r.rsrv),
			"memory_free": float64(6 << 30), "memory_total": float64(12 << 30)}
	}
	prov := map[string]any{"provider": r.provider, "model_revision": "rev1", "load_ms": r.loadMS, "warmup_ms": r.warmupMS, "device": "cuda"}
	if r.dtype != "" {
		prov["dtype"] = r.dtype
	}
	return map[string]any{
		"schema":    api.SchemaV1,
		"runtime":   map[string]any{"model_id": r.id, "model": "org/" + r.id + "/rev1", "device": "cuda", "runtime": "rt1", "home": "h"},
		"uptime_s":  r.uptime,
		"worker":    map[string]any{"ready": r.running, "pid": r.pid, "starts": r.starts, "provider": prov, "accelerator": accel},
		"residents": nil,
	}
}

func (f *fakeResidents) Status() (json.RawMessage, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.statuses++
	doc := f.residents[0].status()
	var rs []map[string]any
	for i, r := range f.residents {
		st := r.status()
		delete(st, "residents")
		rs = append(rs, map[string]any{"model": r.id, "provider": r.provider, "default": i == 0, "running": r.running, "status": st})
	}
	doc["residents"] = rs
	return json.Marshal(doc)
}

func newFake() *fakeResidents {
	return &fakeResidents{residents: []*fakeResident{
		{id: "laya-base", provider: "laya", running: true, pid: 100, starts: 1, uptime: 50, loadMS: 3000, warmupMS: 400, alloc: 2 << 30, rsrv: 3 << 30},
		{id: "opendecider-nano", provider: "opendecider", running: true, pid: 200, starts: 1, uptime: 40, loadMS: 1500, warmupMS: 200, alloc: 1 << 30, rsrv: 2 << 30},
	}}
}

func yn(id string) api.Question {
	return api.Question{ID: id, Type: "choice", Instructions: "i", Choices: []string{"yes", "no"}}
}

func longState(n int) string { return strings.Repeat("x", n) }

func residentCases() []Case {
	mk := func(id, state string, exp map[string]string) Case {
		var qs []api.Question
		for k := range exp {
			qs = append(qs, yn(k))
		}
		// stable question order
		if len(qs) == 2 && qs[0].ID > qs[1].ID {
			qs[0], qs[1] = qs[1], qs[0]
		}
		return Case{ID: id, State: state, Questions: qs, Expected: exp}
	}
	return []Case{
		mk("short", "short state", map[string]string{"a": "yes", "b": "no"}),
		mk("mid", longState(600), map[string]string{"a": "yes"}),
		mk("long", longState(3000), map[string]string{"a": "no", "b": "yes"}),
	}
}

func TestRunResidentsSameInputsWithoutReload(t *testing.T) {
	f := newFake()
	cases := residentCases()
	rep, err := RunResidents(f, cases, "dsha", ResidentOptions{Options: Options{Warmup: 2, Passes: 2},
		Models: []string{"laya-base", "opendecider-nano"}, Families: map[string]string{"a": "readiness"}})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Schema != ComparisonSchema || rep.Alignment.Status != AlignAligned || !rep.ResidentsStable || len(rep.Runs) != 2 {
		t.Fatalf("%+v", rep.Alignment)
	}
	a, b := rep.Runs[0], rep.Runs[1]
	if a.Model != "laya-base" || b.Model != "opendecider-nano" {
		t.Fatalf("pass order %s %s", a.Model, b.Model)
	}
	// Identical normalized inputs, in identical order, for both models.
	if a.InputSHA256 != b.InputSHA256 || a.SentSHA256 != b.SentSHA256 || a.DatasetSHA256 != "dsha" || b.DatasetSHA256 != "dsha" {
		t.Fatalf("inputs differ: %+v %+v", a.SentSHA256, b.SentSHA256)
	}
	if !reflect.DeepEqual(a.Questions, b.Questions) {
		t.Fatal("question identities differ")
	}
	// Both stayed resident: status identity, PID and start count unchanged, and
	// the only operations are decides and status reads.
	for _, r := range rep.Runs {
		if !r.ResidentStable || r.Identity.WorkerStarts != 1 || r.IdentityEnd.WorkerStarts != 1 || r.Identity.WorkerPID != r.IdentityEnd.WorkerPID {
			t.Fatalf("%s not stable: %+v", r.Model, r.IdentityEnd)
		}
	}
	for _, r := range f.residents {
		if r.starts != 1 || r.pid == 0 || r.decides != 2+2*len(cases) {
			t.Fatalf("%s starts %d decides %d", r.id, r.starts, r.decides)
		}
	}
	// Per-observation provenance.
	if len(a.Observations) != 5 {
		t.Fatalf("observations %d", len(a.Observations))
	}
	for _, run := range rep.Runs {
		for _, o := range run.Observations {
			if o.ServedModel != run.Model || o.ServedProvider != run.Provider || o.IdentitySHA256 != run.Identity.Digest || o.InputSHA256 == "" {
				t.Fatalf("provenance %+v", o)
			}
		}
	}
	if a.Provider != "laya" || b.Provider != "opendecider" || a.Identity.Provider["model_revision"] != "rev1" ||
		a.Identity.Runtime["model"] != "org/laya-base/rev1" || a.Identity.Digest == b.Identity.Digest {
		t.Fatalf("identity %+v %+v", a.Identity, b.Identity)
	}
	// Cold start is separate metadata, not request latency.
	if a.Startup.LoadMS == nil || *a.Startup.LoadMS != 3000 || *a.Startup.WarmupMS != 400 || *b.Startup.LoadMS != 1500 {
		t.Fatalf("startup %+v %+v", a.Startup, b.Startup)
	}
	if a.RequestLatency.P95 >= 3000 || a.RequestLatency.N != 2*len(cases) || a.InferenceLatency.N != 2*len(cases) {
		t.Fatalf("latency %+v", a.RequestLatency)
	}
	if a.Requests != 6 || a.Succeeded != 6 || a.ErrorCount != 0 || len(a.ErrorsByClass) != 0 {
		t.Fatalf("counts %+v", a)
	}
	// Memory: resident reading and sampled peak, per resident.
	if !a.Memory.Available || *a.Memory.ResidentAlloc != 2<<30 || *a.Memory.PeakRsrv != 3<<30 || *b.Memory.ResidentAlloc != 1<<30 {
		t.Fatalf("memory %+v", a.Memory)
	}
	phases := []string{}
	for _, s := range a.Memory.Samples {
		phases = append(phases, s.Phase)
	}
	if !reflect.DeepEqual(phases, []string{"before_run", "after_warmup", "after_pass_1", "after_pass_2", "after_run"}) {
		t.Fatalf("memory samples %v", phases)
	}
	// Slices.
	if len(a.PerQuestion) != 2 || a.PerQuestion[0].Key != "a" {
		t.Fatalf("per question %+v", a.PerQuestion)
	}
	fam := map[string]int{}
	for _, s := range a.PerFamily {
		fam[s.Key] = s.N
	}
	if fam["readiness"] != 3 || fam[UnassignedFamily] != 2 {
		t.Fatalf("families %+v", fam)
	}
	// No verdict anywhere in the document.
	raw, _ := json.Marshal(rep)
	for _, w := range []string{"winner", "best", "verdict", "\"pass\":", "\"fail\":", "recommend"} {
		if strings.Contains(strings.ToLower(string(raw)), w) {
			t.Fatalf("report contains %q", w)
		}
	}
	// The report round-trips through the strict decoder.
	back, err := DecodeComparison(raw)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(mustJSON(back), mustJSON(rep)) {
		t.Fatal("round trip changed the report")
	}
}

func mustJSON(v any) string { b, _ := json.Marshal(v); return string(b) }

func TestRunResidentsDetectsReloadDuringTheComparison(t *testing.T) {
	f := newFake()
	// The first resident is restarted while the second is being evaluated.
	f.onDecide = func(f *fakeResidents, model string, n int) {
		if model == "opendecider-nano" && f.find("laya-base").starts == 1 {
			r := f.find("laya-base")
			r.starts, r.pid, r.uptime, r.loadMS = 2, 101, 1, 2900
		}
	}
	rep, err := RunResidents(f, residentCases(), "d", ResidentOptions{Models: []string{"laya-base", "opendecider-nano"}})
	if err != nil {
		t.Fatal(err)
	}
	if rep.ResidentsStable || rep.Runs[0].ResidentStable || !rep.Runs[1].ResidentStable {
		t.Fatalf("a reload must be marked: %v %v %v", rep.ResidentsStable, rep.Runs[0].ResidentStable, rep.Runs[1].ResidentStable)
	}
	if rep.Runs[0].IdentityEnd.WorkerStarts != 2 {
		t.Fatalf("end identity %+v", rep.Runs[0].IdentityEnd)
	}
}

func TestRunResidentsNeverRoutesElsewhere(t *testing.T) {
	for name, mod := range map[string]func(*fakeResidents){
		"misrouted":  func(f *fakeResidents) { f.misroute = "laya-base" },
		"no-served":  func(f *fakeResidents) { f.noServed = true },
		"unanswered": func(f *fakeResidents) { f.misroute = "ghost" },
	} {
		f := newFake()
		mod(f)
		rep, err := RunResidents(f, residentCases(), "d", ResidentOptions{Models: []string{"laya-base", "opendecider-nano"}})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for _, r := range rep.Runs {
			if name == "unanswered" {
				if r.Succeeded != 0 || len(r.Observations) != 0 {
					t.Fatalf("%s: %+v", name, r)
				}
				continue
			}
			// opendecider-nano misrouted to laya-base is never scored; laya-base
			// answering as itself is.
			want := r.Model == "laya-base" && name == "misrouted"
			if (len(r.Observations) > 0) != want {
				t.Fatalf("%s/%s: %d observations", name, r.Model, len(r.Observations))
			}
			if !want && (r.ErrorsByClass[ErrClassServedMismatch] != len(residentCases()) || r.Quality.N != 0 || r.Quality.Accuracy != nil) {
				t.Fatalf("%s/%s: %+v", name, r.Model, r.ErrorsByClass)
			}
		}
	}
}

func TestRunResidentsRefusesUnreadyResidentBeforeAnyRequest(t *testing.T) {
	f := newFake()
	f.residents[1].running = false
	_, err := RunResidents(f, residentCases(), "d", ResidentOptions{Models: []string{"laya-base", "opendecider-nano"}})
	if err == nil || !strings.Contains(err.Error(), "opendecider-nano") {
		t.Fatalf("%v", err)
	}
	if f.total != 0 {
		t.Fatalf("%d requests were sent although a resident is not running", f.total)
	}
	f = newFake()
	if _, err = RunResidents(f, residentCases(), "d", ResidentOptions{Models: []string{"laya-base", "ghost"}}); err == nil || f.total != 0 {
		t.Fatalf("unknown model accepted: %v", err)
	}
	for _, bad := range []ResidentOptions{
		{Models: []string{"laya-base"}},
		{Models: []string{"laya-base", "laya-base"}},
		{Models: []string{"laya-base", "opendecider-nano"}, HighConfidence: 1.5},
		{Models: []string{"laya-base", "opendecider-nano"}, Thresholds: []float64{0.9, 0.5}},
		{Models: []string{"laya-base", "opendecider-nano"}, LengthEdges: []int{512, 256}},
		{Models: []string{"laya-base", "opendecider-nano"}, Families: map[string]string{"nope": "f"}},
	} {
		if _, err := RunResidents(f, residentCases(), "d", bad); err == nil {
			t.Fatalf("accepted %+v", bad)
		}
	}
	if f.total != 0 {
		t.Fatal("options were validated after sending requests")
	}
}

func TestRunResidentsCPUResidentHasNoMemoryClaim(t *testing.T) {
	f := newFake()
	f.residents[1].noAccel = true
	rep, err := RunResidents(f, residentCases(), "d", ResidentOptions{Models: []string{"laya-base", "opendecider-nano"}})
	if err != nil {
		t.Fatal(err)
	}
	m := rep.Runs[1].Memory
	if m.Available || m.PeakAlloc != nil || m.ResidentAlloc != nil || len(m.Samples) == 0 {
		t.Fatalf("memory claimed without statistics: %+v", m)
	}
	if !rep.Runs[0].Memory.Available {
		t.Fatal("the other resident lost its memory evidence")
	}
}

// The OpenDecider-nano long-input behaviour must be measurable: latency and
// accuracy by input length, per model, with every bucket present.
func TestInputLengthBucketsExposeLongInputBehaviour(t *testing.T) {
	f := newFake()
	slow := f.residents[1]
	slow.inferenceMS = func(s string) float64 {
		if utf8.RuneCountInString(s) >= 2048 {
			return 900
		}
		return 20
	}
	slow.answer = func(s string, q api.Question) (string, float64) {
		if utf8.RuneCountInString(s) >= 2048 {
			return "yes", 0.95 // confident and wrong for the long case
		}
		return "yes", 0.9
	}
	rep, err := RunResidents(f, residentCases(), "d", ResidentOptions{Options: Options{Passes: 3},
		Models: []string{"laya-base", "opendecider-nano"}})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range rep.Runs {
		if len(r.LengthBuckets) != len(DefaultLengthEdges)+1 {
			t.Fatalf("%s: %d buckets", r.Model, len(r.LengthBuckets))
		}
	}
	bucket := func(r ModelRun, label string) LengthBucket {
		for _, b := range r.LengthBuckets {
			if b.Label == label {
				return b
			}
		}
		t.Fatalf("no bucket %q in %v", label, r.LengthBuckets)
		return LengthBucket{}
	}
	nano := rep.Runs[1]
	short, long := bucket(nano, "0-255 chars"), bucket(nano, "2048-4095 chars")
	if short.Cases != 1 || long.Cases != 1 || short.Requests != 3 || long.Requests != 3 {
		t.Fatalf("short %+v long %+v", short, long)
	}
	if short.InferenceLatency.P50 != 20 || long.InferenceLatency.P50 != 900 || long.InferenceLatency.P95 != 900 {
		t.Fatalf("inference by length: %+v %+v", short.InferenceLatency, long.InferenceLatency)
	}
	if bucket(rep.Runs[0], "2048-4095 chars").InferenceLatency.P50 != 5 {
		t.Fatal("the other model's latency was mixed in")
	}
	if *long.Accuracy != 0.5 || long.Observations != 2 || bucket(nano, "16384+ chars").Cases != 0 {
		t.Fatalf("long accuracy %v %+v", long.Accuracy, bucket(nano, "16384+ chars"))
	}
	if bucket(nano, "16384+ chars").RequestLatency.N != 0 || bucket(nano, "16384+ chars").Accuracy != nil {
		t.Fatal("an empty bucket must report no latency and no accuracy")
	}
	// Edges are declared and recorded; custom edges are honoured.
	rep, err = RunResidents(f, residentCases(), "d", ResidentOptions{Models: []string{"laya-base", "opendecider-nano"}, LengthEdges: []int{100}})
	if err != nil || len(rep.Runs[0].LengthBuckets) != 2 || rep.Declared.LengthEdges[0] != 100 {
		t.Fatalf("%v %+v", err, rep.Declared)
	}
	if b := rep.Runs[0].LengthBuckets; b[0].Cases != 1 || b[1].Cases != 2 || b[1].MaxChars != nil || *b[0].MaxChars != 100 {
		t.Fatalf("%+v", b)
	}
}

func TestErrorsAreAttributedToTheirInputBucket(t *testing.T) {
	f := newFake()
	f.onDecide = nil
	f.residents[1].answer = nil
	// Fail every request of the long case on the second model only.
	g := &failingOn{fakeResidents: f, model: "opendecider-nano", state: longState(3000)}
	rep, err := RunResidents(g, residentCases(), "d", ResidentOptions{Options: Options{Passes: 2}, Models: []string{"laya-base", "opendecider-nano"}})
	if err != nil {
		t.Fatal(err)
	}
	var long LengthBucket
	for _, b := range rep.Runs[1].LengthBuckets {
		if b.Label == "2048-4095 chars" {
			long = b
		}
	}
	if long.Errors != 2 || long.Requests != 0 || rep.Runs[1].ErrorCount != 2 || rep.Runs[1].Succeeded != 4 || rep.Runs[0].ErrorCount != 0 {
		t.Fatalf("%+v %+v", long, rep.Runs[1].ErrorsByClass)
	}
}

type failingOn struct {
	*fakeResidents
	model, state string
}

func (g *failingOn) Decide(r api.DecideRequest) (api.DecideResponse, error) {
	if r.Model == g.model && r.State == g.state {
		return api.DecideResponse{}, fmt.Errorf("worker_failure")
	}
	return g.fakeResidents.Decide(r)
}

func TestAlignRefusesMismatchedIdentities(t *testing.T) {
	f := newFake()
	rep, err := RunResidents(f, residentCases(), "d", ResidentOptions{Models: []string{"laya-base", "opendecider-nano"}})
	if err != nil {
		t.Fatal(err)
	}
	clone := func() []ModelRun {
		var runs []ModelRun
		if err := json.Unmarshal([]byte(mustJSON(rep.Runs)), &runs); err != nil {
			t.Fatal(err)
		}
		return runs
	}
	if a := Align(clone()); a.Status != AlignAligned || len(a.Incompatibility) != 0 {
		t.Fatalf("%+v", a)
	}
	cases := map[string]struct {
		mod  func([]ModelRun)
		code string
		q    string
	}{
		"dataset":  {func(r []ModelRun) { r[1].DatasetSHA256 = "other" }, IncompatDataset, ""},
		"inputs":   {func(r []ModelRun) { r[1].InputSHA256 = "other" }, IncompatInputs, ""},
		"sent":     {func(r []ModelRun) { r[1].SentSHA256 = "other" }, IncompatInputs, ""},
		"protocol": {func(r []ModelRun) { r[1].Passes = 9 }, IncompatProtocol, ""},
		"cases":    {func(r []ModelRun) { r[1].Cases = 9 }, IncompatCases, ""},
		"qsha":     {func(r []ModelRun) { r[1].Questions[0].SHA256 = "changed" }, IncompatQuestionIdentity, "a"},
		"only-in":  {func(r []ModelRun) { r[1].Questions = r[1].Questions[:1] }, IncompatOnlyInA, "b"},
	}
	for name, c := range cases {
		runs := clone()
		c.mod(runs)
		a := Align(runs)
		if a.Status != AlignRefused {
			t.Fatalf("%s: %+v", name, a)
		}
		found := false
		for _, i := range a.Incompatibility {
			found = found || i.Code == c.code && i.Question == c.q && i.Model == "opendecider-nano"
		}
		if !found {
			t.Fatalf("%s: %+v", name, a.Incompatibility)
		}
	}
	if Align(clone()[:1]).Status != AlignRefused {
		t.Fatal("one run is not a comparison")
	}
}

func TestValidateComparisonRejectsFalseAlignmentAndAcceptsExplicitRefusal(t *testing.T) {
	rep, err := RunResidents(newFake(), residentCases(), "d", ResidentOptions{Models: []string{"laya-base", "opendecider-nano"}})
	if err != nil {
		t.Fatal(err)
	}
	tampered := func(mod func(*ComparisonReport)) []byte {
		var c ComparisonReport
		if err := json.Unmarshal([]byte(mustJSON(rep)), &c); err != nil {
			t.Fatal(err)
		}
		mod(&c)
		return []byte(mustJSON(c))
	}
	// Claims aligned, but the second run used another dataset.
	if _, err := DecodeComparison(tampered(func(c *ComparisonReport) { c.Runs[1].DatasetSHA256 = "other" })); err == nil ||
		!strings.Contains(err.Error(), "recompute") {
		t.Fatalf("false alignment accepted: %v", err)
	}
	// Honestly marked refusal is a valid document and stays refused.
	c, err := DecodeComparison(tampered(func(c *ComparisonReport) {
		c.Runs[1].DatasetSHA256 = "other"
		c.Alignment = Align(c.Runs)
	}))
	if err != nil || c.Alignment.Status != AlignRefused || len(c.Alignment.Incompatibility) == 0 {
		t.Fatalf("%v %+v", err, c.Alignment)
	}
	for name, mod := range map[string]func(*ComparisonReport){
		"foreign observation": func(c *ComparisonReport) { c.Runs[0].Observations[0].ServedModel = "opendecider-nano" },
		"count":               func(c *ComparisonReport) { c.Runs[0].Quality.N++ },
		"order": func(c *ComparisonReport) {
			c.Declared.Models[0], c.Declared.Models[1] = c.Declared.Models[1], c.Declared.Models[0]
		},
		"schema":  func(c *ComparisonReport) { c.Schema = EvidenceSchema },
		"one run": func(c *ComparisonReport) { c.Runs = c.Runs[:1] },
	} {
		if _, err := DecodeComparison(tampered(mod)); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
	if _, err := DecodeComparison([]byte(strings.Replace(mustJSON(rep), `"schema":`, `"extra":1,"schema":`, 1))); err == nil {
		t.Fatal("unknown field accepted")
	}
	if _, err := DecodeComparison([]byte(mustJSON(rep) + " {}")); err == nil {
		t.Fatal("trailing data accepted")
	}
}

func TestModelRunEvidenceIsAV1Report(t *testing.T) {
	rep, err := RunResidents(newFake(), residentCases(), "d", ResidentOptions{Options: Options{Passes: 2}, Models: []string{"laya-base", "opendecider-nano"}})
	if err != nil {
		t.Fatal(err)
	}
	ev := rep.Runs[1].Evidence("ep", "ds.jsonl", rep.Definitions)
	if ev.Schema != EvidenceSchema || ev.Observations != 5 || ev.Served.Model() != "org/opendecider-nano/rev1" || !ev.ServedConsistent ||
		ev.DatasetSHA256 != "d" || ev.RequestLatency != rep.Runs[1].RequestLatency || len(ev.PerQuestion) != 2 {
		t.Fatalf("%+v", ev)
	}
}

func TestResidentSummaryHasNoVerdict(t *testing.T) {
	rep, err := RunResidents(newFake(), residentCases(), "dsha", ResidentOptions{Models: []string{"laya-base", "opendecider-nano"}})
	if err != nil {
		t.Fatal(err)
	}
	var sb strings.Builder
	ResidentSummary(&sb, rep)
	out := sb.String()
	for _, want := range []string{"laya-base", "opendecider-nano", "alignment      aligned", "Brier", "coverage", "input length", "cold start metadata"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in\n%s", want, out)
		}
	}
	for _, w := range []string{"winner", "best", "better"} {
		if strings.Contains(strings.ToLower(out), w) {
			t.Fatalf("summary contains %q", w)
		}
	}
}

// The single-model path is untouched: Run and RunEvidence keep their
// behaviour and the v1 schema, and the v1 document gains no comparison field.
func TestSingleModelEvaluationUnchanged(t *testing.T) {
	s := &stub{answers: map[string]string{"a": "yes"}}
	r := Run(s, []Case{{ID: "c", State: "a", Questions: []api.Question{q("x")}, Expected: map[string]string{"x": "yes"}}}, Options{})
	if r.Schema != EvidenceSchema || s.seen[0].Model != "" {
		t.Fatalf("%+v", s.seen[0])
	}
	raw, _ := json.Marshal(r)
	for _, k := range []string{"alignment", "length_buckets", "calibration", "brier"} {
		if strings.Contains(string(raw), k) {
			t.Fatalf("v1 report gained %q", k)
		}
	}
}
