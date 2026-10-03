package trial

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/yohn-jp/hachidori/internal/worker"
)

type call struct {
	op   string
	args map[string]any
}

// fakeCaller answers the worker's trial protocol with canned results.
type fakeCaller struct {
	pid     int
	calls   []call
	results map[string]any
	errs    map[string]error
}

func (f *fakeCaller) Trial(op string, args map[string]any) (json.RawMessage, int, error) {
	f.calls = append(f.calls, call{op, args})
	if err := f.errs[op]; err != nil {
		return nil, f.pid, err
	}
	b, _ := json.Marshal(f.results[op])
	return b, f.pid, nil
}

func newCaller() *fakeCaller {
	return &fakeCaller{pid: 41, errs: map[string]error{}, results: map[string]any{
		"trial_open": map[string]any{"device": "cuda", "dtype": "bfloat16", "backend": "compressed-tensors 0.19.0", "canonical_bytes": 10, "transform": TransformImplementation,
			"modules": []map[string]any{{"module": "b", "shape": []int{2, 128}, "dtype": "bfloat16"}, {"module": "a", "shape": []int{2, 128}, "dtype": "bfloat16"}}},
		"trial_transform": map[string]any{"bytes": 7, "digest": "abc"},
		"trial_apply":     map[string]any{"bytes_to_gpu": 3, "bytes_released": 4, "modules": 2},
		"trial_state":     map[string]any{"groups": map[string]string{"g1": "dense", "g2": "packed-int4", "g3": "mixed"}, "gpu_allocated": 99},
	}}
}

func TestWorkerBackendSpeaksTheTrialProtocol(t *testing.T) {
	c := newCaller()
	b := &WorkerBackend{Caller: c}
	ctx := context.Background()
	opened, err := b.Open(ctx, map[string][]string{"g": {"a", "b"}})
	if err != nil || opened.Device != "cuda" || opened.Backend != "compressed-tensors 0.19.0" || opened.CanonicalBytes != 10 || opened.Modules[0].Module != "a" {
		t.Fatalf("open %+v (%v)", opened, err)
	}
	id := ComponentIdentity{Schema: ComponentSchema, Group: "g", Members: []Member{{Module: "a", Shape: []int64{2, 128}, DType: "bfloat16"}},
		Transformation: Transformation{Policy: "w4a16", Representation: RepresentationPacked, Implementation: TransformImplementation}}
	built, err := b.Transform(ctx, id)
	if err != nil || built.Bytes != 7 || built.Digest != "abc" {
		t.Fatalf("transform %+v (%v)", built, err)
	}
	sent := c.calls[1].args["identity"].(map[string]any)
	if sent["id"] != id.ID() {
		t.Fatalf("the worker was not told the component identity: %v", sent["id"])
	}
	applied, err := b.Apply(ctx, []Replacement{{Group: "g", Policy: "w4a16", Component: id.ID(), Modules: []string{"a"}, Transformation: id.Transformation}})
	if err != nil || applied != (Applied{BytesToGPU: 3, BytesReleased: 4, Modules: 2}) {
		t.Fatalf("apply %+v (%v)", applied, err)
	}
	wire, _ := json.Marshal(c.calls[2].args["replacements"])
	var back []map[string]any
	_ = json.Unmarshal(wire, &back)
	if back[0]["group"] != "g" || back[0]["component"] != id.ID() || back[0]["transformation"].(map[string]any)["representation"] != RepresentationPacked {
		t.Fatalf("replacement wire form %s", wire)
	}
	st, err := b.State(ctx)
	if err != nil || st.Policies["g1"] != "source-precision" || st.Policies["g2"] != "w4a16" || st.Policies["g3"] != "mixed" || *st.GPUAllocated != 99 || st.HostRSS != nil {
		t.Fatalf("state %+v (%v)", st, err)
	}
}

func TestWorkerBackendMapsWorkerRefusalsToTypedErrors(t *testing.T) {
	c := newCaller()
	b := &WorkerBackend{Caller: c}
	ctx := context.Background()
	if _, err := b.Open(ctx, nil); err != nil {
		t.Fatal(err)
	}
	for class, check := range map[string]func(error) bool{
		"trial_unsupported":  func(e error) bool { var x *UnsupportedError; return errors.As(e, &x) },
		"trial_incompatible": func(e error) bool { var x *IncompatibleError; return errors.As(e, &x) },
		"trial_state_lost":   func(e error) bool { var x *StateLostError; return errors.As(e, &x) },
	} {
		c.errs["trial_validate"] = &worker.RequestError{Class: class, Message: "because"}
		if err := b.Validate(ctx, nil); !check(err) {
			t.Errorf("%s mapped to %T (%v)", class, err, err)
		}
	}
	c.errs["trial_validate"] = &worker.RequestError{Class: "trial_failed", Message: "boom"}
	var re *worker.RequestError
	if err := b.Validate(ctx, nil); !errors.As(err, &re) || re.Class != "trial_failed" {
		t.Fatalf("an unclassified refusal was rewritten: %v", err)
	}
}

func TestWorkerBackendReportsAReplacedWorkerAsLostState(t *testing.T) {
	c := newCaller()
	b := &WorkerBackend{Caller: c}
	ctx := context.Background()
	if _, err := b.Open(ctx, nil); err != nil {
		t.Fatal(err)
	}
	c.pid = 99 // the supervisor restarted the worker: it holds a fresh source model
	_, err := b.State(ctx)
	var lost *StateLostError
	if !errors.As(err, &lost) {
		t.Fatalf("a restarted worker was trusted: %v", err)
	}
}

func TestWorkerBackendRefusesAWorkerWithAnotherTransformation(t *testing.T) {
	c := newCaller()
	c.results["trial_open"].(map[string]any)["transform"] = "hachidori.trial-transform/0"
	if _, err := (&WorkerBackend{Caller: c}).Open(context.Background(), nil); err == nil {
		t.Fatal("a worker with another trial transformation was opened")
	}
	c = newCaller()
	c.results["trial_transform"] = map[string]any{"bytes": 1}
	b := &WorkerBackend{Caller: c}
	if _, err := b.Open(context.Background(), nil); err != nil {
		t.Fatal(err)
	}
	if _, err := b.Transform(context.Background(), ComponentIdentity{}); err == nil {
		t.Fatal("a component without a content digest was accepted")
	}
}
