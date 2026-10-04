package requesthistory

import (
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

func TestLifecycleSeparatesRequestsFromInputItems(t *testing.T) {
	store := New()
	start := time.Date(2026, 10, 4, 1, 2, 3, 0, time.UTC)
	first := store.Begin("/v1/decide/batch", start, Identity{Runtime: "rt", Model: "m", Device: "cuda", Variant: "v"})
	first.SetInput(map[string]any{"requests": []string{"a", "b"}}, 12, 3, 2)
	first.Admit()
	first.Queued(start.Add(2 * time.Millisecond))
	second := store.Begin("/v1/decide", start.Add(time.Millisecond), Identity{})
	second.SetInput(map[string]string{"state": "c"}, 1, 1, 1)
	second.Admit()
	second.Queued(start.Add(3 * time.Millisecond))

	view := store.List()
	if view.QueueDepth != 2 || view.InFlight != 0 || view.ActiveItems != 3 {
		t.Fatalf("queued metrics %+v; want 2 requests, 0 in flight, 3 items", view)
	}

	first.Started(start.Add(7 * time.Millisecond))
	view = store.List()
	if view.QueueDepth != 1 || view.InFlight != 1 || view.ActiveItems != 3 {
		t.Fatalf("running metrics %+v; want 1 queued request, 1 in flight, 3 items", view)
	}
	first.Complete(start.Add(17*time.Millisecond), map[string]any{"results": []string{"yes", "no"}}, 200)
	view = store.List()
	if view.QueueDepth != 1 || view.InFlight != 0 || view.ActiveItems != 1 {
		t.Fatalf("completion metrics %+v; want 1 queued request, 0 in flight, 1 item", view)
	}
	detail, ok := store.Get(first.entry.summary.ID)
	if !ok || detail.State != "completed" || detail.StateBytes != 12 || detail.QuestionCount != 3 || detail.ItemCount != 2 {
		t.Fatalf("request detail %+v, found=%v", detail, ok)
	}
	if detail.QueueWaitMS != 5 || detail.ServiceMS != 10 || detail.TotalMS != 17 || detail.Runtime.Model != "m" || detail.Runtime.Variant != "v" {
		t.Fatalf("timing or identity %+v", detail.Summary)
	}
	if string(detail.Input) != `{"requests":["a","b"]}` || string(detail.Output) != `{"results":["yes","no"]}` {
		t.Fatalf("input/output %s %s", detail.Input, detail.Output)
	}

	second.Reject(start.Add(20*time.Millisecond), 429, "capacity", "full")
	view = store.List()
	if view.QueueDepth != 0 || view.InFlight != 0 || view.ActiveItems != 0 {
		t.Fatalf("terminal requests remain active: %+v", view)
	}
}

func TestRoutedRequestCountsOnlyItsActiveWorkerCall(t *testing.T) {
	store := New()
	start := time.Date(2026, 10, 4, 1, 2, 3, 0, time.UTC)
	entry := store.Begin("/v1/decide", start, Identity{})
	entry.SetInput(map[string]string{"state": "x"}, 1, 1, 1)
	entry.Admit()
	entry.Queued(start.Add(time.Millisecond))
	entry.Started(start.Add(3 * time.Millisecond))
	entry.Finished(start.Add(8 * time.Millisecond))
	view := store.List()
	if view.QueueDepth != 0 || view.InFlight != 0 || view.ActiveItems != 0 || view.Entries[0].State != "running" {
		t.Fatalf("between routed worker calls: %+v", view)
	}

	entry.Queued(start.Add(10 * time.Millisecond))
	entry.Started(start.Add(13 * time.Millisecond))
	entry.Finished(start.Add(18 * time.Millisecond))
	entry.Complete(start.Add(20*time.Millisecond), map[string]string{"choice": "yes"}, 200)
	detail, ok := store.Get(view.Entries[0].ID)
	if !ok || detail.State != "completed" || detail.QueueWaitMS != 5 || detail.ServiceMS != 10 || detail.TotalMS != 20 {
		t.Fatalf("routed timing %+v, found=%v", detail.Summary, ok)
	}
	if view = store.List(); view.QueueDepth != 0 || view.InFlight != 0 || view.ActiveItems != 0 {
		t.Fatalf("routed request remained active: %+v", view)
	}
}

func TestRetentionStaysBoundedWhileRequestsAreActive(t *testing.T) {
	store := New()
	now := time.Now().UTC()
	requests := make([]*Request, MaxRecords+5)
	for i := range requests {
		received := now.Add(time.Duration(i) * time.Nanosecond)
		r := store.Begin("/v1/decide", received, Identity{})
		r.SetInput(map[string]string{"state": "small"}, 5, 1, 1)
		r.Admit()
		r.Queued(received)
		r.Started(received)
		requests[i] = r
	}
	if got := len(store.List().Entries); got != MaxRecords {
		t.Fatalf("retained %d records while active, want %d", got, MaxRecords)
	}
	view := store.List()
	if view.QueueDepth != 0 || view.InFlight != len(requests) || view.ActiveItems != len(requests) {
		t.Fatalf("active counters lost evicted records: %+v", view)
	}
	if _, ok := store.Get(requests[0].entry.summary.ID); ok {
		t.Fatal("old active record was retained past MaxRecords")
	}
	retainedBytes := store.bytes
	requests[0].Complete(now.Add(time.Second), map[string]string{"choice": "yes"}, 200)
	if store.bytes != retainedBytes {
		t.Fatalf("completion of an evicted request changed retained bytes: %d to %d", retainedBytes, store.bytes)
	}
	view = store.List()
	if view.InFlight != len(requests)-1 || view.ActiveItems != len(requests)-1 {
		t.Fatalf("completion of evicted handle left phantom activity: %+v", view)
	}
	for _, r := range requests[1:] {
		r.Complete(now.Add(time.Second), nil, 200)
	}
	if len(store.order) > MaxRecords || store.bytes > MaxBytes {
		t.Fatalf("retention bounds exceeded: records=%d bytes=%d", len(store.order), store.bytes)
	}
}

func TestOversizedActiveInputIsOmittedWithoutExceedingByteLimit(t *testing.T) {
	store := New()
	now := time.Now().UTC()
	entry := store.Begin("/v1/decide", now, Identity{})
	entry.SetInput(map[string]string{"state": strings.Repeat("x", MaxBytes)}, MaxBytes, 1, 1)
	entry.Admit()
	entry.Queued(now)
	if store.bytes > MaxBytes {
		t.Fatalf("active input grew retained bytes to %d", store.bytes)
	}
	entry.Started(now.Add(time.Millisecond))
	entry.Complete(now.Add(2*time.Millisecond), map[string]string{"choice": "yes"}, 200)
	if store.bytes > MaxBytes {
		t.Fatalf("completion grew retained bytes to %d", store.bytes)
	}
	detail, ok := store.Get(entry.entry.summary.ID)
	if !ok || !detail.ContentLimited || len(detail.Input) != 0 || len(detail.Output) == 0 {
		t.Fatalf("bounded detail %+v, found=%v", detail.Summary, ok)
	}
	view := store.List()
	if view.QueueDepth != 0 || view.InFlight != 0 || view.ActiveItems != 0 {
		t.Fatalf("terminal oversized request remains active: %+v", view)
	}
}

func TestCallerControlledMetadataIsBounded(t *testing.T) {
	store := New()
	now := time.Now().UTC()
	large := strings.Repeat("界", MaxMetadataStringBytes)
	entry := store.Begin("/v1/decide", now, Identity{Runtime: large, Model: large, Device: large, Variant: large})
	entry.Reject(now.Add(time.Millisecond), 400, large, large)

	detail, ok := store.Get(entry.entry.summary.ID)
	if !ok {
		t.Fatal("bounded request metadata was not retained")
	}
	values := []string{detail.Runtime.Runtime, detail.Runtime.Model, detail.Runtime.Device,
		detail.Runtime.Variant, detail.ErrorClass, detail.ErrorMessage}
	for _, value := range values {
		if len(value) > MaxMetadataStringBytes || !utf8.ValidString(value) {
			t.Fatalf("metadata is not bounded valid UTF-8: bytes=%d valid=%v", len(value), utf8.ValidString(value))
		}
	}
	if !detail.ContentLimited {
		t.Fatal("truncated metadata did not report omitted content")
	}
}
