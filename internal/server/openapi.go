package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strconv"

	"github.com/yohn-jp/hachidori/internal/api"
	"github.com/yohn-jp/hachidori/internal/worker"
)

// OpenAPIPath serves the OpenAPI description of the public v1 HTTP API.
const OpenAPIPath = "/openapi.json"

// obj is a JSON object of the OpenAPI document. The document is assembled
// from the api package's limits and error classes and from this package's
// status mapping, so those values have one definition; the field names and
// requiredness of every schema are checked against the Go types in tests.
type obj = map[string]any

// openAPIDocument is the marshaled document. It describes the API contract
// only (no provider, model or runtime identity), so it is a constant of the
// build and is encoded once.
var openAPIDocument = mustMarshalOpenAPI()

func mustMarshalOpenAPI() []byte {
	b, err := json.MarshalIndent(buildOpenAPI(), "", "  ")
	if err != nil {
		panic("openapi: " + err.Error())
	}
	return append(b, '\n')
}

func serveOpenAPI(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(openAPIDocument)
}

func ref(name string) obj { return obj{"$ref": "#/components/schemas/" + name} }

func str(desc string) obj { return obj{"type": "string", "description": desc} }

func num(desc string) obj { return obj{"type": "number", "description": desc} }

func schemaConst() obj {
	return obj{"type": "string", "const": api.SchemaV1, "description": "Contract identifier; always \"" + api.SchemaV1 + "\"."}
}

// object builds a closed object schema.
func object(desc string, props obj, required ...string) obj {
	o := obj{"type": "object", "description": desc, "properties": props, "additionalProperties": false}
	if len(required) > 0 {
		o["required"] = required
	}
	return o
}

// openObject builds an object schema whose keys are not enumerated.
func openObject(desc string) obj {
	return obj{"type": "object", "description": desc, "additionalProperties": true}
}

func jsonContent(schema obj) obj { return obj{"application/json": obj{"schema": schema}} }

func buildOpenAPI() obj {
	return obj{
		"openapi": "3.1.0",
		"info": obj{
			"title":       "Hachidori HTTP API",
			"version":     api.SchemaV1,
			"summary":     "Typed semantic decisions over a resident local inference worker.",
			"description": "Public host-local HTTP contract of Hachidori. It describes the typed-decision API independent of the active model or provider; the runtime, model and provider identity of a running instance are reported by GET /v1/status. Every request and response body carries the \"" + api.SchemaV1 + "\" schema identifier. The API is loopback-only: a request whose Host header is not a loopback address, or a POST that carries a cross-origin Origin or Sec-Fetch-Site header, is refused with 403 and a text/plain body.",
			"license":     obj{"name": "MIT", "identifier": "MIT"},
		},
		"servers": []obj{{"url": "http://" + DefaultListen, "description": "Default loopback address of `hachidori serve`."}},
		"tags": []obj{
			{"name": "decide", "description": "Typed semantic decisions."},
			{"name": "runtime", "description": "Readiness and runtime status."},
			{"name": "discovery", "description": "API description."},
		},
		"paths": obj{
			"/v1/decide": obj{"post": obj{
				"operationId": "decide",
				"tags":        []string{"decide"},
				"summary":     "Answer questions about one state.",
				"description": "Asks every question against one state and returns one result per question, in request order. Without model, the default resident answers (unchanged). With model, exactly that resident answers and the response names it in served; a model that is not resident is a request_invalid error and a resident that is not ready is a not_ready error, never answered by another model. With route \"auto\", the runtime's deterministic routing policy answers each question from its first-path resident and hands off only the questions the policy selects; the response names the resident that produced each final result and why in routing, and a routed request that its policy cannot answer is a routing_failed error, never a weaker result. route cannot be combined with model. Unknown fields are rejected.",
				"requestBody": obj{"required": true, "content": jsonContent(ref("DecideRequest"))},
				"responses":   postResponses(ref("DecideResponse"), "The question results."),
			}},
			"/v1/decide/batch": obj{"post": obj{
				"operationId": "decideBatch",
				"tags":        []string{"decide"},
				"summary":     "Answer questions about several states.",
				"description": "Runs independent decide requests; responses are aligned with requests by index. Requests sharing a question set share forward passes. A batch is served by one resident: model on the batch (or the same model on its requests) targets it exactly as for /v1/decide, and route \"auto\" routes every request of the batch by policy as for /v1/decide. Unknown fields are rejected.",
				"requestBody": obj{"required": true, "content": jsonContent(ref("BatchRequest"))},
				"responses":   postResponses(ref("BatchResponse"), "One response per request, in request order."),
			}},
			"/health": obj{"get": obj{
				"operationId": "health",
				"tags":        []string{"runtime"},
				"summary":     "Report readiness.",
				"description": "200 once the worker is ready to answer; 503 with the same body while it is starting, restarting, failed or stopped.",
				"responses": obj{
					"200": obj{"description": "Ready.", "content": jsonContent(ref("Health"))},
					"503": obj{"description": "Not ready.", "content": jsonContent(ref("Health"))},
					"403": forbidden(),
				},
			}},
			"/v1/status": obj{"get": obj{
				"operationId": "status",
				"tags":        []string{"runtime"},
				"summary":     "Report runtime, worker and counter status.",
				"description": "The runtime status document, including the active runtime, model and provider. For a multi-resident runtime residents lists every resident with its own state; runtime and worker stay the default resident's. When a routing policy is configured, routing reports it with its counters per reason code and per resident. It never waits for an inference.",
				"responses": obj{
					"200": obj{"description": "The status document.", "content": jsonContent(ref("Status"))},
					"403": forbidden(),
				},
			}},
			OpenAPIPath: obj{"get": obj{
				"operationId": "openapi",
				"tags":        []string{"discovery"},
				"summary":     "Fetch this OpenAPI description.",
				"responses": obj{
					"200": obj{"description": "This OpenAPI 3.1 document.", "content": jsonContent(obj{"type": "object"})},
					"403": forbidden(),
				},
			}},
		},
		"components": obj{"schemas": schemas()},
	}
}

func forbidden() obj {
	return obj{
		"description": "Refused by the host-local boundary: the Host header is not a loopback address, or (POST only) the request carries a cross-origin Origin or a Sec-Fetch-Site other than same-origin or none.",
		"content":     obj{"text/plain": obj{"schema": obj{"type": "string"}}},
	}
}

// postResponses lists the success response and every documented error status
// of a decide operation. The statuses come from the one class-to-status map
// the handlers use; each response pins the error class it carries.
func postResponses(success obj, desc string) obj {
	res := obj{
		"200": obj{"description": desc, "content": jsonContent(success)},
		"403": forbidden(),
	}
	for _, c := range errorClasses() {
		code := strconv.Itoa(statusFor[c])
		res[code] = obj{
			"description": errorMeaning[c],
			"content": jsonContent(obj{"allOf": []obj{ref("ErrorBody"), {
				"properties": obj{"error": obj{"properties": obj{"class": obj{"const": c}}}},
			}}}),
		}
	}
	return res
}

var errorMeaning = map[string]string{
	api.ErrRequestInvalid:  "request_invalid: schema or limit violation, unknown field, malformed JSON, or a model that is not resident.",
	api.ErrNotReady:        "not_ready: the worker (or the targeted resident) is starting, restarting, failed or stopped.",
	api.ErrCapacity:        "capacity: more requests are queued or in flight than the runtime accepts.",
	api.ErrInferenceFailed: "inference_failed: the healthy worker failed this request.",
	api.ErrWorkerFailure:   "worker_failure: the worker crashed, hung or violated its protocol.",
	api.ErrRoutingFailed:   "routing_failed: a routed request (route \"auto\") that its policy could not answer: no routing policy is configured, the first-path resident was unavailable or failed, or a required handoff target was unavailable or failed. The message starts with a stable code (no_routing_policy, first_path_failed, required_handoff_failed). No result is returned and no other resident answers instead.",
}

func schemas() obj {
	return obj{
		"Question": object("One typed semantic question. Only the \"choice\" type exists in v1.", obj{
			"id":   obj{"type": "string", "minLength": 1, "description": "Caller-chosen identifier, unique within the request and echoed in the result."},
			"type": obj{"type": "string", "const": "choice", "description": "Question type; only \"choice\" is supported."},
			"instructions": obj{"type": "string", "minLength": 1, "maxLength": api.MaxInstructionSize,
				"description": fmt.Sprintf("What to decide. Must not be blank; at most %d bytes of UTF-8 (maxLength bounds characters, which is a necessary condition).", api.MaxInstructionSize)},
			"choices": obj{"type": "array", "minItems": 2, "maxItems": api.MaxChoices, "uniqueItems": true,
				"items":       obj{"type": "string", "minLength": 1},
				"description": "Candidate labels, non-empty and unique."},
			"descriptions": obj{"type": "object", "additionalProperties": obj{"type": "string"},
				"description": "Optional per-choice description, keyed by a label present in choices."},
		}, "id", "type", "instructions", "choices"),

		"DecideRequest": decideRequest(true),
		"BatchItem":     decideRequest(false),

		"Result": object("One typed observation for one question.", obj{
			"id":         str("The question id."),
			"type":       obj{"type": "string", "const": "choice", "description": "Question type."},
			"choice":     str("The most probable choice label."),
			"confidence": obj{"type": "number", "minimum": 0, "maximum": 1, "description": "Calibrated probability mass on the reported choice (max p)."},
			"probabilities": obj{"type": "object", "additionalProperties": obj{"type": "number", "minimum": 0, "maximum": 1},
				"description": "Probability per choice label."},
		}, "id", "type", "choice", "confidence", "probabilities"),

		"Timing": object("Server-side latency in milliseconds.", obj{
			"inference_ms": obj{"type": "number", "minimum": 0, "description": "Time spent inside the resident worker."},
			"total_ms":     obj{"type": "number", "minimum": 0, "description": "Total handling time, including queueing."},
		}, "inference_ms", "total_ms"),

		"DecideResponse": object("One result per question, in request order. timing is present on a /v1/decide response and absent on the entries of a batch response.", obj{
			"schema":  schemaConst(),
			"results": obj{"type": "array", "items": ref("Result"), "description": "Results in question order, ids preserved."},
			"timing":  ref("Timing"),
			"served":  ref("Served"),
			"routing": ref("Routing"),
		}, "schema", "results"),

		"BatchRequest": object("Independent decide requests.", obj{
			"schema": schemaConst(),
			"requests": obj{"type": "array", "minItems": 1, "maxItems": api.MaxBatchRequests, "items": ref("BatchItem"),
				"description": "Decide requests; each entry is validated like a /v1/decide request."},
			"model": modelRef(),
			"route": routeRef(),
		}, "schema", "requests"),

		"BatchResponse": object("Responses aligned with the request entries by index.", obj{
			"schema":    schemaConst(),
			"responses": obj{"type": "array", "items": ref("DecideResponse"), "description": "One response per request entry, in request order."},
			"timing":    ref("Timing"),
			"served":    ref("Served"),
			"routing":   ref("Routing"),
		}, "schema", "responses", "timing"),

		"Routing": object("Provenance of a routed request (route \"auto\"): the policy that decided, which resident produced each final result and why, the handoff count and each resident's latency contribution. providers is present on a /v1/decide response and on a batch response, not on the entries of a batch.", obj{
			"mode":   obj{"type": "string", "const": api.RouteAuto, "description": "Routing mode; always \"" + api.RouteAuto + "\"."},
			"policy": ref("PolicyRef"),
			"results": obj{"type": "array", "items": ref("RoutedResult"),
				"description": "One entry per question, in question order. Absent on a batch response, whose entries carry their own."},
			"handoffs":  obj{"type": "integer", "minimum": 0, "description": "Results replaced by a handoff."},
			"providers": obj{"type": "array", "items": ref("ProviderTiming"), "description": "Residents that served a worker call for this request, in policy order."},
		}, "mode", "policy", "handoffs"),

		"PolicyRef": object("A routing policy by id and canonical content digest.", obj{
			"id":     str("Policy id."),
			"sha256": str("\"sha256:\" + hex SHA-256 of the policy's canonical encoding."),
		}, "id", "sha256"),

		"RoutedResult": object("The routing provenance of one final result.", obj{
			"id":         str("The question id."),
			"served":     ref("Served"),
			"reason":     obj{"type": "string", "enum": routeReasons(), "description": "Stable reason code: why this resident produced the final result."},
			"profile":    str("The measurement family the policy routed the question by; absent for a question the policy routes by default."),
			"first_path": ref("FirstPath"),
		}, "id", "served", "reason"),

		"FirstPath": object("The first-path observation a handoff replaced. Present only on a handed-off result.", obj{
			"model":      str("Catalog model identity of the first-path resident."),
			"provider":   str("Provider kind of the first-path resident."),
			"choice":     str("The first-path choice."),
			"confidence": obj{"type": "number", "minimum": 0, "maximum": 1, "description": "The first-path confidence."},
		}, "model", "provider", "choice", "confidence"),

		"ProviderTiming": object("One resident's contribution to a routed request.", obj{
			"model":        str("Catalog model identity."),
			"provider":     str("Provider kind."),
			"calls":        obj{"type": "integer", "minimum": 0, "description": "Worker calls it served for this request."},
			"questions":    obj{"type": "integer", "minimum": 0, "description": "Questions in those calls."},
			"inference_ms": obj{"type": "number", "minimum": 0, "description": "Inference time of those calls in milliseconds."},
		}, "model", "provider", "calls", "questions", "inference_ms"),

		"RoutingStatus": object("The routing policy and its counters since the runtime began serving.", obj{
			"policy":           ref("PolicyRef"),
			"requests":         obj{"type": "integer", "description": "Routed requests answered."},
			"failures":         obj{"type": "integer", "description": "Routed requests that failed with routing_failed."},
			"questions":        obj{"type": "integer", "description": "Questions answered by routed requests."},
			"handoffs":         obj{"type": "integer", "description": "Results replaced by a handoff."},
			"handoff_failures": obj{"type": "integer", "description": "Questions whose optional handoff failed and kept the first-path result."},
			"reasons":          obj{"type": "object", "additionalProperties": obj{"type": "integer"}, "description": "Final results by reason code."},
			"providers":        obj{"type": "array", "items": ref("ProviderStatus"), "description": "Every resident the policy names, in policy order."},
			"calibration":      ref("Calibration"),
		}, "policy", "requests", "failures", "questions", "handoffs", "handoff_failures", "reasons", "providers"),

		"ProviderStatus": object("One resident's contribution to routed requests.", obj{
			"model":                str("Catalog model identity."),
			"provider":             str("Provider kind."),
			"first_path_questions": obj{"type": "integer", "description": "Questions it answered first."},
			"handoff_questions":    obj{"type": "integer", "description": "Questions it received by handoff."},
			"final_results":        obj{"type": "integer", "description": "Results it produced that were returned."},
			"calls":                obj{"type": "integer", "description": "Worker calls it served for answered routed requests."},
			"inference_ms_total":   num("Inference time of those calls in milliseconds."),
		}, "model", "provider", "first_path_questions", "handoff_questions", "final_results", "calls", "inference_ms_total"),

		"Calibration": object("The resident comparison evidence the policy's thresholds were verified against. Present only when the runtime was started with it.", obj{
			"evidence_sha256": str("\"sha256:\" + hex SHA-256 of the comparison report."),
			"dataset_sha256":  str("Digest of the dataset the evidence was measured on."),
			"rules":           obj{"type": "array", "items": ref("CalibratedRule"), "description": "The evidence slice behind each rule that routes on confidence or margin."},
		}, "evidence_sha256", "dataset_sha256", "rules"),

		"CalibratedRule": object("The evidence behind one threshold rule.", obj{
			"family":       str("The rule's measurement family; absent for the default rule."),
			"model":        str("The rule's first-path model."),
			"observations": obj{"type": "integer", "description": "Observations of that model in that family in the evidence."},
		}, "model", "observations"),

		"Served": object("The resident that answered a directly targeted request, by its stable Hachidori catalog identity. No provider prompt or tokenization detail is exposed.", obj{
			"model":    str("Catalog model identity."),
			"provider": str("Provider kind that loads the model."),
		}, "model", "provider"),

		"ErrorBody": object("Structured error envelope; an error never carries results.", obj{
			"schema": schemaConst(),
			"error":  ref("ErrorInfo"),
		}, "schema", "error"),

		"ErrorInfo": object("One failure.", obj{
			"class":   obj{"type": "string", "enum": errorClasses(), "description": "Stable error class."},
			"message": str("Human-readable detail. For inference_failed and worker_failure it is the worker's own text, redacted of local paths and credentials and at most 1 KiB."),
		}, "class", "message"),

		"Health": object("Readiness.", obj{
			"ready": obj{"type": "boolean", "description": "True when the worker can answer requests."},
			"state": obj{"type": "string", "enum": workerStates(), "description": "Worker lifecycle state."},
		}, "ready", "state"),

		"Status": object("Runtime status. Runtime, model and provider identity belong here, not to the decision schemas.", obj{
			"schema":   schemaConst(),
			"runtime":  ref("Runtime"),
			"uptime_s": obj{"type": "integer", "minimum": 0, "description": "Seconds since the API began serving."},
			"worker":   ref("Worker"),
			"residents": obj{"type": "array", "items": ref("ResidentStatus"),
				"description": "Every resident of a multi-resident runtime, default first, each with its own independent state. Absent for a single worker."},
			"routing": ref("RoutingStatus"),
		}, "schema", "runtime", "uptime_s", "worker"),

		"ResidentStatus": object("One resident: its stable catalog identity and the status of its own supervised worker. A failed or stopped resident does not change another's entry.", obj{
			"model":    str("Catalog model identity; the value of model that targets this resident."),
			"provider": str("Provider kind that loads the model."),
			"default":  obj{"type": "boolean", "description": "True for the resident that answers requests without a model."},
			"running":  obj{"type": "boolean", "description": "True while the resident's supervision is running (false after an operator stop)."},
			"status":   ref("Status"),
		}, "model", "provider", "running", "status"),

		"Runtime": object("The active runtime.", obj{
			"home":     str("The HACHIDORI_HOME path."),
			"runtime":  str("Active runtime version."),
			"model_id": str("Catalog model identity."),
			"model":    str("Model directory: <repo>/<revision>."),
			"device":   str("Requested device."),
		}, "home", "runtime", "model_id", "model", "device"),

		"Worker": object("Supervised worker snapshot.", obj{
			"state":              obj{"type": "string", "enum": workerStates(), "description": "Worker lifecycle state."},
			"phase":              str("Startup phase."),
			"ready":              obj{"type": "boolean", "description": "True when state is ready."},
			"pid":                obj{"type": "integer", "description": "Worker process id, when running."},
			"starts":             obj{"type": "integer", "description": "Worker starts since serving began."},
			"restarts_in_window": obj{"type": "integer", "description": "Restarts within the restart-policy window."},
			"ready_since":        obj{"type": "string", "format": "date-time", "description": "When the worker became ready."},
			"provider":           openObject("Provider details reported by the worker (versions, device, load and warmup timing); keys are provider-defined."),
			"accelerator":        openObject("Accelerator memory statistics; keys are provider-defined."),
			"accelerator_stale":  obj{"type": "boolean", "description": "True when accelerator was taken before the request now in flight."},
			"last_failure":       ref("Failure"),
			"requests":           obj{"type": "integer", "description": "Requests handled."},
			"errors":             obj{"type": "object", "additionalProperties": obj{"type": "integer"}, "description": "Error counts by class."},
			"queue_depth":        obj{"type": "integer", "description": "Requests queued or in flight."},
			"queue_limit":        obj{"type": "integer", "description": "Maximum queued or in-flight requests."},
			"inference_p50_ms":   num("Median inference latency in milliseconds."),
			"inference_p95_ms":   num("95th-percentile inference latency in milliseconds."),
		}, "state", "phase", "ready", "starts", "restarts_in_window", "requests", "errors", "queue_depth", "queue_limit", "inference_p50_ms", "inference_p95_ms"),

		"Failure": object("The worker's last failure.", obj{
			"class":       str("Failure class."),
			"message":     str("Failure detail."),
			"stderr_tail": obj{"type": "array", "items": obj{"type": "string"}, "description": "Last worker stderr lines."},
		}, "class", "message"),
	}
}

// decideRequest describes a decide request. A batch entry may omit schema
// (it defaults to the batch's), so schema is required only for /v1/decide.
func decideRequest(schemaRequired bool) obj {
	desc := "Asks every question against one state."
	req := []string{"state", "questions"}
	if schemaRequired {
		req = append([]string{"schema"}, req...)
	} else {
		desc = "One decide request of a batch; schema may be omitted and then defaults to \"" + api.SchemaV1 + "\"."
	}
	return object(desc, obj{
		"schema": schemaConst(),
		"state": obj{"type": "string", "minLength": 1, "maxLength": api.MaxStateBytes,
			"description": fmt.Sprintf("The text to decide about. Must not be blank; at most %d bytes of UTF-8 (maxLength bounds characters, which is a necessary condition).", api.MaxStateBytes)},
		"questions": obj{"type": "array", "minItems": 1, "maxItems": api.MaxQuestions, "items": ref("Question"),
			"description": "Questions to answer; ids must be unique within the request."},
		"options": openObject("Reserved. Accepted and currently has no effect."),
		"model":   modelRef(),
		"route":   routeRef(),
	}, req...)
}

// errorClasses lists the classes the handlers can answer with.
func errorClasses() []string {
	classes := make([]string, 0, len(statusFor))
	for c := range statusFor {
		classes = append(classes, c)
	}
	sort.Strings(classes)
	return classes
}

func workerStates() []string {
	return []string{worker.StateStarting, worker.StateReady, worker.StateRestarting, worker.StateFailed, worker.StateStopped}
}

// routeRef is the optional routing selector of a decide request or batch.
func routeRef() obj {
	return obj{"type": "string", "const": api.RouteAuto,
		"description": "Optional. \"" + api.RouteAuto + "\" routes the request by the runtime's deterministic routing policy: each question is answered by its first-path resident and handed off to the alternate resident only when the policy requires it. Omitted, the request takes the default route (or model). It cannot be combined with model. A runtime without a routing policy answers routing_failed."}
}

// routeReasons are the stable reason codes of a routed result.
func routeReasons() []string {
	r := []string{api.ReasonFirstPathOnly, api.ReasonFirstPathKept, api.ReasonHandoffAlways, api.ReasonHandoffChoice,
		api.ReasonHandoffLowConf, api.ReasonHandoffLowMargin, api.ReasonHandoffFailed}
	sort.Strings(r)
	return r
}

// modelRef is the optional direct-selection property of a decide request.
func modelRef() obj {
	return obj{"type": "string", "minLength": 1, "maxLength": 128,
		"description": "Optional. A stable Hachidori catalog model ID (a model value listed in GET /v1/status), never a repository or revision. Omitted, the default resident answers. An explicitly empty string is not an omitted selector: it is request_invalid. Present and non-empty, only that resident answers; the response names it in served. A model that is not resident is request_invalid and one that is not ready is not_ready: the request is never redirected to another resident."}
}
