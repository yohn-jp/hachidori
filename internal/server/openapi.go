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
				"description": "Asks every question against one state and returns one result per question, in request order. Unknown fields are rejected.",
				"requestBody": obj{"required": true, "content": jsonContent(ref("DecideRequest"))},
				"responses":   postResponses(ref("DecideResponse"), "The question results."),
			}},
			"/v1/decide/batch": obj{"post": obj{
				"operationId": "decideBatch",
				"tags":        []string{"decide"},
				"summary":     "Answer questions about several states.",
				"description": "Runs independent decide requests; responses are aligned with requests by index. Requests sharing a question set share forward passes. Unknown fields are rejected.",
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
				"description": "The runtime status document, including the active runtime, model and provider. It never waits for an inference.",
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
	api.ErrRequestInvalid:  "request_invalid: schema or limit violation, unknown field, or malformed JSON.",
	api.ErrNotReady:        "not_ready: the worker is starting, restarting or failed.",
	api.ErrCapacity:        "capacity: more requests are queued or in flight than the runtime accepts.",
	api.ErrInferenceFailed: "inference_failed: the healthy worker failed this request.",
	api.ErrWorkerFailure:   "worker_failure: the worker crashed, hung or violated its protocol.",
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
		}, "schema", "results"),

		"BatchRequest": object("Independent decide requests.", obj{
			"schema": schemaConst(),
			"requests": obj{"type": "array", "minItems": 1, "maxItems": api.MaxBatchRequests, "items": ref("BatchItem"),
				"description": "Decide requests; each entry is validated like a /v1/decide request."},
		}, "schema", "requests"),

		"BatchResponse": object("Responses aligned with the request entries by index.", obj{
			"schema":    schemaConst(),
			"responses": obj{"type": "array", "items": ref("DecideResponse"), "description": "One response per request entry, in request order."},
			"timing":    ref("Timing"),
		}, "schema", "responses", "timing"),

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
		}, "schema", "runtime", "uptime_s", "worker"),

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
