package desktopkit

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

// MaybeRunStub must be called first in the TestMain of a package that uses
// InstallFixture. When the binary was started under the name of the private
// interpreter ("python" or "python.exe", which is how the candidate launches
// its worker) it speaks the worker protocol and exits; otherwise it returns
// and the tests run.
//
// The stand-in answers the protocol the Go supervisor speaks (hello, phases,
// ready, then decide/stats/shutdown requests) and nothing else. It is not an
// inference engine: scenarios that use it prove who owns which process and
// endpoint, not what the model answers.
func MaybeRunStub() {
	name := strings.TrimSuffix(strings.ToLower(filepath.Base(os.Args[0])), ".exe")
	if name != "python" {
		return
	}
	os.Exit(runStub(os.Args[1:]))
}

func flagValue(args []string, name string) string {
	for i, a := range args {
		if a == name && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

func runStub(args []string) int {
	out := json.NewEncoder(os.Stdout)
	emit := func(v map[string]any) { _ = out.Encode(v) }
	emit(map[string]any{"event": "hello", "pid": os.Getpid()})
	for _, ph := range []string{"importing", "loading", "warming"} {
		emit(map[string]any{"event": "phase", "phase": ph})
	}
	emit(map[string]any{"event": "ready", "info": map[string]any{
		"provider": flagValue(args, "--provider"), "device": flagValue(args, "--device"), "stand_in": true,
	}})
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	for sc.Scan() {
		var req struct {
			ID    int64  `json:"id"`
			Op    string `json:"op"`
			Items []struct {
				Questions []struct {
					ID      string   `json:"id"`
					Choices []string `json:"choices"`
				} `json:"questions"`
			} `json:"items"`
		}
		if json.Unmarshal(sc.Bytes(), &req) != nil {
			continue
		}
		switch req.Op {
		case "shutdown":
			emit(map[string]any{"id": req.ID, "ok": true})
			return 0
		case "stats":
			emit(map[string]any{"id": req.ID, "ok": true, "stats": map[string]any{}})
		case "decide":
			results := make([][]map[string]any, 0, len(req.Items))
			for _, it := range req.Items {
				var rs []map[string]any
				for _, q := range it.Questions {
					choice := ""
					if len(q.Choices) > 0 {
						choice = q.Choices[0]
					}
					rs = append(rs, map[string]any{"id": q.ID, "type": "choice", "choice": choice, "confidence": 1.0})
				}
				results = append(results, rs)
			}
			emit(map[string]any{"id": req.ID, "ok": true, "results": results, "inference_ms": float64(os.Getpid())})
		default:
			emit(map[string]any{"id": req.ID, "ok": false, "error": map[string]any{"class": "request_invalid", "message": "unsupported operation"}})
		}
	}
	return 0
}
