# Hachidori

Hachidori is a local semantic inference runtime for fast, low-cost, typed semantic observations.

It keeps the machine-learning runtime resident on an accelerator host, exposes a small transport-neutral HTTP contract, and lets higher-level products decide what to do with the returned observations.

Hachidori does **not** own orchestration policy, repository governance, workspace authority, or agent lifecycle. Its job is narrower: turn text or state into small typed semantic signals quickly and reproducibly.

See [docs/architecture.md](docs/architecture.md) for the target architecture.

## Quick start

```sh
go build -o hachidori ./cmd/hachidori
export HACHIDORI_HOME=/path/to/hachidori-home   # the only mutable root
./hachidori setup --device cuda                 # or --device cpu
./hachidori doctor
./hachidori serve                               # 127.0.0.1:7843
# or, with the host-local dashboard and SSH tunnel launcher:
./hachidori dashboard                           # API 127.0.0.1:7843, UI http://127.0.0.1:7844/
# or, on Windows, the same dashboard in a native WebView2 window:
hachidori.exe desktop --home D:\Hachidori        # needs the WebView2 Runtime; one per user

# caller side
export HACHIDORI_ENDPOINT=http://127.0.0.1:7843
./hachidori decide examples/decide.json
./hachidori benchmark dataset.jsonl
```

- [docs/runtime.md](docs/runtime.md): CLI, HTTP API, worker lifecycle, `HACHIDORI_HOME` layout, evaluation format.
- [docs/certification.md](docs/certification.md): certification levels and exact commands for a GPU host.

Verification: `gofmt -l . && go vet ./... && go test -race ./...`
