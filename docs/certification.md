# Golden-path certification

Certification levels are reported separately. An environment that cannot run a
level reports it as **blocked / not checked**, never as passed.

| level | command | needs |
|---|---|---|
| unit / focused | `go vet ./... && go test -race ./...` | Go toolchain (the Python protocol test also uses a host `python3` when present, with stub `torch`/`laya`) |
| integration (real worker, real Laya) | `HACHIDORI_CERT_HOME=<home> go test ./internal/doctor -run TestCertifyRealProvider -v` | materialized home |
| real Laya provider | `hachidori doctor --home <home>` | materialized home |
| CUDA | same two commands with a home set up by `--device cuda` | NVIDIA GPU + driver ≥ 570 |
| benchmark | `hachidori benchmark --out report.json <coding-agent.jsonl>` | running `serve`, the external benchmark dataset |

## Windows RTX host (PowerShell)

```powershell
# 1. build (or copy) the single executable
go build -o hachidori.exe ./cmd/hachidori

# 2. materialize runtime + model (network needed only here). Use a clean
#    home; no system Python, pip or uv is needed (Hachidori bootstraps its
#    pinned uv into $env:HACHIDORI_HOME\tools\uv\0.12.19\uv.exe).
$env:HACHIDORI_HOME = "D:\Hachidori-test"
.\hachidori.exe setup --device cuda
# expect: "private uv 0.12.19 ready", uv python install / venv / sync lines,
#         "runtime cu128-<digest> published", "active: runtime=cu128-<digest> …"

# 3. doctor: home, runtime, model digests, isolation, provider import,
#    CUDA, worker start/load/warmup, real HTTP smoke inference
.\hachidori.exe doctor

# 4. integration certification against the real worker on CUDA
$env:HACHIDORI_CERT_HOME = $env:HACHIDORI_HOME
go test ./internal/doctor -run TestCertifyRealProvider -v -count=1

# 5. serve (loopback only). To show serving does not use uv or package
#    resolution, first move the private uv away and cut the network, e.g.
#    Rename-Item $env:HACHIDORI_HOME\tools\uv\0.12.19\uv.exe uv.exe.off
.\hachidori.exe serve
```

In another shell on the host, or from the NixOS VM through an SSH local
forward (`ssh -N -L 7843:127.0.0.1:7843 <windows-host>`):

```sh
export HACHIDORI_ENDPOINT=http://127.0.0.1:7843
hachidori status                      # runtime.model_id=laya-base, worker.state=ready, provider.device=cuda, provider.device_name, accelerator memory
hachidori decide examples/decide.json # repeat: worker.pid and worker.starts in status stay unchanged
hachidori benchmark --warmup 5 --passes 3 --out coding-agent-report.json <coding-agent.jsonl>
```

Evidence to record: `doctor` output, the integration test log (device, load and
warmup ms, warm p50/p95, same pid over 20 requests), `status` before and after
the benchmark (pid/starts unchanged, request counters), and the benchmark report
compared with the architecture §12 baseline (accuracy ≈ 0.8333, ECE ≈ 0.0784,
mean confidence ≈ 0.8215, p50 ≈ 25.3 ms, p95 ≈ 30.5 ms).

Note: the §12 baseline was measured outside Hachidori; its checkpoint and
question wording are not recorded in this repository. If the benchmark uses a
different checkpoint than the pinned `convaiinnovations/laya` revision, the
accuracy comparison is not like-for-like.

## Linux GPU host

Same commands with `export HACHIDORI_HOME=/srv/hachidori`. On NixOS the private uv and
CPython (the uv-managed python-build-standalone build, glibc) need a dynamic loader (`nix-ld` or an
FHS environment) and `LD_LIBRARY_PATH` must include the driver libraries
(`/run/opengl-driver/lib`).

## Windows desktop shell (manual, real Windows only)

Generic CI and cross-compilation cannot open a WebView2 window; report these
items as **not checked** unless they were run on a real Windows desktop with
the resulting `hachidori.exe`.

```powershell
go test ./internal/desktop -v -count=1          # Windows-only tests: instance guard, runtime detection
.\hachidori.exe desktop --home D:\Hachidori-test
```

1. With the WebView2 Runtime installed: a native window titled "Hachidori"
   shows the dashboard (status, Start/Stop/Restart, doctor, tunnel), and the
   console prints `WebView2 Runtime <version>`. Start/Stop/Restart work.
2. A second `hachidori.exe desktop` by the same user exits with "already
   running" and starts no worker (`status` pid unchanged).
3. F12 / Ctrl+Shift+I open no DevTools and right-click shows no browser
   context menu. The navigation allow-list itself is covered by unit tests
   (`TestPolicyAllowsOnlyTheDashboardOrigin`); any navigation the window
   cancels is logged on the console as `desktop blocked`.
4. Closing the window stops the process: the API and dashboard ports are free
   and no `python` worker or `msedgewebview2.exe` child of this run remains.
5. On a machine without the WebView2 Runtime (or with it uninstalled), the
   command fails before any worker starts with the WebView2 diagnostic, and
   nothing is downloaded.
