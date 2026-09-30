<p align="center">
  <img src="./assets/branding/hero.png" alt="Hachidori — Local AI Runtime. Run AI models locally. Fast, flexible, and yours." width="100%">
</p>

<p align="center">
  <a href="https://github.com/yohn-jp/hachidori/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/yohn-jp/hachidori/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/yohn-jp/hachidori/releases"><img alt="Release" src="https://img.shields.io/github/v/release/yohn-jp/hachidori?include_prereleases"></a>
  <a href="./go.mod"><img alt="Go" src="https://img.shields.io/github/go-mod/go-version/yohn-jp/hachidori"></a>
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/github/license/yohn-jp/hachidori"></a>
</p>

# Hachidori

**Local semantic inference runtime**

Hachidori is a local semantic inference runtime for fast, low-cost, typed semantic observations.

It keeps the machine-learning runtime resident on an accelerator host, exposes a small transport-neutral HTTP contract, and lets higher-level products decide what to do with the returned observations.

Hachidori does **not** own orchestration policy, repository governance, workspace authority, or agent lifecycle. Its job is narrower: turn text or state into small typed semantic signals quickly and reproducibly.

See [docs/architecture.md](docs/architecture.md) for the target architecture.

## Windows quick start

Development releases publish two Windows amd64 assets on the
[Releases page](https://github.com/yohn-jp/hachidori/releases):

- `hachidori-windows-amd64.exe`
- `hachidori-windows-amd64.exe.sha256`

Verify the downloaded executable before running it:

```powershell
$expected = (Get-Content .\hachidori-windows-amd64.exe.sha256).Split()[0].ToLowerInvariant()
$actual = (Get-FileHash .\hachidori-windows-amd64.exe -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "Hachidori SHA-256 mismatch" }
```

Then double-click `hachidori-windows-amd64.exe` (or run it with no arguments).

On first launch Hachidori:

1. opens the Windows desktop UI;
2. asks for one storage root for its managed runtime, models, caches, logs, and state;
3. lets you choose CUDA or CPU and install the private runtime/model;
4. starts and warms the resident worker;
5. reaches READY and remains available from the system tray.

Later launches rediscover the selected home automatically. Closing the main
window hides Hachidori to the tray; **Quit Hachidori** stops the runtime and
exits. Microsoft WebView2 Runtime is required and is not installed silently by
Hachidori.

## CLI / development quick start

Explicit CLI commands remain available for development, automation, and
certification:

```sh
go build -o hachidori ./cmd/hachidori
export HACHIDORI_HOME=/path/to/hachidori-home   # managed runtime/model state root
./hachidori setup --device cuda                 # or --device cpu
./hachidori doctor
./hachidori serve                               # 127.0.0.1:7843
# or, with the host-local dashboard and SSH tunnel launcher:
./hachidori dashboard                           # API 127.0.0.1:7843, UI http://127.0.0.1:7844/

# Windows: explicit desktop entry into the same composition as no-argument launch
hachidori.exe desktop --home D:\Hachidori

# caller side
export HACHIDORI_ENDPOINT=http://127.0.0.1:7843
./hachidori decide examples/decide.json
./hachidori benchmark dataset.jsonl
```

The dashboard, which the Windows desktop UI also hosts, adds runtime
status and Start/Stop/Restart, doctor, the SSH tunnel launcher, and three
operator surfaces: **Question Workbench**, **Experiment Runner**, and
**Error Explorer**.

- [docs/runtime.md](docs/runtime.md): CLI, HTTP API, worker lifecycle, `HACHIDORI_HOME` layout, evaluation format.
- [docs/certification.md](docs/certification.md): certification levels and exact commands for a GPU host.

Verification: `gofmt -l . && go vet ./... && go test -race ./...`
