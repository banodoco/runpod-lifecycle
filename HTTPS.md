# HTTPS pod transport

Select `transport="https"` in `RunPodConfig` or `api.create_pod`, or pass the
CLI's global `--transport https` before the command. SSH remains the default.
The HTTPS agent starts from the selected image's system `python3`, without pip.
Use an image with Python 3.10+ and `/bin/bash` (including stock `runpod/base`).
HTTPS reserves `8000/http`; its URL is
`https://<pod-id>-8000.proxy.runpod.net`. No raw TCP/SSH or proxy SSH tunnel is
used. Custom template overrides are rejected so they cannot replace the agent
start command. The generic default template is disabled for HTTPS configs.

Before creating any pod, lifecycle checks the chosen egress route. The H3
builder and claim script also preflight before recording allocation intent.
SSH clients with an HTTPS proxy configured fail conservatively with
`SSH egress unavailable; use --transport https`. Direct SSH clients probe
`ssh.runpod.io:22` with a five-second timeout; sites can select a known TCP
probe with `RUNPOD_SSH_PREFLIGHT_HOST` / `RUNPOD_SSH_PREFLIGHT_PORT`. This is a
route check, not proof that every future provider IP/port will be reachable.
HTTPS preflight accepts the expected provider 404 from an unallocated pod URL;
authentication/egress denials fail closed. Actual readiness requires token-authenticated
agent health on the allocated pod.

## Python usage

```python
from runpod_lifecycle import RunPodConfig, launch, HttpsPodTransport

config = RunPodConfig.from_env(transport="https")
# Keep config.https_token private. A fresh config generates a random token.
pod = await launch(config)
try:
    await pod.wait_ready()
    code, stdout, stderr = await pod.exec_ssh("uname -a")  # legacy name, HTTPS execution
    await pod.put_file(local_file, "/tmp/input.bin")
    await pod.get_file("/tmp/input.bin", local_copy)
finally:
    await pod.terminate()
```

For direct CPU creation, use `api.create_pod(..., compute_type="CPU",
transport="https", https_token=token)` and `HttpsPodTransport(pod_id, token)`.
CPU REST creation overrides Docker ENTRYPOINT/CMD; SDK creation uses
`docker_args` and needs the companion SDK quoting fix. Keep Luna's minimum-CUDA,
CPU-image and datacenter SDK patches when installing the companion checkout.

`HttpsPodTransport` implements `connect`, `wait_ready`, `execute_command`
(aliases `run` and `exec`), `put`, `get`, and `disconnect`/`close`. For resumable
command polling, use `start`, retain the returned job ID, then call `poll` with
stdout/stderr byte offsets and `cancel` when needed. Jobs survive HTTP request
loss and client disconnects; they are not reconstructed after an agent/pod
restart. A start ID is idempotent during an agent lifetime; surviving job spool
directories prevent silent reuse after restart.

The optional `base_url` must be the exact selected pod's proxy hostname
(`https://<pod-id>-<agent-port>.proxy.runpod.net`, TLS port 443). Only an empty
path or `/` is accepted; URL credentials, query strings, fragments and path
prefixes are rejected before the bearer is attached. Explicit loopback URLs
(`localhost`, `127.0.0.1`, `::1`) remain available for local tests, including
HTTP and a local server port. Redirects are rejected without contacting their
target, even for loopback tests.

For CLI reconnects, generate a token file once per claim without printing it:

```python
import secrets
from runpod_lifecycle.https import write_token_file
write_token_file("claim.token", secrets.token_urlsafe(32))
```

```sh
export RUNPOD_HTTPS_TOKEN_FILE="$PWD/claim.token"
runpod-lifecycle --transport https launch --image runpod/base:1.0.3-dev-fix-pytorch-version-verification-cuda1300-ubuntu2404 --detach
runpod-lifecycle --transport https exec POD_ID -- uname -a
```

Token files must be owned regular 0600 files, with no final symlink or hard link.
`RUNPOD_TRANSPORT=https` is also supported. Library callers reconnect with the
original token; never construct a new random token for an existing claim.
Tokens are not recoverable from public proxy URLs. Do not put them in shell
arguments, logs, patches, or ordinary handles.

## Module bootstrap observation policy

Trusted startup code can restrict the single-file agent through the optional
keyword on `runpod_lifecycle.https.agent_bootstrap`:

```python
from runpod_lifecycle.https import agent_bootstrap

boot = agent_bootstrap(token, observation_policy={
    "schema": "astrid.pod-observation.v1",
    "roots": ["/workspace/astrid/startup/attempt/observations"],
    "files": ["/workspace/h3-golden/releases/release/.astrid-engine/install.json"],
})
```

The exact fields are `schema`, `roots` (up to 8 curated absolute directories),
and `files` (up to 32 exact public file paths). Unknown fields, duplicate paths
(including across the arrays), empty combined allowlists, `/`, relative paths,
dot/traversal or empty components, backslashes and NUL are rejected. Paths may
be absent at startup. Only safe regular files can subsequently be downloaded;
symlinks in any component, hardlinked files and nonregular files are refused.
Directory traversal and the final open use directory descriptors with
`O_NOFOLLOW`; reads and metadata checks use the same opened file.

The bootstrap validates before producing boot material and passes public JSON
in `ASTRID_POD_OBSERVATION_B64`. The complete base64 value is limited to 16384
ASCII bytes (at most 12288 decoded bytes). Startup removes this variable along
with the existing token/source variables and validates before binding a listener.
Policy fields contain no credentials; the existing bearer stays in agent memory.
These are application limits, not provider payload qualification.

Observation mode authenticates before routing and allows only `GET /health`,
`POST /download-chunk` and `GET /files` for the configured files/descendants.
Other routes and methods return a generic refusal, including jobs, polling,
cancellation and uploads. Requests cannot register new paths or change policy.
Health remains `{"ok": true, "protocol": 1}` and describes agent readiness.
Existing JSON/range framing, representation identity and chunk SHA checks apply.
Use `HttpsPodTransport.connect()` and `get()`; observation needs no command or
upload operations. The default `observation_policy=None` retains the general
agent behavior described below.

The startup publisher must sanitize and close public log segments before
publication and keep private state outside every allowlisted root. The intended
publisher limits are 1 MiB per closed log segment and 8 MiB retained per stream
per attempt; this file-access policy does not implement log rotation/retention.
Startup supervisor/installer wiring remains separate. This keyword is only a
module bootstrap interface; provider-create/config/CLI observation options are
not implemented here.

## Wire protocol and limits

Every endpoint requires `Authorization: Bearer <claim-token>` and uses
constant-time authentication. Health returns protocol version 1. The full table
below describes the default general agent; observation restricts it as above.

| Request | Operation |
| --- | --- |
| `GET /health` | Authenticated readiness |
| `POST /jobs` | `{id, command}`; returns immediately with ID |
| `GET /jobs/<id>?stdout=<offset>&stderr=<offset>` | Status, exit code, base64 output and next byte offsets |
| `DELETE /jobs/<id>` | Process-group TERM followed by KILL after two seconds |
| `POST /uploads` | `{path, size, sha256, reserved}`; returns upload ID and committed offset |
| `PUT /uploads/<id>?offset=<offset>` | One Content-Length chunk, at most 1 MiB |
| `GET /uploads/<id>` | Committed offset, completion, sanitized verification error |
| `POST /uploads/<id>` | Start background SHA verification and atomic promotion; poll to completion |
| `GET /files?path=<absolute-path>` | Bounded `Range: bytes=start-end`; returns HTTP 206, Content-Range and ETag |
| `POST /download-chunk` | `{path, offset, etag?}`; JSON with bounded base64 data, SHA256, total size and file identity |

The client defaults to JSON download envelopes to avoid depending on a public
proxy preserving Range response framing. `get(..., use_ranges=True)` selects
HTTP 206 downloads. Both pin file identity across chunks and atomically promote
the local `.https-part` file only on success. Uploads resume across client and
agent restarts using committed offsets, fsync and private metadata. An interrupted
promotion fails closed if its outcome is ambiguous. Re-upload after deletion or
file drift resets completed upload metadata. `reserved=True` preserves an
already-created empty 0600 inode for the worker credential protocol.

Download failures report `operation=download`, the HTTP status (or
`HTTP unavailable` if no response is available), `framing=json|range`, and the local
byte offset; Range validation failures also report the received byte count.
These diagnostics omit remote paths, response bodies, header values, file
identities and tokens. They describe the failed operation, not its root cause.
The original live Range failure is explained below (Range header not delivered
intact through the provider proxy); JSON envelopes are the default.

Upload failures additionally emit a warning from `runpod_lifecycle.https` with
the fixed stage `initiate`, `chunk`, `verify` or `status`, and the HTTP status
(or `unavailable`). These warnings contain no upload ID, target path, metadata,
response body/header or exception text. Retain sanitized stderr alongside the
ordinary smoke receipt: its v1 upload fields identify only the phase and status,
so an upload-phase 404 alone cannot distinguish the failing request or its cause.
If chunk reconciliation also fails, the client preserves the original chunk
exception and stops; it does not switch methods, routes or origins.

### Provider route settling (empty 404)

RunPod's proxy answers `404` with an **empty body** when it has no route to the
pod port. Live probes (2026-10-27) showed the route appearing 20-40 s after the
agent was already listening, then still dropping out intermittently for several
seconds after the first successful health reply (`200, 200, 404, 200, 404...` on
one connection, while the agent's own access log never saw the 404 requests).
The agent never sends an empty 404; its 404s are JSON (`unknown job`,
`unknown upload`, `unknown route`). Therefore:

- `wait_ready` returns only after 3 consecutive authenticated health replies;
  on timeout it logs `pod HTTPS readiness timed out (last=route_unavailable|HTTP n|request_failed|incompatible)`.
  `Pod.wait_ready` (HTTPS) uses the same settled check, bounded to 15 s per
  provider-status iteration.
- Every other request retries an empty 404 in place for up to `route_grace`
  seconds (default 90), logging `pod HTTPS route unavailable (HTTP 404 without
  agent body); retrying` once per affected request. It never retries an agent
  JSON 404 and never changes method, route or origin. After the grace the v1
  error `pod HTTPS request rejected (HTTP 404)` is raised as before.
- `runpod-lifecycle transport-smoke` writes these fixed-label warnings (and the
  upload stage warnings) to stderr; provider SDK calls remain muted.
- An optional absolute `deadline` (local `time.monotonic()`) bounds every
  request wait and every retry/route-grace sleep; past it the transport raises
  `pod HTTPS caller deadline exceeded`.
- The engine relay preserves the signature: an empty-bodied engine 404 is
  replied as the agent's JSON 404, so only the provider can produce an empty one.
- The exact pod origin keeps the normal controller egress (environment proxy and
  CA settings). A loopback `base_url` (route-local gateway, tests) ignores proxy
  environment variables.

Range requests are not used by default: on the live proxy a `Range` request
header did not reach the agent intact (agent answered 400 "bounded byte Range
required"), which explains the first live smoke's Range download failure.

No command or full-file hashing blocks a request. HTTP calls time out within
30 seconds each; command timeouts cancel the process group. Upload verification
runs in a background thread with a one-hour client polling bound. Job logs are
disk-backed and returned in at most 256 KiB per stream per poll. This small agent
retains at most 128 job records per claim; rotate/release long-lived claims before
that bound. It has no multi-user isolation, file quota, or log quota.

## Engine startup selection (trusted, optional)

The original claim can fix the one engine slot and the agent's private state
directory at pod creation. Nothing is configurable over HTTP and the default
(no selection) is unchanged. Provide a local file (owned, not group/other
writable, at most 16 KiB, strict JSON without duplicate keys) with
`launch --engine-startup /abs/engine-startup.json`, or export
`RUNPOD_ENGINE_STARTUP_FILE=/abs/engine-startup.json` so an unmodified claim
script (which builds `RunPodConfig.from_env`) picks it up. It requires
`RUNPOD_TRANSPORT=https` and contains no bearer, provider key or controller key:

```json
{"schema": "astrid.engine-startup.v1",
 "engine_proxy": {"endpoint_id": "vibecomfy-h3", "port": 8188,
                  "input_files": ["/workspace/ComfyUI/input/<approved-basename>"],
                  "output_roots": ["/workspace/ComfyUI/output"]},
 "state_dir": "/workspace/astrid/agent-state"}
```

* Only `vibecomfy-h3` on loopback port 8188 is accepted; the public port list
  stays exactly `8000/http`, and no template override is allowed.
* `input_files` are exact absolute destinations (no globs or directories) and
  `output_roots` the selected output root(s), both non-empty and fixed here; the
  agent cannot expand them later. The caller derives the input list from the
  basenames of the approved asset archive/manifest it will install (one
  destination per approved asset basename under the engine input directory),
  never from request data.
* `state_dir` must be strictly beneath the volume mount path, use only
  `[A-Za-z0-9._/-]`, and must not contain or lie inside any input or output
  path. The agent creates it 0700 and writes the original claim bearer to
  `<state_dir>/engine-bearer.token` (0600) before serving; the later readiness
  producer writes `<state_dir>/engine-readiness.json` into the same directory.
  Health is served before any engine exists and asserts no engine readiness.
* The engine selection travels as base64 JSON (`ASTRID_POD_ENGINE_B64`) with a
  plain-text `ASTRID_POD_STATE`, because the provider SDK interpolates env values
  into GraphQL strings unescaped.

Intended handoff: normal-mode claim (this selection) -> existing remote install
CLI over the same bearer with `root`/`state`/`input`/`output` paths from the
selection -> target readiness producer writing `engine-readiness.json` into
`state_dir` -> finalizer reading `engine-bearer.token` and the profile from
`state_dir`. Those stages own their own behavior; this package only fixes the
startup slot and private state.

## Security and rollout status

Knowing a pod URL alone permits no command, file or health access. A stolen
claim token in default general mode grants arbitrary commands and file access as
the pod user (normally root), equivalent to the SSH credential it replaces.
Observation mode restricts that token to the configured read-only operations.
RunPod terminates public TLS and forwards HTTP to the pod; RunPod, trusted egress
proxies and account administrators who can inspect pod env must be trusted.
Do not expose the origin port through an additional raw TCP mapping. The agent
removes token/bootstrap variables from spawned command environments, but this
does not hide creation env from the provider or privileged processes in the pod.

Live test: command execution and a 50 MiB upload succeeded on a stock base image.
The HTTP Range download failed before receiving bytes; the response detail was
not retained. JSON-envelope downloads were added afterwards and pass localhost
tests, but have **not** been live-verified. The one permitted test pod was released
and exact GET 404 verified. See the delivery report for dates and test evidence.

The Astrid H3 release builder and prepared-worker adapters select this transport.
Legacy generic `runner.py`/shipping raw-SFTP paths still need transport migration;
their SSH-only low-level APIs should not be used with an HTTPS config.
