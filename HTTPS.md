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

## Wire protocol and limits

Every endpoint requires `Authorization: Bearer <claim-token>` and uses
constant-time authentication. Health returns protocol version 1.

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
The original live Range failure predates these diagnostics and remains
unexplained; local JSON success does not resolve that live qualification gap.

No command or full-file hashing blocks a request. HTTP calls time out within
30 seconds each; command timeouts cancel the process group. Upload verification
runs in a background thread with a one-hour client polling bound. Job logs are
disk-backed and returned in at most 256 KiB per stream per poll. This small agent
retains at most 128 job records per claim; rotate/release long-lived claims before
that bound. It has no multi-user isolation, file quota, or log quota.

## Security and rollout status

Knowing a pod URL alone permits no command, file or health access. A stolen
claim token grants arbitrary commands and file access as the pod user (normally
root), equivalent to the SSH credential it replaces. It is not a read-only API.
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
