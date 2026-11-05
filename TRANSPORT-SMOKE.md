# Qualify the HTTPS pod transport

`runpod-lifecycle transport-smoke` rents **one CPU pod**, checks the authenticated
agent, executes `uname -a`, uploads and downloads a generated 50 MiB file, verifies
its full SHA256, and runs a 102-second asynchronous command when time remains.
Every agent HTTP request has a 10-second timeout; job polling crosses the proxy's
roughly 100-second request window without holding a request open. Downloads use
the transport's default JSON envelopes, not HTTP Range framing.

This is a paid provider operation. Run it only within your allocation authority.
It uses lifecycle's standard `api.create_pod` CPU REST path with two vCPUs,
`cpu3c`/`cpu5c` in custom priority, a 5 GiB container disk, **no volume**, and the
embedded HTTPS agent on `8000/http`. There is no SSH, GPU fallback, provider create
retry, or alternative allocation implementation. It uses the fixed Python/bash
image `runpod/base:1.0.3-dev-fix-pytorch-version-verification-cuda1300-ubuntu2404`.
The two-vCPU request is the minimum supported by this lifecycle CPU path; the
command does not introduce another catalogue or placement implementation.

## Run

Install lifecycle with the B0 HTTPS transport changes first. Use Linux/POSIX and
a foreground main-thread process with an unused `SIGALRM` timer. Put handles on
persistent local storage owned by your user. The parent directory must exist,
contain no symlink components, and have no group/other write permissions.

Point `ASTRID_ENV_FILE` at your existing credential file using the established
Astrid credential resolver. The command uses the CLI's existing loading path;
there is no command-specific credential parser. Neither credential values nor an
agent bearer belong in arguments, shell history, logs, or documentation.

```sh
# ASTRID_ENV_FILE is already set to the existing credential file's absolute path.
mkdir -m 700 -p "$HOME/transport-qualification"
runpod-lifecycle transport-smoke \
  --handle "$HOME/transport-qualification/https-roundtrip-001.json" \
  --max-seconds 600 \
  --max-hourly 0.10 \
  --payload-bytes 52428800
```

`--handle` is required. `--max-hourly` is required for a new allocation and must
be finite and positive. The actual provider `costPerHr` is checked immediately
after recording the pod ID; missing, invalid, or excessive pricing causes release
before agent use. This is an observed-price guard, not a provider offer-price
constraint: a brief allocation can incur charges before its price is available.
The cap does not include provider storage charges, taxes, or billing adjustments.

`--max-seconds` defaults to **600**, accepts integers from 1 through **900**, and
includes a cleanup reserve of one third of the budget, capped at 90 seconds.
The work deadline interrupts blocking transfers and provider retries as well as
polling. Cleanup has its own bounded alarm. Short budgets or slow readiness may
leave the long command untested. The payload defaults to **52,428,800 bytes**
(50 MiB); `--payload-bytes` accepts 1 byte through 1 GiB. Smaller payloads are useful
for diagnosis but do not satisfy the 50 MiB gate. The command always selects HTTPS,
regardless of launch-oriented environment defaults; no agent token input is needed.

The JSON printed on stdout is also the durable handle. A full gate requires all
of these, not just exit zero:

- `status: "passed"`, `qualified: true`, `protocol: 1`, and `uname: "passed"`.
- `transfer_receipt.bytes` at least 52,428,800, `download_mode: "json-envelope"`,
  `full_sha256_verified: true`, and the full-file `sha256`.
- `long_job.status: "passed"` and `long_job.seconds` greater than 100.
- `cleanup: "verified_absent"` and `cleanup_get_status: 404`.

If the remaining work budget cannot accommodate 102 seconds plus 15 seconds of
margin after transfer, `long_job` records `skipped` with a reason, and `qualified`
remains false. `status` describes transport execution; `cleanup` independently
records custody. `qualified` requires both full qualification and verified release.
Remote stdout/stderr and raw provider response bodies are never printed or saved.
Failures retain a phase and fixed error category, without exception text or secrets.

Exit codes: **0** means the requested round trip passed and release was verified
(or cleanup-only verified release/no allocation). Check `qualified` for the full
gate. **1** means qualification failed with cleanup settled; **2** means a local
invocation/custody error; **3** means cleanup is still pending. Interrupts during
work record `failure: "interrupted"` and still attempt cleanup. An interrupt before
a handle is established can return **130**.

## Recover without allocating again

The command refuses **any** existing handle for a new run, including completed,
failed, or unresolved handles. Never delete or rename a pending handle to make a
new allocation possible. Use exactly the original handle for recovery:

```sh
runpod-lifecycle transport-smoke \
  --handle "$HOME/transport-qualification/https-roundtrip-001.json" \
  --cleanup-only --max-seconds 120
```

Cleanup-only requires provider credentials, but does not read or require an agent
token, SSH key, price cap, or preflight connection to the pod. It does not run jobs
or transfers. It verifies the exact recorded ID and matching unique request name
before termination, and independently reads that exact ID until GET returns 404.
An already completed handle is rechecked. A successful deletion acknowledgment or
an empty account inventory alone never counts as verified release.

Before create, the handle atomically records and fsyncs a unique request name and
single allocation intent. After create, the exact pod ID is atomically saved and
fsynced before any pricing or transport operation. If create loses its response,
cleanup reconciles **only an exact match** on the unique request name. One match is
recorded and cleaned; zero matches remain pending because a delayed create can
still appear; multiple matches remain pending rather than guessing ownership.
Repeat cleanup-only after visibility recovers. It never allocates again. Repeated
failures or multiple exact-name matches require provider-account investigation;
do not edit a handle to guess a pod ID. The account must be clear of unresolved
allocation intents before any new handle is used.

The command refuses allocation if the account already contains **any pod**, even
a stopped pod. A user-wide local process lock serializes this command across
handles; a separate handle lock protects recovery. Coordinate with other hosts and
allocation tools: provider inventory plus a local lock is not a distributed account
transaction, and cannot prevent an unrelated concurrent allocator from racing the
pre-create check. This command never deletes another pod found in the inventory.

## Retain evidence

The owned mode-0600 handle contains only a token-file reference. The randomly
generated agent bearer lives exclusively in the owned mode-0600
`HANDLE.files/agent.token` sidecar inside an exclusively created mode-0700 directory.
The provider receives the bearer through lifecycle's standard HTTPS bootstrap env.
Do not print or copy that file into logs. Keep its containing directory private.

`HANDLE.files/source.bin`, `download.bin`, and a transport-generated
`download.bin.https-part` may remain after failure. Failed evidence is preserved,
even when cleanup subsequently succeeds. On a successful round trip with durable
GET-404 evidence, the original invocation removes its generated local source,
download and token. Cleanup-only preserves local evidence and never follows the
handle's token-file reference. Once cleanup is verified, archive the handle and
failed payloads as appropriate; the released pod's token has no remaining endpoint.
The `.lock` files and empty evidence directory can remain.

The handle records UTC intent/allocation/readiness/transfer/release timestamps,
observed hourly price, and a rounded-up lifetime from **before** the create call to
verified absence, including create and cleanup overhead. With a known observed
rate it records an estimated cost upper bound, not an invoice. Across process loss,
UTC elapsed time accounts for downtime; accurate nondecreasing host time is required
for that bound. Pending cleanup has no final lifetime/cost bound and may continue
to incur charges beyond the command's runtime limit.

SIGINT, SIGTERM and the work timeout enter cleanup; repeated interrupts are ignored
while cleanup's alarm remains active. SIGKILL, host loss, filesystem failure, or a
provider outage can prevent release. The durable intent/identity enables recovery
but cannot itself guarantee provider availability. Storage must honor fsync/atomic
rename, remain writable, and persist across restarts. An empty/torn handle from a
failure before the initial durable intent cannot have allocated a pod through this
command. Do not modify handles or sidecars while a command is running.

After cleanup is settled, use an explicitly **new handle path** for another
separately authorized qualification attempt. This command supplies repeatable
product evidence; local tests alone do not establish a live RunPod proxy pass.
