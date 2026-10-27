"""Real loopback observation agent and compressed startup; no provider network."""
from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import threading
import time

import httpx
import pytest

from runpod_lifecycle import pod_agent
from runpod_lifecycle.https import HttpsPodTransport, PodTransportError, agent_bootstrap
from runpod_lifecycle.pod_agent import (
    AgentServer, CHUNK, OBSERVATION_ENV, OBSERVATION_LIMIT, OBSERVATION_SCHEMA,
    ObservationPolicy,
)

TOKEN = "test-only-observation-token-" + "x" * 32


def policy(*, roots=(), files=()):
    return {"schema": OBSERVATION_SCHEMA, "roots": list(roots), "files": list(files)}


def encode(data):
    return base64.b64encode(json.dumps(data).encode()).decode()


@pytest.fixture(autouse=True)
def local_only(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    original = httpx.Client.request

    def request(client, method, url, **kwargs):
        assert client.base_url.join(url).host == "127.0.0.1", "external HTTP forbidden"
        return original(client, method, url, **kwargs)

    monkeypatch.setattr(httpx.Client, "request", request)


@pytest.fixture
def running_agent(tmp_path):
    @contextmanager
    def run(config):
        state = tmp_path / "state"
        state.mkdir(exist_ok=True)
        # Observation must not read/recover a general-agent upload journal.
        journal = state / "old.json"
        journal.write_text('{"verifying":true}')
        before = journal.read_bytes()
        server = AgentServer(("127.0.0.1", 0), TOKEN, state, observation_policy=config)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = HttpsPodTransport("local", TOKEN,
            base_url=f"http://127.0.0.1:{server.server_port}", request_timeout=1, poll_interval=.01)
        try:
            yield server, client
            assert not server.jobs and journal.read_bytes() == before
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            thread.join(2)
    return run


def download(client, path, ranges=False, **extra):
    if ranges:
        return client.http.get("/files", params={"path": path, **extra},
                               headers={"Range": "bytes=0-1048575"})
    return client.http.post("/download-chunk", json={"path": path, "offset": 0, **extra})


@pytest.mark.parametrize("bad", [
    {}, [], "policy", False,
    {"schema": OBSERVATION_SCHEMA, "roots": []},
    {"schema": "other", "roots": ["/public"], "files": []},
    {"schema": OBSERVATION_SCHEMA, "roots": ["/public"], "files": [], "token": TOKEN},
    policy(roots=["/public", "/public"]), policy(files=["/receipt", "/receipt"]),
    policy(roots=["/same"], files=["/same"]), policy(),
    policy(roots=[f"/root{i}" for i in range(9)]),
    policy(files=[f"/file{i}" for i in range(33)]),
    {"schema": OBSERVATION_SCHEMA, "roots": "/public", "files": []},
    {"schema": OBSERVATION_SCHEMA, "roots": [], "files": ("/receipt",)},
    {"schema": OBSERVATION_SCHEMA, "roots": None, "files": ["/receipt"]},
], ids=["missing", "array", "string", "bool", "missing-files", "schema", "unknown-secret-field",
        "duplicate-roots", "duplicate-files", "duplicate-cross-fields", "empty", "root-count",
        "file-count", "roots-type", "files-type", "null-roots"])
def test_invalid_policy_rejected_before_boot_material_or_listener(bad, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid policy reached source read or listener")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(pod_agent.ThreadingHTTPServer, "__init__", forbidden)
    for create in (lambda: agent_bootstrap(TOKEN, observation_policy=bad),
                   lambda: AgentServer(("127.0.0.1", 0), TOKEN, observation_policy=bad)):
        with pytest.raises(ValueError, match="^invalid observation policy$") as caught:
            create()
        assert TOKEN not in str(caught.value)


@pytest.mark.parametrize("field", ["roots", "files"])
@pytest.mark.parametrize("path", ["/", "", "relative", "./receipt", "/a/./b", "/a/../b",
    "/a//b", "//a", "/a/", "/a\\b", "/a\x00b", 1, True, None, ["/a"]])
def test_noncanonical_config_paths_rejected(field, path):
    data = policy()
    data[field] = [path]
    with pytest.raises(ValueError, match="^invalid observation policy$"):
        agent_bootstrap(TOKEN, observation_policy=data)


def test_maximum_allowlist_and_encoded_size_boundaries():
    data = policy(roots=[f"/public/{i}" for i in range(8)], files=[f"/receipt/{i}" for i in range(32)])
    assert len(ObservationPolicy(data).roots) == 8
    overhead = len(base64.b64decode(ObservationPolicy(policy(files=["/x"])).encoded)) - 1
    # A canonical JSON of exactly 12288 bytes fills the 16384-byte base64 cap.
    data = policy(files=["/" + "x" * (OBSERVATION_LIMIT // 4 * 3 - overhead)])
    encoded = ObservationPolicy(data).encoded
    assert len(encoded) == OBSERVATION_LIMIT
    assert ObservationPolicy.from_environment(encoded).files == tuple(data["files"])
    data["files"][0] += "x"
    with pytest.raises(ValueError, match="^invalid observation policy$"):
        agent_bootstrap(TOKEN, observation_policy=data)
    with pytest.raises(ValueError, match="^invalid observation policy$"):
        ObservationPolicy.from_environment(encoded + "AAAA")


@pytest.mark.parametrize("value", ["", "not-base64!", "é", "=", "A" * (OBSERVATION_LIMIT + 1),
    base64.b64encode(b"\xff").decode(), base64.b64encode(b'{"schema":').decode(),
    encode(None), encode(policy()),
    base64.b64encode(b'{"schema":"astrid.pod-observation.v1","roots":[],"files":["/r"],"files":["/s"]}').decode(),
    base64.b64encode(b"[" * 2000).decode(),
], ids=["empty", "alphabet", "non-ascii", "padding", "oversize", "utf8", "truncated-json",
        "null", "empty-policy", "duplicate-json-key", "deep-json"])
def test_malformed_environment_fails_before_serving_and_is_consumed(value, monkeypatch):
    monkeypatch.setenv("ASTRID_POD_TOKEN", TOKEN)
    monkeypatch.setenv("ASTRID_POD_AGENT_B64", "unused")
    monkeypatch.setenv(OBSERVATION_ENV, value)
    def forbidden(*_a, **_kw):
        pytest.fail("malformed config reached listener")
    monkeypatch.setattr(pod_agent, "AgentServer", forbidden)
    with pytest.raises(ValueError, match="^invalid observation policy$"):
        pod_agent.main()
    assert all(key not in os.environ for key in ("ASTRID_POD_TOKEN", "ASTRID_POD_AGENT_B64", OBSERVATION_ENV))


def test_main_consumes_policy_and_token_before_server_creation(monkeypatch):
    data = policy(roots=["/public"], files=["/release/.astrid-engine/install.json"])
    monkeypatch.setenv("ASTRID_POD_TOKEN", TOKEN)
    monkeypatch.setenv("ASTRID_POD_AGENT_B64", "unused")
    monkeypatch.setenv(OBSERVATION_ENV, encode(data))
    calls = []
    class Server:
        def __init__(self, _address, token, _state, *, observation_policy):
            assert token == TOKEN and observation_policy == data
            assert all(key not in os.environ for key in ("ASTRID_POD_TOKEN", "ASTRID_POD_AGENT_B64", OBSERVATION_ENV))
        def serve_forever(self):
            calls.append("serve")
        def server_close(self):
            calls.append("close")
    monkeypatch.setattr(pod_agent, "AgentServer", Server)
    pod_agent.main()
    assert calls == ["serve", "close"]


@pytest.mark.parametrize("ranges", [False, True])
def test_receipts_closed_logs_and_empty_files_without_exec_or_put(running_agent, tmp_path, monkeypatch, ranges):
    root = tmp_path / "observations"
    (root / "nested").mkdir(parents=True)
    receipt = tmp_path / "release" / ".astrid-engine" / "install.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b'{"schema":"astrid.engine-install.v1","status":"installed","qualified":false}')
    log = root / "nested" / "stdout.000001.log"
    log.write_bytes((b"closed sanitized log\n" * 60000)[:CHUNK])
    sidecar = log.with_suffix(".log.sha256")
    sidecar.write_text(hashlib.sha256(log.read_bytes()).hexdigest() + "\n")
    empty = root / "stderr.000001.log"
    empty.touch()
    snapshot = root / "object_info.json"
    snapshot.write_bytes(b"x" * (CHUNK + 123))
    def forbidden(*_a, **_kw):
        pytest.fail("observation attempted execution or upload")
    for name in ("start", "execute_command", "put", "poll", "cancel"):
        monkeypatch.setattr(HttpsPodTransport, name, forbidden)
    monkeypatch.setattr(pod_agent.subprocess, "Popen", forbidden)
    config = policy(roots=[str(root)], files=[str(receipt)])
    with running_agent(config) as (_, client):
        client.connect()
        assert client.http.get("/health").json() == {"ok": True, "protocol": 1}
        for index, source in enumerate((receipt, log, sidecar, empty, snapshot)):
            destination = tmp_path / f"download-{index}"
            client.get(str(source), destination, use_ranges=ranges)
            assert destination.read_bytes() == source.read_bytes()
            assert hashlib.sha256(destination.read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest()
        response = download(client, str(log), ranges)
        if ranges:
            assert response.status_code == 206
            assert response.headers["Content-Range"] == f"bytes 0-{CHUNK - 1}/{log.stat().st_size}"
            assert response.headers["ETag"].startswith('"')
        else:
            data = response.json()
            assert set(data) == {"offset", "total", "etag", "sha256", "data"}
            chunk = base64.b64decode(data["data"], validate=True)
            assert chunk == log.read_bytes()[:CHUNK] and len(chunk) == CHUNK
            assert data["sha256"] == hashlib.sha256(chunk).hexdigest()


@pytest.mark.parametrize("method,path", [
    ("GET", "/health"), ("POST", "/health"), ("POST", "/jobs"), ("GET", "/jobs/id"),
    ("DELETE", "/jobs/id"), ("POST", "/uploads"), ("GET", "/uploads/id"),
    ("PUT", "/uploads/id"), ("POST", "/uploads/id"), ("DELETE", "/uploads/id"),
    ("GET", "/files"), ("POST", "/download-chunk"), ("GET", "/unknown"),
    ("PATCH", "/health"), ("OPTIONS", "/health"), ("HEAD", "/health"),
    ("TRACE", "/health"), ("FROB", "/unknown"),
])
def test_auth_first_and_all_control_routes_refused(running_agent, method, path, tmp_path, monkeypatch, capsys):
    def forbidden(*_a, **_kw):
        pytest.fail("route denial parsed a body or touched the filesystem")
    with running_agent(policy(files=[str(tmp_path / "receipt")])) as (server, client):
        monkeypatch.setattr(pod_agent.Handler, "json_body", forbidden)
        monkeypatch.setattr(pod_agent.Handler, "open_download", forbidden)
        monkeypatch.setattr(server, "finish_upload", forbidden)
        monkeypatch.setattr(server, "cancel", forbidden)
        # Deliberately invalid body must never be parsed before auth/policy.
        for authorization in ("Bearer wrong", ""):
            bad = client.http.request(method, path, content=b"not-json",
                                      headers={"Authorization": authorization})
            assert bad.status_code == 401
            if method != "HEAD":
                assert bad.json() == {"error": "unauthorized"}
        if (method, path) not in {("GET", "/health"), ("GET", "/files"), ("POST", "/download-chunk")}:
            response = client.http.request(method, path, content=b"not-json")
            assert response.status_code == 403
            if method != "HEAD":
                assert response.json() == {"error": "forbidden"}
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("ranges", [False, True])
def test_paths_cannot_expand_startup_authority(running_agent, tmp_path, ranges, caplog, capsys):
    root = tmp_path / "public"
    root.mkdir()
    receipt = tmp_path / "receipt"
    receipt.write_text("public receipt")
    private = tmp_path / "private" / TOKEN
    private.parent.mkdir()
    private.write_text(TOKEN)
    sibling = tmp_path / "public-other"
    sibling.write_text(TOKEN)
    nested = root / "nested"
    nested.mkdir()
    config = policy(roots=[str(root)], files=[str(receipt)])
    with running_agent(config) as (_, client):
        # The original mapping is no longer authority after construction.
        config["files"].append(str(private))
        config["roots"].append(str(private.parent))
        for path in (str(private), str(sibling), str(receipt) + "/child", str(root) + "/../private/" + TOKEN,
                     str(root) + "/nested/../../private/" + TOKEN, str(root) + "/./log", str(root) + "//log",
                     str(root) + "/log\\escape", str(root) + "/log\x00secret", "relative", "/"):
            response = download(client, path, ranges, observation_policy=policy(roots=[str(tmp_path)]),
                                roots=[str(tmp_path)], files=[str(private)])
            assert response.status_code == 403
            assert response.json() == {"error": "forbidden"}
            assert path not in response.text and TOKEN not in response.text
        with pytest.raises(PodTransportError) as caught:
            client.get(str(private), tmp_path / "output", use_ranges=ranges)
        assert TOKEN not in str(caught.value) and str(private) not in str(caught.value)
        assert not (tmp_path / "output").exists()
        assert download(client, str(receipt), ranges).status_code == (206 if ranges else 200)
    output = capsys.readouterr()
    assert not output.out and not output.err
    assert TOKEN not in caplog.text and str(private) not in caplog.text


@pytest.mark.parametrize("ranges", [False, True])
@pytest.mark.parametrize("hostile", ["final-symlink", "parent-symlink", "root-symlink",
    "ancestor-symlink", "hardlink", "fifo", "directory", "socket", "missing"])
def test_hostile_files_fail_closed(running_agent, tmp_path, ranges, hostile):
    outside = tmp_path / "private"
    outside.mkdir()
    secret = outside / "file"
    secret.write_bytes(TOKEN.encode())
    root = tmp_path / "public"
    root.mkdir()
    target = root / "file"
    held_socket = None
    if hostile == "final-symlink":
        target.symlink_to(secret)
    elif hostile == "parent-symlink":
        (root / "parent").symlink_to(outside, target_is_directory=True)
        target = root / "parent" / "file"
    elif hostile == "root-symlink":
        root.rmdir()
        root.symlink_to(outside, target_is_directory=True)
    elif hostile == "ancestor-symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        root = alias / "public"
        target = root / "file"
        (tmp_path / "public" / "file").write_bytes(TOKEN.encode())
    elif hostile == "hardlink":
        os.link(secret, target)
    elif hostile == "fifo":
        os.mkfifo(target)
    elif hostile == "directory":
        target.mkdir()
    elif hostile == "socket":
        held_socket = socket.socket(socket.AF_UNIX)
        # Avoid the UNIX socket pathname length limit in long --basetemp paths.
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            held_socket.bind(f"/proc/self/fd/{directory}/file")
        finally:
            os.close(directory)
    try:
        with running_agent(policy(roots=[str(root)])) as (_, client):
            started = time.monotonic()
            response = download(client, str(target), ranges)
            assert response.status_code in (400, 403)
            assert TOKEN not in response.text and str(target) not in response.text
            assert time.monotonic() - started < 2  # FIFO cannot wait for a writer.
    finally:
        if held_socket:
            held_socket.close()


@pytest.mark.parametrize("ranges", [False, True])
def test_missing_receipt_becomes_readable_only_after_safe_publication(running_agent, tmp_path, ranges):
    receipt = tmp_path / "not-yet-installed" / ".astrid-engine" / "install.json"
    with running_agent(policy(files=[str(receipt)])) as (_, client):
        assert download(client, str(receipt), ranges).status_code == 400
        receipt.parent.mkdir(parents=True)
        temporary = receipt.with_suffix(".new")
        temporary.write_bytes(b"canonical receipt bytes")
        temporary.replace(receipt)
        destination = tmp_path / "download"
        client.get(str(receipt), destination, use_ranges=ranges)
        assert destination.read_bytes() == b"canonical receipt bytes"


@pytest.mark.parametrize("ranges", [False, True])
@pytest.mark.parametrize("timing", ["before-read", "during-read"])
def test_hardlink_created_after_open_is_refused(running_agent, tmp_path, monkeypatch, ranges, timing):
    receipt = tmp_path / "receipt"
    receipt.write_bytes(b"public")
    alias = tmp_path / "private-alias"
    original = ObservationPolicy.open
    class LinkingStream:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            self.stream.close()
        def __getattr__(self, name):
            return getattr(self.stream, name)
        def read(self, size):
            data = self.stream.read(size)
            os.link(receipt, alias)
            return data
    def linked_open(self, path):
        stream = original(self, path)
        if timing == "before-read":
            os.link(receipt, alias)
            return stream
        return LinkingStream(stream)
    with running_agent(policy(files=[str(receipt)])) as (_, client):
        monkeypatch.setattr(ObservationPolicy, "open", linked_open)
        assert download(client, str(receipt), ranges).status_code == 403
        assert alias.exists()


@pytest.mark.parametrize("ranges", [False, True])
@pytest.mark.parametrize("swap", ["parent-before-open", "final-before-open", "parent-after-open"])
def test_component_swaps_at_actual_open_never_follow_symlinks(running_agent, tmp_path, monkeypatch, ranges, swap):
    root = tmp_path / "public"
    parent = root / "nested"
    parent.mkdir(parents=True)
    target = parent / "file"
    target.write_bytes(b"public")
    outside = tmp_path / "private"
    outside.mkdir()
    secret = outside / "file"
    secret.write_bytes(TOKEN.encode())
    original = os.open
    swapped = False
    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped and dir_fd is not None and path == ("file" if swap == "final-before-open" else "nested"):
            swapped = True
            if swap == "final-before-open":
                target.unlink()
                target.symlink_to(secret)
            elif swap == "parent-before-open":
                parent.rename(root / "saved")
                parent.symlink_to(outside, target_is_directory=True)
            else:
                fd = original(path, flags, mode, dir_fd=dir_fd)
                parent.rename(root / "saved")
                parent.symlink_to(outside, target_is_directory=True)
                return fd
        return original(path, flags, mode, dir_fd=dir_fd)
    with running_agent(policy(roots=[str(root)])) as (_, client):
        monkeypatch.setattr(pod_agent.os, "open", racing_open)
        response = download(client, str(target), ranges)
        assert swapped and TOKEN not in response.text
        if swap == "parent-after-open":
            assert response.status_code == (206 if ranges else 200)
            body = response.content if ranges else base64.b64decode(response.json()["data"])
            assert body == b"public"
        else:
            assert response.status_code == 400


@pytest.mark.parametrize("ranges", [False, True])
def test_representation_replacement_and_mid_read_change_rejected(running_agent, tmp_path, monkeypatch, ranges):
    receipt = tmp_path / "receipt"
    receipt.write_bytes(b"original receipt")
    with running_agent(policy(files=[str(receipt)])) as (_, client):
        first = download(client, str(receipt), ranges)
        etag = first.headers["ETag"] if ranges else first.json()["etag"]
        replacement = tmp_path / "new"
        replacement.write_bytes(b"new receipt")
        replacement.replace(receipt)
        if ranges:
            response = client.http.get("/files", params={"path": str(receipt)},
                                       headers={"Range": "bytes=0-10", "If-Match": etag})
        else:
            response = download(client, str(receipt), etag=etag)
        assert response.status_code == 412
        original = ObservationPolicy.open
        class ChangingStream:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                self.stream.close()
            def __getattr__(self, name):
                return getattr(self.stream, name)
            def read(self, size):
                data = self.stream.read(size)
                with receipt.open("ab") as writer:
                    writer.write(b"changed")
                return data
        monkeypatch.setattr(ObservationPolicy, "open", lambda self, path: ChangingStream(original(self, path)))
        assert download(client, str(receipt), ranges).status_code == 412


@pytest.mark.parametrize("startup", ["sdk", "rest"])
def test_actual_compressed_bootstrap_observation_config_arrives_intact(tmp_path, monkeypatch, startup):
    import zlib
    root = tmp_path / "public spaces 'quote' $value;\nnew-line é"
    root.mkdir()
    receipt = root / "install.json"
    receipt.write_bytes(b"startup public receipt")
    config = policy(roots=[str(root)], files=[str(tmp_path / "z-future"), str(tmp_path / "a-future")])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    boot = agent_bootstrap(TOKEN, port, observation_policy=config)
    assert json.loads(base64.b64decode(boot["env"][OBSERVATION_ENV])) == config
    assert len(boot["env"][OBSERVATION_ENV]) <= OBSERVATION_LIMIT
    assert zlib.decompress(base64.b64decode(boot["env"]["ASTRID_POD_AGENT_B64"])) == Path(pod_agent.__file__).read_bytes()
    assert TOKEN not in boot["docker_args"] and TOKEN not in boot["docker_start_cmd"][0]
    environment = dict(os.environ, **boot["env"], ASTRID_POD_STATE=str(tmp_path / "unused-state"))
    command = (shlex.split(boot["docker_args"]) if startup == "sdk"
               else boot["docker_entrypoint"] + boot["docker_start_cmd"])
    command[-1] = command[-1].replace("python3 -u", shlex.quote(sys.executable) + " -S -u")
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    client = HttpsPodTransport("local", TOKEN, base_url=f"http://127.0.0.1:{port}",
                               request_timeout=.2, poll_interval=.02)
    def forbidden(*_a, **_kw):
        pytest.fail("bootstrap observation needed exec/upload")
    for name in ("start", "execute_command", "put", "poll", "cancel"):
        monkeypatch.setattr(HttpsPodTransport, name, forbidden)
    try:
        client.wait_ready(timeout=5)
        client.get(str(receipt), tmp_path / "download")
        assert (tmp_path / "download").read_bytes() == receipt.read_bytes()
        assert client.http.post("/jobs", json={"command": "false"}).status_code == 403
        assert not (tmp_path / "unused-state").exists()
    finally:
        client.close()
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
    assert stdout == stderr == b""


def test_none_preserves_normal_bootstrap_and_command_upload_download(tmp_path):
    assert OBSERVATION_ENV not in agent_bootstrap(TOKEN)["env"]
    assert OBSERVATION_ENV not in agent_bootstrap(TOKEN, observation_policy=None)["env"]
    server = AgentServer(("127.0.0.1", 0), TOKEN, tmp_path / "normal-state", observation_policy=None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HttpsPodTransport("local", TOKEN, base_url=f"http://127.0.0.1:{server.server_port}", poll_interval=.01)
    try:
        client.connect()
        assert client.execute_command("printf normal", timeout=5) == (0, "normal", "")
        source, remote, dest = (tmp_path / name for name in ("source", "remote", "destination"))
        source.write_bytes(b"normal upload")
        client.put(source, str(remote))
        client.get(str(remote), dest)
        assert dest.read_bytes() == source.read_bytes()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(2)
