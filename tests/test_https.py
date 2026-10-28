from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx
import pytest

from runpod_lifecycle import api
from runpod_lifecycle import pod_agent
from runpod_lifecycle.config import RunPodConfig
from runpod_lifecycle.https import HttpsPodTransport, PodTransportError, _detached_record, agent_bootstrap, read_token_file, write_token_file
from runpod_lifecycle.pod import Pod
from runpod_lifecycle.pod_agent import AgentServer, CHUNK
from runpod_lifecycle.preflight import preflight_transport

TOKEN = "test-only-claim-token-" + "x" * 32


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    server = AgentServer(("127.0.0.1", 0), TOKEN, tmp_path / "state")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HttpsPodTransport("local", TOKEN, base_url=f"http://127.0.0.1:{server.server_port}",
                               request_timeout=0.5, poll_interval=0.02)
    try:
        yield server, client
    finally:
        client.disconnect()
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_auth_every_route_and_no_redirects(agent):
    _, client = agent
    for method, path in [("GET", "/health"), ("POST", "/jobs"), ("GET", "/jobs/id"),
                         ("DELETE", "/jobs/id"), ("POST", "/uploads"), ("PUT", "/uploads/id"), ("GET", "/files")]:
        response = client.http.request(method, path, headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401
    client.connect()
    with pytest.raises(ValueError, match="HTTPS required"):
        HttpsPodTransport("abc", TOKEN, base_url="http://example.com")


def test_job_offsets_idempotency_and_binary_logs(agent):
    _, client = agent
    command = shlex.join([sys.executable, "-c", "import sys;sys.stdout.buffer.write(b'x'*700000+b'\\xe2\\x82\\xac');sys.stderr.write('err');sys.exit(7)"])
    job = client.start(command)
    assert client.start(command, job_id=job) == job
    with pytest.raises(PodTransportError, match="409"):
        client.start("false", job_id=job)
    out, err, offsets = b"", b"", (0, 0)
    while True:
        result = client.poll(job, *offsets)
        out += base64.b64decode(result["stdout"])
        err += base64.b64decode(result["stderr"])
        offsets = result["stdout_offset"], result["stderr_offset"]
        if result["status"] == "exited" and result["stdout_eof"] and result["stderr_eof"]:
            break
    assert out == b"x" * 700000 + "€".encode()
    assert err == b"err" and result["exit_code"] == 7
    assert client.execute_command("printf hello; printf error >&2; exit 3") == (3, "hello", "error")


def test_cancel_and_command_timeout(agent):
    server, client = agent
    code, _, stderr = client.execute_command("sleep 30", timeout=0.1)
    assert code == -1 and "timed out" in stderr
    for _ in range(100):
        if all(job["process"].poll() is not None for job in server.jobs.values()):
            break
        time.sleep(0.02)
    assert all(job["process"].poll() is not None for job in server.jobs.values())


def test_50mb_resume_after_interruption_and_lost_reply(agent, tmp_path, monkeypatch):
    _, client = agent
    source, remote, output = (tmp_path / name for name in ("source", "remote", "output"))
    with source.open("wb") as stream:
        for i in range(50):
            stream.write(bytes([i]) * CHUNK)
    real = client._request
    seen = []
    def interrupt(method, path, **kwargs):
        if method == "PUT":
            seen.append(kwargs["params"]["offset"])
            if len(seen) == 3:
                raise KeyboardInterrupt
        return real(method, path, **kwargs)
    monkeypatch.setattr(client, "_request", interrupt)
    with pytest.raises(KeyboardInterrupt):
        client.put(source, str(remote))
    assert not remote.exists()
    lost = False
    def lose_reply(method, path, **kwargs):
        nonlocal lost
        response = real(method, path, **kwargs)
        if method == "PUT":
            seen.append(kwargs["params"]["offset"])
            if not lost:
                lost = True
                raise PodTransportError("simulated lost committed reply")
        return response
    monkeypatch.setattr(client, "_request", lose_reply)
    client.put(source, str(remote))
    assert seen[:4] == [0, CHUNK, 2 * CHUNK, 2 * CHUNK]
    client.get(str(remote), output)
    assert hashlib.sha256(output.read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest()
    assert output.stat().st_size == 50 * CHUNK


def test_upload_sha_rejection_and_reserved_inode(agent, tmp_path):
    _, client = agent
    source, remote = tmp_path / "secret-local", tmp_path / "secret-remote"
    source.write_bytes(b"private data")
    remote.touch(mode=0o600)
    inode = remote.stat().st_ino
    client.put(source, str(remote), reserved=True)
    assert remote.read_bytes() == source.read_bytes()
    assert remote.stat().st_ino == inode and remote.stat().st_mode & 0o777 == 0o600
    source.write_bytes(b"other data")
    with pytest.raises(PodTransportError, match="verification"):
        client.put(source, str(remote), reserved=True)
    assert remote.read_bytes() == b"private data"
    bad = tmp_path / "bad"
    state = client._request("POST", "/uploads", json={"path": str(bad), "size": 3, "sha256": "0" * 64}).json()
    client._request("PUT", "/uploads/" + state["id"], params={"offset": 0}, content=b"bad")
    client._request("POST", "/uploads/" + state["id"])
    for _ in range(100):
        state = client._request("GET", "/uploads/" + state["id"]).json()
        if state["error"]:
            break
        time.sleep(.01)
    assert state["error"] and not bad.exists()


@pytest.mark.parametrize("use_ranges", [False, True])
def test_empty_files_and_range_changed_representation(agent, tmp_path, use_ranges):
    _, client = agent
    source, remote, dest = (tmp_path / name for name in ("empty", "remote", "dest"))
    source.touch()
    client.put(source, str(remote))
    client.get(str(remote), dest, use_ranges=use_ranges)
    assert dest.read_bytes() == b""
    response = client.http.get("/files", params={"path": str(remote)},
                               headers={"Range": "bytes=0-10", "If-Match": '"stale"'})
    assert response.status_code == 412


async def test_fake_provider_and_pod_facade(agent, tmp_path, monkeypatch, runpod_sdk_mock):
    server, client = agent
    monkeypatch.setattr("runpod_lifecycle.preflight.preflight_transport", lambda _: None)
    runpod_sdk_mock.create_pod.return_value = {"id": "fakepod"}
    result = api.create_pod("test", "gpu", "runpod/base:latest", transport="https", https_token=TOKEN)
    assert result["id"] == "fakepod"
    payload = runpod_sdk_mock.create_pod.call_args.kwargs
    assert payload["ports"] == "8000/http" and payload["env"]["ASTRID_POD_TOKEN"] == TOKEN
    assert "ASTRID_POD_AGENT_B64" in payload["env"] and "docker_args" in payload
    config = RunPodConfig(api_key="test", transport="https", https_token=TOKEN, template_id="")
    assert TOKEN not in repr(config)
    pod = Pod("fakepod", "test", config)
    def local_client():
        return HttpsPodTransport("local", TOKEN, base_url=f"http://127.0.0.1:{server.server_port}", poll_interval=.01)
    monkeypatch.setattr(pod, "https_transport", local_client)
    async def status():
        return {"desired_status": "RUNNING", "ports": []}
    monkeypatch.setattr(pod, "status", status)
    assert (await pod.wait_ready(timeout=2))["desired_status"] == "RUNNING"
    assert await pod.exec_ssh("printf facade") == (0, "facade", "")
    tree = tmp_path / "tree"
    (tree / "nested").mkdir(parents=True)
    (tree / "nested" / "hello").write_text("world")
    remote = tmp_path / "uploaded"
    await pod.upload_path(tree, str(remote))
    assert (remote / "nested" / "hello").read_text() == "world"
    dest = tmp_path / "dest"
    await pod.get_file(str(remote / "nested" / "hello"), dest)
    assert dest.read_text() == "world"


def test_preflight_proxy_fails_before_any_paid_create(monkeypatch, runpod_sdk_mock):
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:1")
    with pytest.raises(RuntimeError, match="SSH egress unavailable; use --transport https"):
        api.create_pod("test", "gpu", "image")
    runpod_sdk_mock.create_pod.assert_not_called()


def test_preflight_https_checks_404_and_rejects_denial(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(404))
    preflight_transport("https")
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(403))
    with pytest.raises(RuntimeError, match="HTTP 403"):
        preflight_transport("https")


def test_token_file_permissions_and_no_symlinks(tmp_path):
    path = tmp_path / "token"
    write_token_file(path, TOKEN)
    assert read_token_file(path) == TOKEN
    path.chmod(0o644)
    with pytest.raises(ValueError):
        read_token_file(path)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        read_token_file(link)


def test_inline_bootstrap_runs_with_system_python_no_site_packages(tmp_path):
    boot = agent_bootstrap(TOKEN)
    import zlib
    source = zlib.decompress(base64.b64decode(boot["env"]["ASTRID_POD_AGENT_B64"]))
    assert source.startswith(b'"""Single-file')
    assert source == Path(pod_agent.__file__).read_bytes()
    # -S prevents third-party imports. Compile/import without starting main.
    result = subprocess.run([sys.executable, "-S", "-c", "import sys;ns={'__name__':'test'};exec(sys.stdin.read(),ns)"],
                            input=source.decode(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert TOKEN not in boot["docker_args"]


def test_job_polling_across_real_proxy_timeout_window(agent):
    _, client = agent
    start = time.monotonic()
    # Each HTTP request is limited to .5s. Job runs past RunPod's ~100s
    # per-request window while incremental stdout/stderr remain available.
    job = client.start("printf before; sleep 102; printf after; printf err >&2")
    out = err = b""
    offsets = (0, 0)
    polls = 0
    while True:
        result = client.poll(job, *offsets)
        out += base64.b64decode(result["stdout"])
        err += base64.b64decode(result["stderr"])
        offsets = result["stdout_offset"], result["stderr_offset"]
        polls += 1
        if result["status"] == "exited":
            break
        time.sleep(.5)
    assert time.monotonic() - start >= 102
    assert out == b"beforeafter" and err == b"err" and result["exit_code"] == 0 and polls > 100


def test_cpu_rest_no_volume_bootstrap_and_error_redaction(monkeypatch):
    monkeypatch.setattr("runpod_lifecycle.preflight.preflight_transport", lambda _: None)
    calls = []
    def create(url, **kwargs):
        calls.append(kwargs["json"])
        return httpx.Response(201, json={"id": "cpu1"})
    monkeypatch.setattr(api.httpx, "post", create)
    result = api.create_pod("test", None, "runpod/base:latest", compute_type="CPU", transport="https", https_token=TOKEN)
    assert result["id"] == "cpu1"
    payload = calls[-1]
    assert "networkVolumeId" not in payload
    assert payload["ports"] == ["8000/http"]
    assert payload["dockerEntrypoint"] == ["/bin/bash", "-lc"]
    assert TOKEN not in payload["dockerStartCmd"][0]
    def reject(url, **kwargs):
        return httpx.Response(400, json={"error": "bad " + TOKEN})
    monkeypatch.setattr(api.httpx, "post", reject)
    with pytest.raises(Exception) as caught:
        api.create_pod("test", None, "runpod/base:latest", compute_type="CPU", transport="https", https_token=TOKEN)
    assert TOKEN not in str(caught.value)


@pytest.mark.parametrize("startup", ["sdk", "rest"])
def test_real_inline_start_command(tmp_path, monkeypatch, startup):
    import socket
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    boot = agent_bootstrap(TOKEN, port)
    environment = dict(os.environ, **boot["env"], ASTRID_POD_STATE=str(tmp_path / "state"))
    # Exercise both provider startup representations with no site packages.
    command = (shlex.split(boot["docker_args"]) if startup == "sdk"
               else boot["docker_entrypoint"] + boot["docker_start_cmd"])
    command[-1] = command[-1].replace("python3 -u", shlex.quote(sys.executable) + " -S -u")
    process = subprocess.Popen(command, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    client = HttpsPodTransport("local", TOKEN, base_url=f"http://127.0.0.1:{port}", request_timeout=.2, poll_interval=.02)
    try:
        client.wait_ready(timeout=5)
        assert client.execute_command("printf booted", timeout=5) == (0, "booted", "")
        code, out, _ = client.execute_command("env | cut -d= -f1", timeout=5)
        assert code == 0
        assert "ASTRID_POD_TOKEN" not in out and "ASTRID_POD_AGENT_B64" not in out
        source, remote, output = (tmp_path / name for name in ("source", "remote", "output"))
        source.write_bytes(b"x" * CHUNK + b"bootstrap upload")
        client.put(source, str(remote))
        client.get(str(remote), output)
        assert output.read_bytes() == source.read_bytes()
    finally:
        client.disconnect()
        process.terminate()
        process.wait(timeout=5)
        process.stderr.close()


def test_both_download_framings_and_chunk_corruption(agent, tmp_path, monkeypatch):
    _, client = agent
    source = tmp_path / "source"
    source.write_bytes(b"a" * (CHUNK + 123))
    ranged = tmp_path / "ranged"
    client.get(str(source), ranged, use_ranges=True)
    assert ranged.read_bytes() == source.read_bytes()
    real = client._request
    def corrupt(method, path, **kwargs):
        response = real(method, path, **kwargs)
        if path == "/download-chunk":
            data = response.json()
            data["data"] = base64.b64encode(b"bad").decode()
            return httpx.Response(200, json=data)
        return response
    monkeypatch.setattr(client, "_request", corrupt)
    with pytest.raises(PodTransportError, match="invalid download chunk"):
        client.get(str(source), tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_reupload_after_staging_refresh_removes_old_tree(agent, tmp_path):
    _, client = agent
    source, remote = tmp_path / "source", tmp_path / "remote"
    source.write_bytes(b"same immutable bytes")
    client.put(source, str(remote))
    remote.unlink()
    client.put(source, str(remote))
    assert remote.read_bytes() == source.read_bytes()
    remote.write_bytes(b"drifted")
    client.put(source, str(remote))
    assert remote.read_bytes() == source.read_bytes()


def test_upload_resume_after_server_restart(agent, tmp_path):
    server, client = agent
    source, remote = tmp_path / "source", tmp_path / "remote"
    source.write_bytes(b"x" * (CHUNK + 19))
    state = client._request("POST", "/uploads", json={"path": str(remote), "size": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "reserved": False}).json()
    client._request("PUT", "/uploads/" + state["id"], params={"offset": 0}, content=b"x" * CHUNK)
    # A second server reads the same durable upload journal, with no jobs.
    restarted = AgentServer(("127.0.0.1", 0), TOKEN, server.state_dir)
    thread = threading.Thread(target=restarted.serve_forever, daemon=True)
    thread.start()
    recovered = HttpsPodTransport("local", TOKEN, base_url=f"http://127.0.0.1:{restarted.server_port}", poll_interval=.01)
    try:
        recovered.put(source, str(remote))
        assert remote.read_bytes() == source.read_bytes()
    finally:
        recovered.close()
        restarted.shutdown()
        restarted.server_close()
        thread.join(2)


def _detached_state(client, job, root, want, timeout=10):
    deadline = time.monotonic() + timeout
    while True:
        state = client.detached_poll(job["id"], root=root)
        if state["state"] == want or time.monotonic() >= deadline:
            return state
        time.sleep(0.02)


def _gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        # An unreaped zombie has been killed already; only a live state counts.
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] == "Z"
    except OSError:
        # Without /proc, an existing process is conservatively still live.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        return False


def test_gone_checks_live_process_when_proc_is_unavailable(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("/proc unavailable")
    monkeypatch.setattr(Path, "read_text", unavailable)
    assert not _gone(os.getpid())


def test_detached_job_without_setsid_binary(agent, tmp_path, monkeypatch):
    _, client = agent
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("bash", "mkdir", "base64", "mv"):
        executable = shutil.which(name)
        assert executable is not None
        (bin_dir / name).symlink_to(executable)
    (bin_dir / "python3").symlink_to(sys.executable)
    monkeypatch.setenv("PATH", str(bin_dir))
    root = str(tmp_path / "volume" / ".astrid-jobs")
    job = client.detached_start("echo complete", root=root)
    state = _detached_state(client, job, root, "exited")
    assert state["exit_code"] == 0
    assert (Path(job["dir"]) / "stdout").read_text() == "complete\n"


def test_detached_job_start_poll_writes_exit_file(agent, tmp_path):
    _, client = agent
    root = str(tmp_path / "volume" / ".astrid-jobs")
    job = client.detached_start("echo out; echo err >&2; sleep 0.2; exit 7", root=root)
    state = _detached_state(client, job, root, "exited")
    assert state["state"] == "exited" and state["exit_code"] == 7
    directory = Path(job["dir"])
    assert (directory / "exit").read_text() == "7\n"
    assert (directory / "stdout").read_text() == "out\n"
    assert (directory / "stderr").read_text() == "err\n"
    assert state["pid"] == int((directory / "pid").read_text())
    assert oct(directory.stat().st_mode & 0o777) == "0o700"


def test_detached_job_cancel_signals_group_and_keeps_state(agent, tmp_path):
    _, client = agent
    root = str(tmp_path / "volume" / ".astrid-jobs")
    child_pid = tmp_path / "child.pid"
    job = client.detached_start(f"sleep 30 & echo $! > {shlex.quote(str(child_pid))}; wait", root=root)
    for _ in range(200):
        if child_pid.exists() and child_pid.read_text().strip():
            break
        time.sleep(0.02)
    running = _detached_state(client, job, root, "running")
    assert running["state"] == "running"
    assert not _gone(int(child_pid.read_text()))
    assert os.getpgid(running["pid"]) == running["pid"]
    assert os.getsid(running["pid"]) == running["pid"]
    cancelled = client.detached_cancel(job["id"], root=root)
    assert cancelled["state"] == "cancelled" and cancelled["exit_code"] is None
    assert not (Path(job["dir"]) / "exit").exists()
    deadline = time.monotonic() + 5
    while not _gone(int(child_pid.read_text())) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _gone(int(child_pid.read_text()))
    # A job that already finished keeps its exit code under cancel.
    done = client.detached_start("exit 3", root=root)
    _detached_state(client, done, root, "exited")
    assert client.detached_cancel(done["id"], root=root)["exit_code"] == 3


def test_client_timeout_does_not_cancel_detached_job(agent, tmp_path, monkeypatch):
    _, client = agent
    root = str(tmp_path / "volume" / ".astrid-jobs")
    job_id = uuid.uuid4().hex
    cancels = []
    real_cancel = client.cancel
    monkeypatch.setattr(client, "cancel", lambda job: cancels.append(job) or real_cancel(job))
    # The client's wait for the launcher times out, as a dropped connection would.
    with monkeypatch.context() as timed_out:
        def dropped(*_args, **_kwargs):
            raise PodTransportError("pod HTTPS request failed")
        timed_out.setattr(client, "poll", dropped)
        with pytest.raises(PodTransportError, match="request failed"):
            client.detached_start("sleep 0.3; echo done; exit 7", root=root, job_id=job_id)
    state = _detached_state(client, {"id": job_id}, root, "exited")
    assert state["exit_code"] == 7
    assert (Path(root) / job_id / "stdout").read_text() == "done\n"
    assert cancels == []


def test_detached_job_refuses_relaunch_and_bad_input(agent, tmp_path):
    _, client = agent
    root = str(tmp_path / "volume" / ".astrid-jobs")
    job_id = uuid.uuid4().hex
    client.detached_start("echo first", root=root, job_id=job_id)
    _detached_state(client, {"id": job_id}, root, "exited")
    with pytest.raises(PodTransportError, match="launch failed"):
        client.detached_start("echo second", root=root, job_id=job_id)
    assert (Path(root) / job_id / "stdout").read_text() == "first\n"
    with pytest.raises(ValueError):
        client.detached_poll("../escape", root=root)
    with pytest.raises(ValueError):
        client.detached_cancel(job_id, root="relative/root")
    assert client.detached_poll(uuid.uuid4().hex, root=root)["state"] == "missing"


def _record_requests(client):
    """Record (method, path) for every request the client sends, in order."""
    seen = []
    client.http.event_hooks["request"].append(lambda request: seen.append((request.method, request.url.path)))
    return seen


def _legacy_detached_poll(client, job_id, root):
    """The former shell-job read, kept only as the oracle for field equivalence."""
    directory = f"{root}/{job_id}"
    script = (
        f"d={shlex.quote(directory)}\n"
        'if [ -f "$d/exit" ]; then state=exited; code=$(cat "$d/exit")\n'
        'elif [ -f "$d/status" ]; then state=$(cat "$d/status"); code=\n'
        "else state=missing; code=; fi\n"
        'pid=$(cat "$d/pid" 2>/dev/null || true)\n'
        'printf "state=%s\\nexit_code=%s\\npid=%s\\n" "$state" "$code" "$pid"\n'
    )
    code, stdout, _stderr = client.execute_command(script, timeout=15)
    assert code == 0
    return _detached_record(job_id, directory, dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line))


def test_detached_poll_creates_no_agent_job_record(agent, tmp_path):
    server, client = agent
    root = str(tmp_path / "volume" / ".astrid-jobs")
    running = client.detached_start("sleep 30", root=root)
    _detached_state(client, running, root, "running")
    done = client.detached_start("exit 4", root=root)
    _detached_state(client, done, root, "exited")
    records = len(server.jobs)
    seen = _record_requests(client)
    try:
        for job in (running, done):
            client.detached_poll(job["id"], root=root)
        polled, records_after = list(seen), len(server.jobs)
    finally:
        client.detached_cancel(running["id"], root=root)
    assert polled and set(polled) == {("POST", "/download-chunk")}
    assert records_after == records


def test_detached_poll_fields_match_the_former_shell_read(agent, tmp_path):
    _, client = agent
    root = str(tmp_path / "volume" / ".astrid-jobs")
    running = client.detached_start("sleep 30", root=root)
    _detached_state(client, running, root, "running")
    exited = client.detached_start("echo out; exit 7", root=root)
    _detached_state(client, exited, root, "exited")
    try:
        for job, state, code in ((running, "running", None), (exited, "exited", 7)):
            polled = client.detached_poll(job["id"], root=root)
            assert polled == _legacy_detached_poll(client, job["id"], root)
            assert polled == {"id": job["id"], "dir": job["dir"], "state": state, "exit_code": code,
                              "pid": int((Path(job["dir"]) / "pid").read_text())}
    finally:
        client.detached_cancel(running["id"], root=root)


def test_detached_poll_reads_state_on_an_engine_proxy_agent(tmp_path, monkeypatch):
    # Engine-enabled agents refuse every request that carries a Range header.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    server = AgentServer(("127.0.0.1", 0), TOKEN, tmp_path / "state",
                         engine_proxy={"endpoint_id": "engine-a", "port": 9})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HttpsPodTransport("local", TOKEN, base_url=f"http://127.0.0.1:{server.server_port}",
                               request_timeout=0.5, poll_interval=0.02)
    root = str(tmp_path / "volume" / ".astrid-jobs")
    try:
        job = client.detached_start("echo out; exit 5", root=root)
        assert _detached_state(client, job, root, "exited")["exit_code"] == 5
    finally:
        client.disconnect()
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize("job_id", ["../escape", "A" * 32, "a" * 31, "a" * 33, "g" * 32, "a" * 32 + "/x", ""])
def test_detached_poll_rejects_bad_job_id_before_any_request(agent, tmp_path, job_id):
    _, client = agent
    root = str(tmp_path / "volume" / ".astrid-jobs")
    seen = _record_requests(client)
    with pytest.raises(ValueError, match="invalid detached job ID"):
        client.detached_poll(job_id, root=root)
    with pytest.raises(ValueError, match="absolute path"):
        client.detached_poll(uuid.uuid4().hex, root="relative/root")
    assert seen == []


def test_rate_limit_error_names_status_and_body_without_retrying(agent):
    server, client = agent

    class Finished:
        pid = 0

        def poll(self):
            return 0

    # Stand-in records fill the 128-record cap, so the agent's own 429 is real.
    for index in range(128):
        server.jobs[f"{index:032x}"] = {"process": Finished(), "directory": None, "command": "true"}
    seen = _record_requests(client)
    with pytest.raises(PodTransportError, match=r"HTTP 429; body: .*job limit reached; rotate claim") as caught:
        client.start("echo hi")
    assert "Retry-After" not in str(caught.value)
    assert seen == [("POST", "/jobs")]


def test_rate_limit_detail_keeps_retry_after_redacts_token_and_never_retries(agent, monkeypatch):
    _, client = agent
    calls = []

    def throttled(request):
        calls.append(request)
        return httpx.Response(429, content=f'{{"error": "slow down {TOKEN}"}}'.encode(),
                              headers={"Retry-After": "30"})

    stub = httpx.Client(base_url=str(client.http.base_url), headers=client.http.headers,
                        transport=httpx.MockTransport(throttled))
    monkeypatch.setattr(client, "http", stub)
    with pytest.raises(PodTransportError) as caught:
        client.start("echo hi")
    text = str(caught.value)
    assert "HTTP 429; body: " in text and "slow down [redacted]" in text and "Retry-After: 30" in text
    assert TOKEN not in text and len(calls) == 1


def test_job_poll_rejects_bad_id_before_pod_discovery(monkeypatch, capsys):
    from runpod_lifecycle import cli
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)

    async def no_discovery(*_args, **_kwargs):
        raise AssertionError("pod discovery must not run for a malformed job ID")

    monkeypatch.setattr("runpod_lifecycle.cli.discovery.get_pod", no_discovery)
    assert cli.main(["--transport", "https", "job", "poll", "pod-1", "../escape"]) == 2
    assert "invalid detached job ID" in capsys.readouterr().err


def test_job_verbs_parse_under_https_transport():
    from runpod_lifecycle.cli import build_parser
    args = build_parser().parse_args(["--transport", "https", "job", "start", "pod1", "--command", "echo hi"])
    assert (args.cmd, args.job_cmd, args.pod_id, args.command) == ("job", "start", "pod1", "echo hi")
    args = build_parser().parse_args(["job", "cancel", "pod1", "a" * 32])
    assert (args.job_cmd, args.job_id) == ("cancel", "a" * 32)
