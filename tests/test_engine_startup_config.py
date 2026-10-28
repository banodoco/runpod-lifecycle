"""Engine startup selection for the one original HTTPS agent.

Provider clients are stubbed; nothing here contacts RunPod, starts a pod or
reads the shared Astrid env file (conftest isolates ASTRID_ENV_FILE).
"""
from __future__ import annotations

import asyncio
import base64
import copy
import json
import os
import re
import shlex
import socket
import stat
import subprocess
import sys
from unittest.mock import MagicMock

import httpx
import pytest

from runpod_lifecycle import api, cli, pod_agent
from runpod_lifecycle.config import RunPodConfig
from runpod_lifecycle.engine_startup import SCHEMA, load_engine_startup, normalize_engine_startup
from runpod_lifecycle.https import agent_bootstrap

TOKEN = "original-claim-bearer-" + "x" * 32
STATE = "/workspace/astrid/agent-state"
INPUT = "/workspace/ComfyUI/input/approved-image.png"
OUTPUT = "/workspace/ComfyUI/output"


def selection(**changes):
    value = {"schema": SCHEMA,
             "engine_proxy": {"endpoint_id": "vibecomfy-h3", "port": 8188,
                              "input_files": [INPUT], "output_roots": [OUTPUT]},
             "state_dir": STATE}
    value.update(changes)
    return value


def write(tmp_path, value, name="engine-startup.json", mode=0o600):
    path = tmp_path / name
    path.write_text(value if isinstance(value, str) else json.dumps(value))
    path.chmod(mode)
    return str(path)


def decoded_engine(boot):
    return json.loads(base64.b64decode(boot["env"][pod_agent.ENGINE_B64_ENV], validate=True))


# ---------------------------------------------------------------- file parser


def test_selection_round_trips_and_normalization_is_idempotent(tmp_path):
    loaded = load_engine_startup(write(tmp_path, selection()))
    assert loaded == selection()
    assert normalize_engine_startup(loaded) == loaded
    assert normalize_engine_startup(loaded, volume_mount_path="/workspace") == loaded
    assert set(loaded) == {"schema", "engine_proxy", "state_dir"}  # no secret-bearing field exists


def test_loaded_selection_is_a_deep_copy():
    value = selection()
    normalized = normalize_engine_startup(value)
    value["engine_proxy"]["input_files"].append("/workspace/other.png")
    assert normalized["engine_proxy"]["input_files"] == [INPUT]


def mutated(path, replacement):
    value = selection()
    target = value
    *parents, last = path
    for part in parents:
        target = target[part]
    if replacement is KeyError:
        del target[last]
    else:
        target[last] = replacement
    return value


BAD_SELECTIONS = {
    "no-schema": mutated(["schema"], KeyError),
    "other-schema": mutated(["schema"], "astrid.engine-startup.v2"),
    "extra-top-level-secret": {**selection(), "token": "must-not-be-here"},
    "extra-proxy-host": mutated(["engine_proxy", "host"], "example.com"),
    "other-endpoint": mutated(["engine_proxy", "endpoint_id"], "another-engine"),
    "other-port": mutated(["engine_proxy", "port"], 8189),
    "comfy-public-port": mutated(["engine_proxy", "port"], 8000),
    "bool-port": mutated(["engine_proxy", "port"], True),
    "string-port": mutated(["engine_proxy", "port"], "8188"),
    "no-inputs": mutated(["engine_proxy", "input_files"], []),
    "missing-inputs": mutated(["engine_proxy", "input_files"], KeyError),
    "no-outputs": mutated(["engine_proxy", "output_roots"], []),
    "missing-outputs": mutated(["engine_proxy", "output_roots"], KeyError),
    "string-inputs": mutated(["engine_proxy", "input_files"], INPUT),
    "relative-input": mutated(["engine_proxy", "input_files"], ["input/a.png"]),
    "dotdot-input": mutated(["engine_proxy", "input_files"], ["/workspace/../etc/a.png"]),
    "dot-segment-output": mutated(["engine_proxy", "output_roots"], ["/workspace/./output"]),
    "double-slash-output": mutated(["engine_proxy", "output_roots"], ["/workspace//output"]),
    "duplicate-inputs": mutated(["engine_proxy", "input_files"], [INPUT, INPUT]),
    "too-many-inputs": mutated(["engine_proxy", "input_files"], [f"/workspace/in/{i}.png" for i in range(33)]),
    "no-state": mutated(["state_dir"], KeyError),
    "relative-state": mutated(["state_dir"], "workspace/state"),
    "dotdot-state": mutated(["state_dir"], "/workspace/a/../state"),
    "trailing-slash-state": mutated(["state_dir"], STATE + "/"),
    "quote-state": mutated(["state_dir"], '/workspace/st"ate'),
    "space-state": mutated(["state_dir"], "/workspace/st ate"),
    "state-is-mount": mutated(["state_dir"], "/workspace"),
    "state-off-volume": mutated(["state_dir"], "/tmp/astrid-pod-agent"),
    "state-sibling-prefix": mutated(["state_dir"], "/workspace-other/state"),
    "state-is-root": mutated(["state_dir"], "/"),
    "output-contains-state": mutated(["engine_proxy", "output_roots"], ["/workspace/astrid"]),
    "output-is-state": mutated(["engine_proxy", "output_roots"], [STATE]),
    "output-under-state": mutated(["engine_proxy", "output_roots"], [STATE + "/out"]),
    "input-under-state": mutated(["engine_proxy", "input_files"], [STATE + "/x.png"]),
}


@pytest.mark.parametrize("name", sorted(BAD_SELECTIONS))
def test_bad_selection_is_refused(name):
    with pytest.raises((ValueError, TypeError)):
        normalize_engine_startup(copy.deepcopy(BAD_SELECTIONS[name]), volume_mount_path="/workspace")


def test_non_mapping_selection_refused():
    for value in (None, [], "x", 3):
        with pytest.raises(ValueError):
            normalize_engine_startup(value)


@pytest.mark.parametrize("text", [
    "{", "[]", "null", '{"schema": "' + SCHEMA + '", "schema": "' + SCHEMA + '"}',
    json.dumps(selection()).replace('"port": 8188', '"port": NaN'),
    json.dumps(selection()) + " " * 20000,
])
def test_malformed_oversized_or_duplicate_json_refused(tmp_path, text):
    with pytest.raises(ValueError):
        load_engine_startup(write(tmp_path, text))


def test_duplicate_nested_key_refused(tmp_path):
    text = json.dumps(selection()).replace('"port": 8188', '"port": 8188, "port": 8188')
    with pytest.raises(ValueError):
        load_engine_startup(write(tmp_path, text))


def test_unsafe_files_refused(tmp_path):
    good = write(tmp_path, selection())
    assert load_engine_startup(good) == selection()
    link = tmp_path / "link.json"
    link.symlink_to(good)
    for unsafe in (str(link), "relative.json", str(tmp_path / "missing.json"), str(tmp_path),
                   str(tmp_path / ".." / "engine-startup.json")):
        with pytest.raises(ValueError):
            load_engine_startup(unsafe)
    with pytest.raises(ValueError):
        load_engine_startup(write(tmp_path, selection(), "writable.json", 0o666))
    hardlinked = tmp_path / "hard.json"
    os.link(good, hardlinked)
    with pytest.raises(ValueError):
        load_engine_startup(str(hardlinked))
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    with pytest.raises(ValueError):
        load_engine_startup(str(fifo))
    parent_link = tmp_path / "parent"
    parent_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        load_engine_startup(str(parent_link / "engine-startup.json"))


# ------------------------------------------------------------ bootstrap shape


def test_bootstrap_without_selection_is_unchanged():
    boot = agent_bootstrap(TOKEN)
    assert set(boot["env"]) == {"ASTRID_POD_TOKEN", "ASTRID_POD_AGENT_B64", "ASTRID_POD_PORT"}
    assert boot["ports"] == "8000/http"


def test_bootstrap_with_selection_adds_only_state_and_engine_env():
    chosen = normalize_engine_startup(selection())
    boot = agent_bootstrap(TOKEN, engine_proxy=chosen["engine_proxy"], state_dir=chosen["state_dir"])
    assert set(boot["env"]) == {"ASTRID_POD_TOKEN", "ASTRID_POD_AGENT_B64", "ASTRID_POD_PORT",
                                pod_agent.ENGINE_B64_ENV, "ASTRID_POD_STATE"}
    assert boot["env"]["ASTRID_POD_STATE"] == STATE
    assert decoded_engine(boot) == chosen["engine_proxy"]
    assert boot["ports"] == "8000/http"
    for text in (boot["docker_args"], *boot["docker_entrypoint"], *boot["docker_start_cmd"],
                 *[v for k, v in boot["env"].items() if k != "ASTRID_POD_TOKEN"]):
        assert TOKEN not in text
    # Every env value is plain text that the SDK can safely embed in GraphQL.
    assert all(re.fullmatch(r"[A-Za-z0-9+/=_.:@-]*", v) for v in boot["env"].values())


def test_state_requires_a_slot_and_a_safe_path():
    with pytest.raises(ValueError):
        agent_bootstrap(TOKEN, state_dir=STATE)
    slot = normalize_engine_startup(selection())["engine_proxy"]
    for state in ("relative", "/workspace/../x", '/a"b', "/a b", "/", STATE + "/", 5):
        with pytest.raises(ValueError):
            agent_bootstrap(TOKEN, engine_proxy=slot, state_dir=state)


def test_legacy_slot_without_state_keeps_json_env():
    boot = agent_bootstrap(TOKEN, engine_proxy={"endpoint_id": "single", "port": 8188})
    assert json.loads(boot["env"][pod_agent.ENGINE_ENV]) == {"endpoint_id": "single", "port": 8188}
    assert "ASTRID_POD_STATE" not in boot["env"] and pod_agent.ENGINE_B64_ENV not in boot["env"]


# ------------------------------------------------- original create payloads


@pytest.fixture
def provider(monkeypatch, runpod_sdk_mock):
    """Both stubbed provider paths; any call is recorded, none leaves the process."""
    runpod_sdk_mock.create_pod.return_value = {"id": "gpu-pod"}
    posts = []

    def post(url, **kwargs):
        posts.append(kwargs["json"])
        return httpx.Response(201, json={"id": "cpu-pod"})

    monkeypatch.setattr(api.httpx, "post", post)
    return MagicMock(sdk=runpod_sdk_mock, posts=posts)


def assert_agent_only(boot_env, ports):
    assert ports in ("8000/http", ["8000/http"])
    assert boot_env["ASTRID_POD_PORT"] == "8000"
    assert "8188" not in json.dumps(ports) and "22/tcp" not in json.dumps(ports) and "8888" not in json.dumps(ports)


def test_gpu_sdk_create_payload_with_selection(provider):
    result = api.create_pod("rpa-test-key", "gpu-1", "image:tag", transport="https", https_token=TOKEN,
                            network_volume_id="vol", engine_startup=selection())
    assert result["id"] == "gpu-pod"
    provider.sdk.create_pod.assert_called_once()
    params = provider.sdk.create_pod.call_args.kwargs
    assert_agent_only(params["env"], params["ports"])
    assert params["ports"] == "8000/http"
    assert re.fullmatch(re.escape(STATE) + r"/[0-9a-f]{16}", params["env"]["ASTRID_POD_STATE"])
    assert params["env"]["ASTRID_POD_TOKEN"] == TOKEN
    assert json.loads(base64.b64decode(params["env"][pod_agent.ENGINE_B64_ENV])) == selection()["engine_proxy"]
    assert params["volume_mount_path"] == "/workspace"
    assert params["docker_args"].startswith("bash -lc ")
    assert "rpa-test-key" not in json.dumps(params) and TOKEN not in params["docker_args"]
    assert "template_id" not in params
    assert not provider.posts


def test_gpu_graphql_text_has_no_unescaped_env_text(provider):
    from runpod.api.mutations.pods import generate_pod_deployment_mutation
    api.create_pod("k", "gpu-1", "image:tag", transport="https", https_token=TOKEN, engine_startup=selection())
    mutation = generate_pod_deployment_mutation(**provider.sdk.create_pod.call_args.kwargs)
    assert 'ports: "8000/http"' in mutation
    assert "8188/" not in mutation and "vibecomfy-h3" not in mutation


def test_cpu_rest_create_payload_with_selection(provider):
    api.create_pod("k", None, "image:tag", compute_type="CPU", transport="https", https_token=TOKEN,
                   network_volume_id="vol", engine_startup=selection())
    assert not provider.sdk.create_pod.called
    payload, = provider.posts
    assert_agent_only(payload["env"], payload["ports"])
    assert payload["env"]["ASTRID_POD_STATE"].startswith(STATE + "/")
    assert decoded_engine({"env": payload["env"]}) == selection()["engine_proxy"]
    assert payload["dockerEntrypoint"] == ["/bin/bash", "-lc"] and TOKEN not in payload["dockerStartCmd"][0]
    assert payload["networkVolumeId"] == "vol" and payload["volumeMountPath"] == "/workspace"


def test_cpu_sdk_create_payload_with_selection(provider):
    provider.sdk.create_pod.return_value = {"id": "cpu-sdk"}
    api.create_pod("k", None, "image:tag", compute_type="CPU", cpu_api="sdk", transport="https",
                   https_token=TOKEN, network_volume_id="vol", engine_startup=selection())
    params = provider.sdk.create_pod.call_args.kwargs
    assert params["ports"] == "8000/http" and params["env"]["ASTRID_POD_STATE"].startswith(STATE + "/")
    assert not provider.posts


def test_default_create_payloads_are_unchanged(provider):
    api.create_pod("k", "gpu-1", "image:tag", transport="https", https_token=TOKEN)
    env = provider.sdk.create_pod.call_args.kwargs["env"]
    assert set(env) == {"ASTRID_POD_TOKEN", "ASTRID_POD_AGENT_B64", "ASTRID_POD_PORT"}
    api.create_pod("k", None, "image:tag", compute_type="CPU", transport="https", https_token=TOKEN)
    assert set(provider.posts[-1]["env"]) == set(env)


@pytest.mark.parametrize("name", ["bad-selection", "ssh", "template", "foreign-mount", "unsafe-env", "other-port"])
@pytest.mark.parametrize("compute", ["GPU", "CPU"])
def test_refusal_happens_before_any_provider_mutation(provider, name, compute):
    options = dict(transport="https", https_token=TOKEN, engine_startup=selection(), compute_type=compute)
    if name == "bad-selection":
        options["engine_startup"] = BAD_SELECTIONS["other-port"]
    if name == "ssh":
        options["transport"] = "ssh"
    if name == "template":
        options["template_id"] = "some-template"
    if name == "foreign-mount":
        options["volume_mount_path"] = "/mnt/elsewhere"
    if name == "unsafe-env":
        options["env_vars"] = {"EXTRA": 'a"b'}
    if name == "other-port":
        options["engine_startup"] = mutated(["engine_proxy", "port"], 8000)
    with pytest.raises(ValueError):
        api.create_pod("k", "gpu-1" if compute == "GPU" else None, "image:tag", **options)
    assert not provider.sdk.create_pod.called and not provider.posts


# --------------------------------------------------- config, lifecycle and CLI


def https_config(**kw):
    return RunPodConfig(api_key="k", transport="https", https_token=TOKEN, **kw)


def test_config_carries_normalized_selection_and_survives_merge():
    config = https_config(engine_startup=selection())
    assert config.engine_startup == selection()
    assert config.merge(name_prefix="again").engine_startup == selection()
    assert TOKEN not in json.dumps(config.engine_startup) and "token" not in repr(config.engine_startup)
    assert RunPodConfig(api_key="k", transport="https").engine_startup is None


def test_config_refuses_ssh_or_off_volume_selection():
    with pytest.raises(ValueError):
        RunPodConfig(api_key="k", engine_startup=selection())
    with pytest.raises(ValueError):
        https_config(engine_startup=selection(), volume_mount_path="/mnt/elsewhere")
    with pytest.raises(ValueError):
        https_config(engine_startup=BAD_SELECTIONS["other-port"])


def test_from_env_reads_selection_file_and_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setenv("RUNPOD_TRANSPORT", "https")
    monkeypatch.setenv("RUNPOD_HTTPS_TOKEN", TOKEN)
    monkeypatch.delenv("RUNPOD_ENGINE_STARTUP_FILE", raising=False)
    assert RunPodConfig.from_env().engine_startup is None
    monkeypatch.setenv("RUNPOD_ENGINE_STARTUP_FILE", write(tmp_path, selection()))
    assert RunPodConfig.from_env().engine_startup == selection()
    assert RunPodConfig.from_env(engine_startup=None).engine_startup is None
    monkeypatch.setenv("RUNPOD_ENGINE_STARTUP_FILE", write(tmp_path, "{", "bad.json"))
    with pytest.raises(ValueError):
        RunPodConfig.from_env()


def test_launch_passes_selection_to_the_single_create(monkeypatch):
    from runpod_lifecycle.lifecycle import launch
    create = MagicMock(return_value={"id": "pod-1"})
    monkeypatch.setattr("runpod_lifecycle.lifecycle.find_gpu_type", lambda g, k: {"id": "gpu-1", "displayName": g})
    monkeypatch.setattr("runpod_lifecycle.lifecycle.create_pod", create)
    asyncio.run(launch(https_config(engine_startup=selection(), storage_volumes=(), storage_name=None),
                       name="claim"))
    create.assert_called_once()
    kwargs = create.call_args.kwargs
    assert kwargs["engine_startup"] == selection() and kwargs["transport"] == "https"
    assert kwargs["https_token"] == TOKEN
    plain = MagicMock(return_value={"id": "pod-2"})
    monkeypatch.setattr("runpod_lifecycle.lifecycle.create_pod", plain)
    asyncio.run(launch(https_config(storage_volumes=(), storage_name=None), name="claim"))
    assert "engine_startup" not in plain.call_args.kwargs


def test_cli_option_loads_the_file_and_default_is_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setenv("RUNPOD_TRANSPORT", "https")
    token = tmp_path / "token"
    token.write_text(TOKEN)
    token.chmod(0o600)
    monkeypatch.setenv("RUNPOD_HTTPS_TOKEN_FILE", str(token))
    monkeypatch.delenv("RUNPOD_ENGINE_STARTUP_FILE", raising=False)
    parser = cli.build_parser()
    assert cli._resolve_config(parser.parse_args(["launch"])).engine_startup is None
    path = write(tmp_path, selection())
    args = parser.parse_args(["launch", "--engine-startup", path])
    assert cli._resolve_config(args).engine_startup == selection()
    for bad in (write(tmp_path, "{", "bad.json"), "relative.json", str(tmp_path / "absent.json")):
        with pytest.raises(ValueError):
            cli._resolve_config(parser.parse_args(["launch", "--engine-startup", bad]))
    monkeypatch.setenv("RUNPOD_TRANSPORT", "ssh")
    with pytest.raises(ValueError):
        cli._resolve_config(parser.parse_args(["launch", "--engine-startup", path]))


def test_cli_refusal_precedes_any_provider_call(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    forbidden = MagicMock(side_effect=AssertionError("provider call"))
    monkeypatch.setattr("runpod_lifecycle.cli._launch", forbidden)
    monkeypatch.setattr("runpod_lifecycle.cli._launch_when_available", forbidden)
    monkeypatch.setattr(api, "create_pod", forbidden)
    assert cli.main(["launch", "--engine-startup", write(tmp_path, "{")]) == 1
    assert "engine startup" in capsys.readouterr().err
    forbidden.assert_not_called()


# -------------------------------------------- real isolated agent bootstrap


@pytest.mark.parametrize("startup", ["sdk", "rest"])
def test_isolated_agent_start_projects_private_state(tmp_path, monkeypatch, startup):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    state = tmp_path / "volume" / "agent-state"
    chosen = normalize_engine_startup({
        **selection(), "state_dir": str(state),
        "engine_proxy": {"endpoint_id": "vibecomfy-h3", "port": 8188,
                         "input_files": [str(tmp_path / "input" / "approved.png")],
                         "output_roots": [str(tmp_path / "output")]}})
    boot = agent_bootstrap(TOKEN, port, engine_proxy=chosen["engine_proxy"], state_dir=chosen["state_dir"])
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ASTRID_", "RUNPOD_"))}
    env.update(boot["env"])
    command = (shlex.split(boot["docker_args"]) if startup == "sdk"
               else boot["docker_entrypoint"] + boot["docker_start_cmd"])
    command[-1] = command[-1].replace("python3 -u", shlex.quote(sys.executable) + " -S -u")
    process = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    headers = {"Authorization": "Bearer " + TOKEN}
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(base + "/health", headers=headers, trust_env=False, timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if process.poll() is not None:
                pytest.fail(process.stderr.read().decode())
            import time
            time.sleep(.05)
        else:
            pytest.fail("agent did not start")
        # Health is served before any engine exists; no readiness is claimed.
        assert not (state / "engine-readiness.json").exists()
        bearer = state / "engine-bearer.token"
        assert bearer.read_text() == TOKEN
        assert stat.S_IMODE(bearer.stat().st_mode) == 0o600
        assert stat.S_IMODE(state.stat().st_mode) == 0o700
        # The configured slot has no upstream: the agent answers, never a ComfyUI port.
        assert httpx.get(base + "/engines/vibecomfy-h3/object_info", trust_env=False).status_code == 401
        assert httpx.get(base + "/engines/other/object_info", headers=headers, trust_env=False).status_code == 404
        assert httpx.get(base + "/engines/vibecomfy-h3/object_info", headers=headers, trust_env=False).status_code >= 400
    finally:
        process.terminate()
        process.wait(timeout=5)
        process.stderr.close()


def test_observation_policy_stays_read_only_beside_the_selection(tmp_path):
    public = tmp_path / "public.json"
    policy = {"schema": pod_agent.OBSERVATION_SCHEMA, "roots": [], "files": [str(public)]}
    chosen = normalize_engine_startup({**selection(), "state_dir": str(tmp_path / "state"),
                                       "engine_proxy": {"endpoint_id": "vibecomfy-h3", "port": 8188,
                                                        "input_files": [str(tmp_path / "in" / "a.png")],
                                                        "output_roots": [str(tmp_path / "out")]}})
    boot = agent_bootstrap(TOKEN, observation_policy=policy, engine_proxy=chosen["engine_proxy"],
                           state_dir=chosen["state_dir"])
    assert pod_agent.OBSERVATION_ENV in boot["env"] and boot["env"]["ASTRID_POD_STATE"] == chosen["state_dir"]
    server = pod_agent.AgentServer(("127.0.0.1", 0), TOKEN, chosen["state_dir"],
                                   observation_policy=policy, engine_proxy=decoded_engine(boot))
    import threading
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        headers = {"Authorization": "Bearer " + TOKEN}
        assert httpx.post(base + "/jobs", json={}, headers=headers, trust_env=False).status_code == 403
        assert httpx.get(base + "/health", headers=headers, trust_env=False).status_code == 200
        assert (tmp_path / "state" / "engine-bearer.token").read_text() == TOKEN
    finally:
        server.shutdown()
        server.server_close()


# ------------------------------------------- second-round review conditions

HOSTILE = {
    "quote": 'a"b',
    "quote-injection": 'x", dockerArgs: "rm -rf /',
    "newline": "a\nb",
    "trailing-newline": "ab\n",
    "carriage-return": "a\rb",
    "backslash": "a\\b",
    "escaped-quote": 'a\\"b',
    "brace": "a}b",
    "close-all": '"}]}) { id } } #',
    "space": "a b",
    "unicode": "café",
    "unicode-quote": "a“b",
    "nul": "a\x00b",
    "long-with-quote": "a" * 20000 + '"',
    "long-with-newline": "a" * 20000 + "\n",
    "empty-ok-but-not-str": 7,
}

_STRING = r'"((?:[^"\\\n]|\\.)*)"'
_ENV = r'\[((?:\{ key: "[^"\\\n]*", value: "[^"\\\n]*" \}(?:, )?)*)\]'
_PAIR = re.compile(r'([A-Za-z]+): (?:%s|%s|([A-Za-z0-9_]+))(?:, |\Z)' % (_STRING, _ENV), re.S)


def parse_mutation(text):
    """Strictly parse the SDK's pod mutation: every character must be consumed
    by a known token, so an injected quote or brace cannot hide in a substring."""
    lines = [line.strip() for line in text.strip().splitlines()]
    assert lines[:3] == ["mutation {", "podFindAndDeployOnDemand(", "input: {"], lines[:3]
    assert lines[4:7] == ["}", ") {", "id"] and lines[-3:] == ["}", "}", "}"], lines
    body, fields, position = lines[3], {}, 0
    while position < len(body):
        match = _PAIR.match(body, position)
        assert match, f"unparseable mutation at {body[position:position + 40]!r}"
        key = match.group(1)
        assert key not in fields, f"duplicate field {key}"
        if match.group(2) is not None:
            fields[key] = match.group(2)
        elif match.group(3) is not None:
            fields[key] = [tuple(re.fullmatch(r'\{ key: "(.*)", value: "(.*)" \}', item).groups())
                           for item in re.findall(r'\{[^{}]*\}', match.group(3))]
        else:
            fields[key] = ("bare", match.group(4))
        position = match.end()
    return fields


def gpu_mutation(provider):
    from runpod.api.mutations.pods import generate_pod_deployment_mutation
    return generate_pod_deployment_mutation(**provider.sdk.create_pod.call_args.kwargs)


def test_generated_mutation_parses_strictly_and_carries_only_plain_text(provider):
    api.create_pod("k", "gpu-1", "image:tag", name="claim-1", transport="https", https_token=TOKEN,
                   network_volume_id="vol-1", data_center_id="US-TX-1", engine_startup=selection(),
                   public_key_string="ssh-ed25519 AAAA key")
    fields = parse_mutation(gpu_mutation(provider))
    assert fields["name"] == "claim-1" and fields["imageName"] == "image:tag"
    assert fields["gpuTypeId"] == "gpu-1" and fields["networkVolumeId"] == "vol-1"
    assert fields["dataCenterId"] == "US-TX-1" and fields["ports"] == "8000/http"
    env = dict(fields["env"])
    assert len(env) == len(fields["env"])
    assert set(env) == {"ASTRID_POD_TOKEN", "ASTRID_POD_AGENT_B64", "ASTRID_POD_PORT",
                        pod_agent.ENGINE_B64_ENV, "ASTRID_POD_STATE"}  # no PUBLIC_KEY
    assert all(re.fullmatch(r"[A-Za-z0-9+/=_.:@-]*", v) for v in env.values())
    # Only dockerArgs is JSON-escaped by the SDK; everything else is raw text.
    assert json.loads('"' + fields["dockerArgs"] + '"') == provider.sdk.create_pod.call_args.kwargs["docker_args"]
    assert all("\\" not in value for key, value in fields.items() if isinstance(value, str) and key != "dockerArgs")


def test_strict_parser_rejects_an_injected_quote():
    from runpod.api.mutations.pods import generate_pod_deployment_mutation
    text = generate_pod_deployment_mutation(name='a"b', image_name="i", gpu_type_id="g", cloud_type="SECURE",
                                            container_disk_in_gb=5, ports="8000/http")
    with pytest.raises(AssertionError):
        parse_mutation(text)


@pytest.mark.parametrize("compute", ["GPU", "CPU"])
@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_public_key_is_refused_on_the_engine_path_not_sent(provider, name, compute):
    value = HOSTILE[name]
    options = dict(transport="https", https_token=TOKEN, engine_startup=selection(), compute_type=compute,
                   public_key_string=value)
    gpu = "gpu-1" if compute == "GPU" else None
    if name == "space":  # an ordinary SSH key is plain enough, but is never sent over HTTPS
        api.create_pod("k", gpu, "image:tag", **options)
        env = (provider.sdk.create_pod.call_args.kwargs["env"] if compute == "GPU" else provider.posts[-1]["env"])
        assert "PUBLIC_KEY" not in env
        return
    with pytest.raises(ValueError):
        api.create_pod("k", gpu, "image:tag", **options)
    assert not provider.sdk.create_pod.called and not provider.posts


def test_public_key_is_still_sent_without_engine_startup(provider):
    api.create_pod("k", "gpu-1", "image:tag", public_key_string="ssh-ed25519 AAAA key")
    assert provider.sdk.create_pod.call_args.kwargs["env"]["PUBLIC_KEY"] == "ssh-ed25519 AAAA key"


FIELDS = ["name", "image_name", "gpu_type_id", "network_volume_id", "data_center_id"]


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("hostile", sorted(HOSTILE))
def test_unsafe_provider_fields_are_refused_before_graphql(provider, field, hostile):
    arguments = {"name": "claim", "image_name": "image:tag", "network_volume_id": "vol", "data_center_id": "dc"}
    gpu = "gpu-1"
    if field == "gpu_type_id":
        gpu = HOSTILE[hostile]
    else:
        arguments[field] = HOSTILE[hostile]
    with pytest.raises(ValueError):
        api.create_pod("k", gpu, transport="https", https_token=TOKEN, engine_startup=selection(), **{
            "image_name": arguments.pop("image_name"), **arguments})
    assert not provider.sdk.create_pod.called and not provider.posts


@pytest.mark.parametrize("field", ["name", "image_name", "network_volume_id", "data_center_id"])
def test_unsafe_provider_fields_are_refused_on_the_cpu_path_too(provider, field):
    arguments = {"name": "claim", "image_name": "image:tag", "network_volume_id": "vol", "data_center_id": "dc"}
    arguments[field] = 'a"b'
    with pytest.raises(ValueError):
        api.create_pod("k", None, compute_type="CPU", transport="https", https_token=TOKEN,
                       engine_startup=selection(), **{"image_name": arguments.pop("image_name"), **arguments})
    assert not provider.sdk.create_pod.called and not provider.posts


def test_plain_provider_fields_including_a_long_one_reach_one_strictly_parsed_mutation(provider):
    long_name = "n" * 5000
    api.create_pod("k", "NVIDIA-GeForce.RTX_5090", "ghcr.io/org/img:1.2@sha256-abc", name=long_name,
                   transport="https", https_token=TOKEN, network_volume_id="v0l_1", data_center_id="EU-RO-1",
                   engine_startup=selection())
    fields = parse_mutation(gpu_mutation(provider))
    assert fields["name"] == long_name
    assert fields["imageName"] == "ghcr.io/org/img:1.2@sha256-abc"
    assert fields["gpuTypeId"] == "NVIDIA-GeForce.RTX_5090"
    assert fields["networkVolumeId"] == "v0l_1" and fields["dataCenterId"] == "EU-RO-1"


@pytest.mark.parametrize("hostile", sorted(set(HOSTILE) - {"empty-ok-but-not-str"}))
def test_unsafe_env_vars_are_refused(provider, hostile):
    with pytest.raises(ValueError):
        api.create_pod("k", "gpu-1", "image:tag", transport="https", https_token=TOKEN,
                       engine_startup=selection(), env_vars={"EXTRA": HOSTILE[hostile]})
    assert not provider.sdk.create_pod.called and not provider.posts


def big_selection(length):
    return selection(engine_proxy={
        "endpoint_id": "vibecomfy-h3", "port": 8188,
        "input_files": [f"/workspace/ComfyUI/input/{i:02d}-" + "a" * length + ".png" for i in range(32)],
        "output_roots": [OUTPUT]})


def test_oversized_selection_raises_before_any_create_call(provider):
    chosen = normalize_engine_startup(big_selection(370), volume_mount_path="/workspace")
    raw = len(json.dumps(chosen["engine_proxy"], separators=(",", ":")))
    assert 12000 < raw < 14000 and len(base64.b64encode(b"x" * raw)) > pod_agent.ENGINE_B64_LIMIT
    for compute, gpu in (("GPU", "gpu-1"), ("CPU", None)):
        with pytest.raises(ValueError, match="too large"):
            api.create_pod("k", gpu, "image:tag", compute_type=compute, transport="https", https_token=TOKEN,
                           engine_startup=big_selection(370))
    assert provider.sdk.create_pod.call_count == 0 and provider.posts == []
    with pytest.raises(ValueError, match="too large"):
        agent_bootstrap(TOKEN, engine_proxy=chosen["engine_proxy"], state_dir=STATE)


def test_largest_accepted_selection_is_accepted_by_the_agent_limit(provider):
    for length in range(300, 371):
        try:
            boot = agent_bootstrap(TOKEN, engine_proxy=normalize_engine_startup(
                big_selection(length), volume_mount_path="/workspace")["engine_proxy"], state_dir=STATE)
        except ValueError:
            break
        assert len(boot["env"][pod_agent.ENGINE_B64_ENV]) <= pod_agent.ENGINE_B64_LIMIT
    else:
        pytest.fail("the size limit never engaged")
    assert length > 300


def test_oversized_selection_launch_makes_no_provider_call(monkeypatch, provider):
    from runpod_lifecycle.lifecycle import launch
    from runpod_lifecycle.errors import AllocationUnknown
    monkeypatch.setattr("runpod_lifecycle.lifecycle.find_gpu_type", lambda g, k: {"id": "gpu-1"})
    with pytest.raises((ValueError, AllocationUnknown)):
        asyncio.run(launch(https_config(engine_startup=big_selection(370), storage_volumes=(), storage_name=None),
                           name="claim"))
    assert provider.sdk.create_pod.call_count == 0 and provider.posts == []


def claim_env(provider):
    return provider.sdk.create_pod.call_args.kwargs["env"]


def test_each_claim_on_one_volume_gets_its_own_state_dir(provider):
    results, states = [], []
    for _ in range(2):
        results.append(api.create_pod("k", "gpu-1", "image:tag", transport="https", https_token=TOKEN,
                                      network_volume_id="vol", engine_startup=selection()))
        states.append(claim_env(provider)["ASTRID_POD_STATE"])
    assert states[0] != states[1]
    assert results[0]["claim_id"] != results[1]["claim_id"]
    for result, state in zip(results, states):
        assert re.fullmatch(r"[0-9a-f]{16}", result["claim_id"])
        assert state == f"{STATE}/{result['claim_id']}" == result["state_dir"]
    # CPU REST returns the same handle fields.
    cpu = api.create_pod("k", None, "image:tag", compute_type="CPU", transport="https", https_token=TOKEN,
                         engine_startup=selection())
    assert cpu["state_dir"] == provider.posts[-1]["env"]["ASTRID_POD_STATE"] == f"{STATE}/{cpu['claim_id']}"


def test_no_claim_fields_without_engine_startup(provider):
    result = api.create_pod("k", "gpu-1", "image:tag", transport="https", https_token=TOKEN)
    assert "claim_id" not in result and "state_dir" not in result


def test_claim_state_dir_stays_beneath_the_volume_and_clear_of_grants(provider):
    result = api.create_pod("k", "gpu-1", "image:tag", transport="https", https_token=TOKEN,
                            engine_startup=selection())
    assert result["state_dir"].startswith("/workspace/") and ".." not in result["state_dir"]
    for path in (INPUT, OUTPUT):
        assert not result["state_dir"].startswith(path) and not path.startswith(result["state_dir"])


def test_second_claim_agent_starts_where_the_first_left_a_bearer(tmp_path):
    """The failure this prevents: a repeat claim met the first claim's bearer file."""
    from runpod_lifecycle.engine_startup import claim_state_dir
    from runpod_lifecycle.pod_agent import _write_engine_bearer_file
    base = str(tmp_path / "volume" / "agent-state")
    a, b = (claim_state_dir(base)["state_dir"] for _ in range(2))
    assert a != b
    _write_engine_bearer_file(a, TOKEN)
    _write_engine_bearer_file(b, "another-claim-bearer-" + "y" * 32)  # raised when it met the old file
    with pytest.raises(ValueError):  # the same directory with another bearer is the old failure
        _write_engine_bearer_file(a, "another-claim-bearer-" + "y" * 32)


def test_launch_records_the_claim_on_the_pod_and_the_cli_handle(monkeypatch, capsys):
    from runpod_lifecycle.lifecycle import launch
    create = MagicMock(return_value={"id": "pod-1", "claim_id": "0123456789abcdef",
                                     "state_dir": STATE + "/0123456789abcdef"})
    monkeypatch.setattr("runpod_lifecycle.lifecycle.find_gpu_type", lambda g, k: {"id": "gpu-1"})
    monkeypatch.setattr("runpod_lifecycle.lifecycle.create_pod", create)
    pod = asyncio.run(launch(https_config(engine_startup=selection(), storage_volumes=(), storage_name=None),
                             name="claim"))
    assert pod.claim_id == "0123456789abcdef" and pod._engine_state_dir == STATE + "/0123456789abcdef"
    assert cli._claim_record(pod) == {"claim_id": "0123456789abcdef", "state_dir": STATE + "/0123456789abcdef"}
    monkeypatch.setattr("runpod_lifecycle.lifecycle.create_pod", MagicMock(return_value={"id": "pod-2"}))
    plain = asyncio.run(launch(https_config(storage_volumes=(), storage_name=None), name="claim"))
    assert cli._claim_record(plain) == {}


class LocalShell:
    """Stands in for the pod's HTTPS exec route: runs the command on a local directory."""

    def __init__(self):
        self.commands = []

    def execute_command(self, command, timeout=600):
        self.commands.append(command)
        done = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=20)
        return done.returncode, done.stdout, done.stderr

    def disconnect(self):
        pass


def released_pod(monkeypatch, state_dir, shell):
    from runpod_lifecycle.pod import Pod
    pod = Pod("pod-1", "claim", https_config(storage_volumes=(), storage_name=None))
    pod._engine_state_dir = str(state_dir)
    monkeypatch.setattr(Pod, "https_transport", lambda self: shell)
    terminated = MagicMock()
    monkeypatch.setattr("runpod_lifecycle.pod.api.terminate_pod", terminated)
    return pod, terminated


def test_release_removes_the_engine_bearer_token(tmp_path, monkeypatch):
    state = tmp_path / "vol one" / "agent-state" / "0123456789abcdef"  # a space must survive quoting
    state.mkdir(parents=True, mode=0o700)
    token = state / "engine-bearer.token"
    token.write_text(TOKEN)
    keep = state / "engine-readiness.json"
    keep.write_text("{}")
    shell = LocalShell()
    pod, terminated = released_pod(monkeypatch, state, shell)
    asyncio.run(pod.terminate())
    assert not token.exists() and keep.exists()
    assert shell.commands == ["rm -f -- " + shlex.quote(str(token))]
    terminated.assert_called_once()


def test_release_still_terminates_when_the_token_cannot_be_removed(tmp_path, monkeypatch):
    class Broken(LocalShell):
        def execute_command(self, command, timeout=600):
            raise RuntimeError("pod unreachable")

    pod, terminated = released_pod(monkeypatch, tmp_path, Broken())
    asyncio.run(pod.terminate())
    terminated.assert_called_once()


def test_release_without_engine_state_runs_no_command(tmp_path, monkeypatch):
    shell = LocalShell()
    pod, terminated = released_pod(monkeypatch, tmp_path, shell)
    pod._engine_state_dir = None
    asyncio.run(pod.terminate())
    assert shell.commands == [] and terminated.call_count == 1
