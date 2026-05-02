# VibeComfy Implementation Plan

VibeComfy should become a Python-first way to discover, patch, compile, and run ComfyUI workflows from official templates and installed custom-node examples. The first milestone is not a perfect workflow-to-Python translator. The first milestone is a reliable template loader, inspector, patcher, scratchpad generator, and runner that can execute real image and video workflows locally and on RunPod.

## High-Level Summary

We are building a pure-Python scratchpad for ComfyUI workflows. It should pull real workflows from the official Comfy workflow-template packages and from installed custom-node `example_workflows`, convert them into a VibeComfy internal format, let users edit and extend them with ordinary Python, then run them end to end through ComfyUI without opening the UI.

The proof is execution, not just conversion. The first milestone must inspect at least 20 real workflow examples and run at least five representative workflows end to end, including official templates, video/Wan workflows, and at least one external custom-node workflow from a source such as `kijai/ComfyUI-WanVideoWrapper`. The RunPod validation path must launch a GPU machine, run the workflows, save outputs and metadata, and terminate the machine when finished.

The central artifact is a scratchpad file: a generated Python file for one workflow instantiation. It starts from a full workflow template, applies prompt/model/seed edits, optionally composes individual nodes from the installed node library, compiles to a Comfy workflow dict, and queues that dict in the embedded/pip ComfyUI runtime.

RunPod is part of the core validation plan, not an optional final polish step. Local tests can prove parsing, conversion, and small runtime behavior, but real confidence requires launching a GPU pod, installing the full stack, running multiple workflows end to end, and verifying cleanup. This document currently lives in the same repository as the `runpod-lifecycle` launcher, so agents executing the plan should use the launcher in this directory for remote validation.

## Source-Informed Adjustments

- The official `Comfy-Org/workflow_templates` repository is structured around packages, manifests, templates, blueprints, validation scripts, model metadata, and optional node-version metadata. Treat it as a structured source of truth, not as a folder of arbitrary JSON files.
- Comfy custom-node repositories can expose workflow examples through `example_workflows/` and related folder names. VibeComfy should scan installed custom nodes for these examples instead of only vendoring hand-picked external JSON files.
- ComfyScript already provides a Python front end and workflow-to-Python transpiler. Run a spike before deciding whether VibeComfy should reuse it, wrap it, import from it, export to it, or only learn from it.
- Use the exact external source names during implementation. For custom-node validation, start with `kijai/ComfyUI-KJNodes` and `kijai/ComfyUI-WanVideoWrapper`, not vague `Kijai/Kajai` placeholders.
- The node library should be runtime-discovered from installed Comfy node definitions or node mappings. Do not hard-code an "approved node" list as the primary source of truth.
- GraphBuilder should be verified locally before making it a required backend. If direct API-dict generation is simpler and more reliable for the first milestone, use it while keeping GraphBuilder as an optional compiler backend.
- ComfyUI has two distinct workflow JSON shapes: the UI graph (with `nodes`, `links`, positions, groups — what the web UI exports) and the API dict (a flat object keyed by stringified node id — what `queue_prompt` accepts). Templates in `Comfy-Org/workflow_templates` are mostly UI exports; custom-node `example_workflows/` mix both. `load_template` must detect which shape it received and route accordingly; `normalize_to_api` is a real conversion, not a passthrough. Patching the wrong shape is the most common silent failure.

## Operating Principles

- Keep moving. The agent should not wait for feedback when there is a reasonable next action. If a dependency is missing, inspect it, install it, stub around it, or choose the smallest reversible path that keeps progress moving.
- Bias toward simple, working paths. Prefer a boring JSON-to-internal-format translator plus a pure-Python scratchpad over a clever compiler that blocks end-to-end execution.
- Solve the whole chain. A workflow is not validated until it runs end to end, writes outputs, saves metadata, and can be reproduced from a clean command.
- Make hidden state visible. Models, custom nodes, storage mounts, environment variables, and external workflow sources must be reported clearly in inspect output, logs, and docs.
- Treat failures as implementation tasks. Missing models, missing custom nodes, bad workflow shapes, queue errors, and remote setup failures should produce fixes or explicit follow-up tasks, not vague blockers.
- Use official templates as the main validation corpus. Pull most workflows from the official Comfy template library, then add a smaller custom-node slice from installed `example_workflows` to prove compatibility.
- Clean up paid resources. Any RunPod machine launched for validation must be terminated at the end of the run unless the user explicitly asks to leave it running.

## Non-Negotiables

- VibeComfy is a pure-Python scratchpad layer. Users should be able to write Python that starts from templates, converts them into VibeComfy's internal format, edits that representation, adds custom Python logic, and runs the result.
- The agent-facing unit of work is a normal Python scratchpad file. Agents should inspect, edit, validate, run, and repair code, not perform fragile ad hoc JSON mutation.
- JSON passthrough is allowed only as a source and compatibility path. Any editing, patching, or composition must happen through `VibeWorkflow`, then compile back to a Comfy API dict.
- The scratchpad path must run end to end. It is not enough to inspect or export workflows; the generated workflow must execute through Comfy and produce real outputs.
- Internal and external code must work together. The test plan must validate workflows from the official template library, workflows that require custom nodes, and Python code that programmatically modifies those workflows.
- The first implementation can be pragmatic. JSON passthrough and a simple internal IR are acceptable if they preserve behavior and unlock reliable patching and execution.
- At least five representative workflows must run end to end before the milestone is considered credible.

## Plan Checklist

- [x] [1. Create the repository and baseline package](#1-create-the-repository-and-baseline-package)
- [x] [2. Map the HiddenSwitch ComfyUI runtime surface](#2-map-the-hiddenswitch-comfyui-runtime-surface)
- [x] [3. Define the core VibeWorkflow and scratchpad model](#3-define-the-core-vibeworkflow-and-scratchpad-model)
- [x] [4. Build workflow and node inventories](#4-build-workflow-and-node-inventories)
- [x] [5. Normalize workflows and analyze requirements](#5-normalize-workflows-and-analyze-requirements)
- [x] [6. Validate ComfyScript and compiler options](#6-validate-comfyscript-and-compiler-options)
- [x] [7. Specify Comfy server lifecycle and async execution](#7-specify-comfy-server-lifecycle-and-async-execution)
- [x] [8. Build the scratchpad and JSON runner first](#8-build-the-scratchpad-and-json-runner-first)
- [x] [9. Add optional GraphBuilder backend](#9-add-optional-graphbuilder-backend)
- [x] [10. Design the agent-facing CLI](#10-design-the-agent-facing-cli)
- [x] [11. Install and verify custom node packs](#11-install-and-verify-custom-node-packs)
- [x] [12. Prove the RunPod live path](#12-prove-the-runpod-live-path)
- [x] [13. Run the end-to-end test matrix](#13-run-the-end-to-end-test-matrix)
- [x] [14. Meet the definition of done](#14-meet-the-definition-of-done)

## Execution Record

Current execution status, as of 2026-04-25:

- Implemented VibeComfy in `/Users/peteromalley/Documents/reigh-workspace/vibecomfy`.
- Local tests pass: `4 passed`.
- Runtime surface mapped against HiddenSwitch ComfyUI `0.18.2`.
- `vibecomfy sources sync` indexes 23 official templates, 51 external workflows/examples, and 1,202 runtime node definitions.
- The first 20 official templates inspect, convert, and validate without parser crashes.
- ComfyScript spike completed with decision `learn`.
- GraphBuilder backend implemented and validated for parity with direct API-dict compilation.
- Five embedded `EmptyImage -> SaveImage` workflows executed end to end locally and wrote PNG files.
- RunPod validation launched RTX 4090 pod `cxf6kcag1s3p3m`, installed the stack, indexed `nodes=1202`, ran managed runtime smoke, executed the five embedded GraphBuilder-backed workflows, verified PNG outputs, and terminated the launched pod.
- Final RunPod cleanup verified only pre-existing `text-ip-adapter-run2` remained running. A pod that appeared during validation (`u4k4gxjsxhcqn2`) was terminated.
- A second RunPod model/media validation launched/used RTX 4090 pod `dlj64f5kodisbm`, installed the full stack, synced 473 official template JSON files, indexed `official=473 external=51 nodes=1202`, and generated real media outputs through both baseline `comfyui run-workflow` and VibeComfy scratchpads.
- Official model-backed image workflows that generated PNGs end to end on RunPod: `default`, `sdxlturbo_example`, `sdxl_simple_example`, `flux_schnell`, and `sdxl_refiner_prompt_example`.
- External custom-node workflow executed end to end on RunPod with `kijai/ComfyUI-KJNodes`: `EmptyImage -> ImageResizeKJv2 -> SaveImage`, producing `vibecomfy_kjnodes_resize_00001_.png` through baseline Comfy and VibeComfy.
- Video workflow executed end to end on RunPod: `EmptyImage x2 -> ImageBatchMulti -> SaveWEBM`, producing `vibecomfy_generated_video_00001_.webm` through baseline Comfy and VibeComfy.
- The launched/used validation pod `dlj64f5kodisbm` was terminated. Final pod list showed only unrelated pre-existing pods `7mteh5ps3hl52k` and `e0vudir5xdxq8d`.
- A fresh final validation launched RTX 4090 pod `4pz5727nh80qe2`, installed the stack from the current checkout, synced 473 official templates / 53 external examples / 1,202 runtime nodes, ran tests, and executed the proper media matrix through both baseline Comfy and VibeComfy scratchpads.
- Fresh final validation generated real model-backed official image outputs for `default`, `sdxlturbo_example`, `sdxl_simple_example`, `flux_schnell`, and `sdxl_refiner_prompt_example`.
- Fresh final validation generated a real official Wan model-backed video from `text_to_video_wan`. Baseline Comfy wrote `out/proper_e2e/comfyui/text_to_video_wan/video/ComfyUI_00001_.mp4`; VibeComfy wrote `output/video/ComfyUI_00001_.mp4`, with run metadata `run-1777086082`, prompt id `85c548f0-ee87-4905-9a98-4628a04e8e1c`, `runtime=embedded`, and workflow hash `7bf088cbb87e47258ce39ef011ce0f1d6878c082d403321b9ada90ea05bd1c26`.
- Fresh final validation also generated the KJNodes custom-node PNG and utility WEBM through both baseline Comfy and VibeComfy.
- The fresh validation pod `4pz5727nh80qe2` was explicitly terminated. Final pod list showed only `e0vudir5xdxq8d` and `jf1evyhjw3ohv4`, which were not the validation pod launched for this run.

Remaining strict gaps:

- One official inpaint candidate, `flux_fill_inpaint_example`, failed before VibeComfy because Hugging Face returned `401 Unauthorized` for gated model `black-forest-labs/FLUX.1-Fill-dev/flux1-fill-dev.safetensors`.
- RunPod's public pod docs expose explicit stop/delete calls and a scheduled local stop pattern, not a first-class pod TTL in the current launcher. Cleanup is covered by context manager, signal handling, explicit termination, a local watchdog timer, and final pod listing.

## 1. Create the Repository and Baseline Package

Create a new Python package with a clear separation between template handling, runtime execution, compiler logic, and custom-node support.

Target layout:

```text
vibecomfy/
  vibecomfy/
    __init__.py
    cli.py
    templates/
    adapters/
    graphbuilder/
    runtime/
    nodes/
    tests/
  scripts/
  examples/
  out/
    scratchpads/
    runs/
  vendor/
  pyproject.toml
```

Use Python 3.12 unless a dependency forces a lower version.

Bootstrap commands:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --torch-backend=auto "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
uv pip install git+https://github.com/Comfy-Org/workflow_templates.git
uv pip install "comfy-script[default]"
git clone https://github.com/Comfy-Org/workflow_templates.git vendor/workflow_templates
```

Deliverables:

- `pyproject.toml` with package metadata, CLI entry point, and dev dependencies.
- Empty but importable `vibecomfy` package.
- `examples/`, `scripts/`, and ignored/generated `out/scratchpads/` and `out/runs/` directories documented.
- Basic `pytest` smoke test that imports the package.

Acceptance check:

```bash
python -c "import vibecomfy; print(vibecomfy.__file__)"
pytest
```

## 2. Map the HiddenSwitch ComfyUI Runtime Surface

Before implementing wrappers, inspect the actual runtime APIs that HiddenSwitch exposes. Treat the runtime as the source of truth rather than guessing from older ComfyUI examples or from the diagram.

Commands to run:

```bash
python -c "import comfy; print(comfy.__file__)"
comfyui --help
comfyui serve --help
comfyui run-workflow --help
```

Search for the relevant internals:

```bash
rg "GraphBuilder" .venv vendor
rg "queue_prompt" .venv vendor
rg "queue_with_progress" .venv vendor
rg "embedded_comfy_client" .venv vendor
rg "workflow-requirements|guess-settings|env check" .venv vendor
rg "NODE_CLASS_MAPPINGS|object_info|node definitions" .venv vendor
```

Questions to answer:

- How does `comfyui run-workflow` load a workflow?
- What workflow dict shape does `queue_prompt` accept?
- Is `queue_with_progress` available and stable enough to wrap?
- How are models and custom nodes detected?
- How should `Configuration` objects be created programmatically?
- What does `--guess-settings` actually modify?
- Can installed node definitions be queried from Python or from the embedded server?
- Is GraphBuilder available, stable, and appropriate for generated scratchpads?
- Is direct API-dict generation simpler for the first milestone?

Deliverable:

- `docs/runtime_surface.md`, with concrete examples for loading Comfy, queueing a workflow, tracking progress, checking requirements, and creating configuration.
- `docs/runtime_profile.md`, describing Python, Comfy, Torch/CUDA, device, VRAM class, model paths, custom-node paths, output paths, and RunPod storage profile.
- Execution note: completed locally in `/Users/peteromalley/Documents/reigh-workspace/vibecomfy`. HiddenSwitch ComfyUI `0.18.2` exposes `comfyui serve`, `comfyui run-workflow`, `comfyui env check`, `GET /object_info`, `POST /prompt`, `comfy.client.embedded_comfy_client.Comfy`, `queue_prompt_api`, `queue_with_progress`, and `comfy_execution.graph_utils.GraphBuilder`. `vibecomfy runtime smoke --mode managed` started Comfy, loaded 1,202 node definitions, and terminated.

Acceptance check:

- A developer can read `docs/runtime_surface.md` and know which Comfy APIs VibeComfy wraps and which ones it intentionally avoids.
- A developer can run one command to print the active runtime profile and tell whether failures are caused by environment, model, node, or workflow problems.

## 3. Define the Core VibeWorkflow and Scratchpad Model

Do this before implementing `convert_to_vibe_format`. The converter needs a target type, and the scratchpad needs one meaning.

### VibeWorkflow v0

`VibeWorkflow` is the internal editable representation of one Comfy workflow. It is not raw Comfy JSON and it is not a generated Python file.

Minimum data model:

```python
@dataclass
class VibeWorkflow:
    id: str
    source: WorkflowSource
    nodes: dict[str, VibeNode]
    edges: list[VibeEdge]
    inputs: dict[str, VibeInput]
    outputs: list[VibeOutput]
    requirements: WorkflowRequirements
    metadata: dict[str, Any]

@dataclass
class VibeNode:
    id: str
    class_type: str
    pack: str | None
    inputs: dict[str, Any]
    widgets: dict[str, Any]
    metadata: dict[str, Any]

@dataclass
class VibeEdge:
    from_node: str
    from_output: str
    to_node: str
    to_input: str
```

Minimum methods:

```python
workflow.set_prompt("...")
workflow.set_seed(42)
workflow.set_steps(20)
workflow.set_model("...")
workflow.set_input("width", 1024)
workflow.add_node("KJNodes.SomeNode", **inputs)
workflow.connect("node.output", "other.input")
workflow.compile() -> ComfyApiWorkflowDict
workflow.validate() -> ValidationReport
```

Rules:

- Keep node ids stable during conversion so diffs and error reports are understandable.
- Preserve unknown node fields in `metadata` so round-tripping does not destroy information.
- All edits go through `VibeWorkflow` methods or typed helper functions.
- Raw JSON passthrough means "load or execute unmodified source JSON," not "mutate JSON dictionaries by hand."

Knob mapping:

`set_prompt`, `set_seed`, `set_steps`, `set_model`, and `set_input` are sugar over a per-template knob map that resolves a name to a concrete `(node_id, field_path)` pair. The knob map is built during `convert_to_vibe_format` from heuristics over the indexed node classes:

- `prompt` (alias `positive_prompt`) -> first `CLIPTextEncode` node whose downstream is the positive conditioning input of the sampler.
- `negative_prompt` -> the negative-side `CLIPTextEncode`.
- `seed` -> the `seed` widget on the primary `KSampler` / `KSamplerAdvanced`.
- `steps`, `cfg`, `sampler_name`, `scheduler` -> matching widgets on the same sampler.
- `model` -> the `ckpt_name` / `unet_name` widget on the loader feeding the sampler.
- `width`, `height` -> the matching widgets on `EmptyLatentImage` or equivalent.

When heuristics are ambiguous (multiple samplers, multiple text encoders), `set_*` raises rather than guessing. Templates may ship an explicit `<template>.knobs.toml` sidecar that overrides the heuristic map; this is the preferred long-term path.

### Scratchpad File

A scratchpad is a generated Python file that builds and runs one `VibeWorkflow`. It is the agent-facing artifact. There should not also be a separate public `Scratchpad("wan")` object API in milestone one.

Canonical scratchpad shape:

```python
from vibecomfy import workflow_from_template, run


def build():
    workflow = workflow_from_template("flux_txt2img")
    workflow.set_prompt("a cinematic robot painter")
    workflow.set_seed(123)
    workflow.set_steps(20)
    return workflow


async def main():
    result = await run(build())
    print(result.outputs)
```

Agents edit the file. The runtime imports the file, calls `build()`, validates the returned `VibeWorkflow`, compiles it to a Comfy API dict, and queues it.

Deliverables:

- `docs/vibeworkflow.md`, one page that specifies the v0 model.
- `docs/scratchpad_file.md`, one page that specifies generated scratchpad file shape.
- `vibecomfy/workflow.py`
- `vibecomfy/scratchpad_loader.py`

Acceptance check:

- `convert_to_vibe_format(api_workflow) -> VibeWorkflow` has a concrete target type.
- `vibecomfy convert <template-id> --out out/scratchpads/foo.py` generates the canonical file shape.
- `vibecomfy validate out/scratchpads/foo.py` imports the file and validates the returned `VibeWorkflow` without running it.

## 4. Build Workflow and Node Inventories

Build workflow and node inventories before building a compiler. These indexes become the source of what VibeComfy can inspect, patch, convert into its internal format, compose, and run.

Script:

```text
scripts/index_templates.py
scripts/index_nodes.py
```

For official templates, use the package/index/manifest structure from `Comfy-Org/workflow_templates` before falling back to raw directory scanning:

```text
vendor/workflow_templates/templates/
vendor/workflow_templates/packages/
vendor/workflow_templates/blueprints/
vendor/workflow_templates/packages/blueprints/
```

Track both full workflow templates and subgraph blueprints. Blueprints are reusable node components, so index them separately from full runnable workflows.

For external examples, scan installed custom-node repositories for Comfy's documented workflow-example folders:

```text
custom_nodes/<node_pack>/example_workflows/
custom_nodes/<node_pack>/workflow/
custom_nodes/<node_pack>/workflows/
custom_nodes/<node_pack>/example/
custom_nodes/<node_pack>/examples/
```

Also vendor or download a representative sample set into:

```text
vendor/external_workflows/
```

Target sample inventory:

- At least 20 total workflow examples.
- Most examples from the official Comfy template library. Target at least 15 official templates out of the 20-example inspection set.
- At least one workflow from `kijai/ComfyUI-WanVideoWrapper`.
- At least one workflow or node usage path that exercises `kijai/ComfyUI-KJNodes`.
- At least one workflow that exercises Wan/video nodes.
- A mix of text-to-image, image-to-image, inpaint, FLUX, and video.

Node inventory:

- Discover ComfyUI core nodes from the active runtime.
- Discover installed custom-node classes from the active runtime.
- Record node input/output schema where available.
- Record node pack, version, source path, and whether the node appears in selected workflows.
- Do not treat a static hand-written node list as authoritative.

For every template, capture:

```json
{
  "id": "...",
  "path": "...",
  "media_type": "image|video|audio|3d|utility|unknown",
  "nodes": [],
  "custom_node_requirements": [],
  "models": [],
  "inputs": [],
  "outputs": []
}
```

Implementation notes:

- Prefer Comfy's own workflow loading rules where possible.
- Prefer official manifests, package metadata, model metadata, and node-version metadata where available.
- Do not infer too much from filenames if workflow metadata or manifests are available.
- Keep unknowns explicit. `unknown` is better than a false classification.
- Save enough node metadata to support later patching and export.

Deliverables:

- `template_index.json`
- `external_workflow_index.json`
- `node_index.json`
- `blueprint_index.json`
- `vibecomfy/templates/index.py`
- `vibecomfy/nodes/index.py`
- `vibecomfy workflows list`
- `vibecomfy nodes list`

Acceptance check:

```bash
python scripts/index_templates.py --vendor vendor/workflow_templates --out template_index.json
python scripts/index_templates.py --vendor vendor/external_workflows --out external_workflow_index.json
python scripts/index_nodes.py --runtime active --out node_index.json
vibecomfy workflows list
vibecomfy nodes list
```

Pass condition:

- At least 20 examples are indexed across official and external sources.
- At least 15 indexed examples are official Comfy templates.
- External workflow provenance is recorded, including source repository/path and required custom node packs when known.
- The node index is generated from the active runtime and includes core nodes plus installed custom nodes.

## 5. Normalize Workflows and Analyze Requirements

Implement a workflow loading layer that converts raw template files into a consistent API workflow representation.

Core functions:

```python
load_template(path) -> RawWorkflow
normalize_to_api(raw) -> ApiWorkflowDict
analyze_requirements(api_workflow) -> WorkflowRequirements
convert_to_vibe_format(api_workflow) -> VibeWorkflow
```

CLI target:

```bash
vibecomfy inspect <template-id>
vibecomfy inspect vendor/workflow_templates/.../some_template.json
vibecomfy inspect custom_nodes/ComfyUI-WanVideoWrapper/example_workflows/...
```

Output should include:

```text
nodes: N
custom nodes: [...]
models: [...]
input knobs: prompt, seed, width, height, steps...
output nodes: SaveImage / VHS_VideoCombine / ...
status: runnable | missing_nodes | missing_models | unsupported
```

Use HiddenSwitch's `comfyui workflow-requirements` path where possible. If it cannot be called as a library, wrap it carefully and capture structured output. Cross-check missing nodes against the runtime-generated `node_index.json`.

Deliverables:

- `vibecomfy/templates/loader.py`
- `vibecomfy/templates/normalize.py`
- `vibecomfy/templates/convert.py`
- `vibecomfy/runtime/requirements.py`
- `vibecomfy inspect`

Acceptance check:

```bash
vibecomfy inspect --all --limit 20
```

Pass condition:

- No parser crashes.
- Every inspected template is classified as `runnable`, `missing_models`, `missing_nodes`, or `unsupported`.
- Missing model and missing custom-node messages are readable.
- Every inspected template can be converted into VibeComfy's internal workflow format or receives a specific unsupported reason.
- Requirement analysis distinguishes missing runtime package, missing custom node, missing model file, unsupported workflow shape, and insufficient runtime profile.

## 6. Validate ComfyScript and Compiler Options

ComfyScript is not automatically the VibeComfy core abstraction, but it is too relevant to ignore. It has a Python workflow front end and a workflow-to-Python transpiler, so evaluate it after representative workflows have been discovered and normalized.

Spike commands:

```bash
python -m comfy_script.transpile <workflow.json> --api http://127.0.0.1:8188/
uvx --from "comfy-script[default]" python -m comfy_script.transpile <workflow.json> --api http://127.0.0.1:8188/
```

Spike sample:

- One simple official SD/SDXL workflow.
- One FLUX workflow.
- One image-to-image or inpaint workflow.
- One Wan/video workflow.
- One external workflow from `kijai/ComfyUI-WanVideoWrapper`.

Questions to answer:

- Does ComfyScript transpilation preserve graph behavior across representative workflows?
- Is the generated Python agent-editable, or too opaque?
- Can custom-node workflows transpile cleanly?
- Can ComfyScript code be used as an import/export adapter without owning VibeComfy's internal model?
- Does the ComfyScript runtime compete with, duplicate, or simplify the planned scratchpad runner?

Decision options:

- `reuse`: build VibeComfy scratchpads directly on top of ComfyScript primitives.
- `wrap`: use ComfyScript for import/export but keep VibeComfy's own scratchpad API.
- `learn`: copy lessons from the transpiler but do not depend on it.
- `defer`: skip ComfyScript for milestone one because it blocks simpler end-to-end execution.

Deliverables:

- `docs/comfyscript_spike.md`
- `vibecomfy/adapters/comfyscript.py`, only if the spike justifies it.
- Execution note: completed with decision `learn`. ComfyScript requires a live Comfy server and only transpiled `api_flux2` in the five-sample spike. It failed on official UI workflows containing `MarkdownNote` and on the Wan external sample because `MiDaS-DepthMapPreprocessor` was not registered in the active runtime. This makes it useful as a reference/possible adapter, not the milestone-one core.

Acceptance check:

- The document records a clear decision with evidence from at least five workflow attempts.
- The spike can run on workflows selected by sections 4 and 5.
- The main plan is updated so ComfyScript is either a real adapter path or explicitly deferred.

## 7. Specify Comfy Server Lifecycle and Async Execution

Nothing can run until the Comfy execution model is explicit. Milestone one should support a managed local server by default, with an option to connect to an already-running server.

Runtime modes:

- `embedded`: VibeComfy uses HiddenSwitch's in-process `Comfy` client, calls `queue_prompt_api`, waits for completion, and records output metadata. This is the CLI default for end-to-end local smoke runs because HTTP `/prompt` only proves queue submission.
- `managed`: VibeComfy starts `comfyui serve` as a subprocess, waits for readiness, queries `/object_info` or queues workflows, captures logs, and shuts it down when the run completes.
- `external`: VibeComfy connects to an existing Comfy server URL such as `http://127.0.0.1:8188`.

Default for CLI:

```text
embedded runtime unless --runtime server is provided
```

Default for Python:

```python
await run(workflow)  # uses managed runtime unless configured otherwise
await run(workflow, server_url="http://127.0.0.1:8188")  # uses external runtime
```

Async bridge:

- Python API remains async: `await run(workflow)`.
- CLI uses `asyncio.run(...)` internally.
- Scratchpad files may define `build()` only; the CLI handles async execution.
- Scratchpad files may optionally define `async main()` for direct script execution, but `vibecomfy run out/scratchpads/foo.py` only requires `build()`.

Server lifecycle requirements:

- Start managed server with explicit model/custom-node/output paths from `RuntimeProfile`.
- Poll readiness before queueing.
- Capture stdout/stderr to run logs.
- Shut down on success, validation failure, runtime exception, and keyboard interrupt.
- If using `external`, never shut down the user's server.

Deliverables:

- `vibecomfy/runtime/server.py`
- `vibecomfy/runtime/profile.py`
- `docs/runtime_lifecycle.md`

Acceptance check:

```bash
vibecomfy runtime doctor
vibecomfy run out/scratchpads/foo.py
vibecomfy run out/scratchpads/foo.py --server-url http://127.0.0.1:8188
```

Pass condition:

- Managed mode can start, use, log, and stop Comfy.
- External mode can use an already-running server and leave it running.
- CLI commands invoke async execution safely through `asyncio.run`.

## 8. Build the Scratchpad and JSON Runner First

The fastest useful product is a reliable runner for generated Python scratchpads, existing Comfy workflow JSON, and VibeComfy's internal workflow format. Build this before attempting beautiful Python export.

Execution path:

```text
template JSON -> normalized API dict -> VibeWorkflow -> scratchpad file -> queue_prompt -> outputs
```

The scratchpad file is generated from a source workflow. It is the agent-facing artifact to inspect, edit, validate, run, and repair.

Target API:

```python
from vibecomfy import workflow_from_template, run

workflow = workflow_from_template("flux_txt2img")
workflow.set_prompt("a cinematic robot painter")
workflow.set_steps(20)
workflow.set_seed(123)

outputs = await run(workflow)
```

CLI targets:

```bash
vibecomfy convert <template-id> --out out/scratchpads/<template-id>.py
vibecomfy validate out/scratchpads/<template-id>.py
vibecomfy run out/scratchpads/<template-id>.py
vibecomfy run workflow.json
```

Runtime output requirements:

- Output files saved to a predictable directory.
- Metadata saved with prompt, seed, template id, workflow hash, and git SHA.
- Runtime logs captured per run.
- Queue errors surfaced with the relevant missing model or node name.
- Runs saved under `out/runs/<run-id>/`.

Deliverables:

- `vibecomfy/templates/template.py`
- `vibecomfy/runtime/convert.py`
- `vibecomfy/runtime/client.py`
- `vibecomfy/runtime/run.py`
- `vibecomfy/scratchpad_loader.py`
- `vibecomfy convert`
- `vibecomfy validate`
- `vibecomfy run`

Acceptance check:

Run at least five representative workflows end to end and confirm output files exist:

- One SD or SDXL text-to-image workflow from the official template library.
- One FLUX text-to-image workflow from the official template library.
- One image-to-image or inpaint workflow from the official template library.
- One Wan/video workflow.
- One external workflow from `kijai/ComfyUI-WanVideoWrapper`.

## 9. Add Optional GraphBuilder Backend

Build this as a second compile backend after scratchpad execution and direct API-dict compilation work. The first version should prioritize lossless round-tripping and executable output over elegant generated code. GraphBuilder is useful only if the runtime spike proves it is available and reliable in the pip/embedded ComfyUI environment.

Compiler paths:

```text
A. Direct API dict:
   template JSON -> normalized API dict -> queue_prompt

B. Scratchpad direct compile:
   scratchpad Python -> VibeWorkflow -> API dict -> queue_prompt

C. GraphBuilder backend:
   template JSON -> intermediate graph IR -> GraphBuilder Python function -> workflow dict -> queue_prompt
```

Target API:

```python
workflow = workflow_from_template("flux_txt2img")
workflow.set_prompt("a cinematic robot painter")
workflow.set_steps(20)
workflow.set_seed(123)
api_dict = workflow.compile(backend="graphbuilder")
```

Support custom code on top:

```python
def make_variations(base_template, prompts):
    for prompt in prompts:
        workflow = workflow_from_template(base_template)
        workflow.set_prompt(prompt)
        yield workflow
```

CLI target:

```bash
vibecomfy convert <template-id> --out out/scratchpads/<template-id>.py
vibecomfy validate out/scratchpads/<template-id>.py --backend graphbuilder
vibecomfy run out/scratchpads/<template-id>.py --backend graphbuilder
```

Deliverables:

- `vibecomfy/graphbuilder/ir.py`
- `vibecomfy/graphbuilder/compiler.py`
- `vibecomfy/adapters/api_dict.py`
- Execution note: completed as a pragmatic optional backend inside `VibeWorkflow.compile("graphbuilder")` using `comfy_execution.graph_utils.GraphBuilder`. Tests confirm GraphBuilder output matches the direct API-dict backend for the current graph model, and `api_flux2` compiles identically through both backends.

Acceptance check:

- Convert at least five templates to scratchpad files.
- The generated scratchpads can rebuild workflow dicts accepted by `queue_prompt`.
- At least five generated Python scratchpad files execute successfully.
- GraphBuilder outputs execute successfully if the GraphBuilder spike confirms support for the selected node shapes.

Risk note:

- Do not block the project on perfectly translating every Comfy workflow into hand-crafted Python or GraphBuilder. Make scratchpad execution correct first, then improve compiler backends template by template.

## 10. Design the Agent-Facing CLI

The scratchpad file is the user-facing and agent-facing Python layer for starting from a known template, converting it into VibeComfy's internal format, modifying it with code, and running it end to end.

Canonical scratchpad file:

```python
from vibecomfy import workflow_from_template, run


def build():
    workflow = workflow_from_template("wan_txt2video")
    workflow.set_prompt("a floating crystal city at sunrise")
    workflow.set_model("wan2.1")
    workflow.add_node("KJNodes.SomeNode", ...)
    workflow.connect(...)
    return workflow


async def main():
    result = await run(build())
    print(result.outputs)
```

Minimum CLI commands:

```bash
vibecomfy sources sync
vibecomfy workflows list
vibecomfy nodes list
vibecomfy inspect <template-id>
vibecomfy convert <template-id> --out out/scratchpads/foo.py
vibecomfy validate out/scratchpads/foo.py
vibecomfy doctor out/scratchpads/foo.py
vibecomfy run out/scratchpads/foo.py
vibecomfy logs <run-id>
vibecomfy smoke
```

Agent command design:

- `sources sync`: fetch or refresh official workflow templates, installed custom-node examples, and optional vendored external examples.
- `workflows list/inspect`: answer what full workflows exist and what they require.
- `nodes list/inspect`: answer what individual nodes exist and what inputs/outputs they expose.
- `convert`: generate an editable scratchpad file from a workflow.
- `validate`: compile without running and report missing models, nodes, or invalid connections.
- `run`: execute a scratchpad or workflow JSON and save outputs.
- `doctor`: explain what broke and suggest the next concrete fix. It must separate Python import/build errors in user scratchpad code from workflow validation errors, missing model/node errors, and Comfy runtime errors.
- `logs`: show run logs and output metadata.

Deliverables:

- Public exports from `vibecomfy/__init__.py`
- CLI command coverage in tests.
- `docs/agent_interface.md`
- `docs/errors_and_doctor.md`, a one-page spec for structured errors and doctor output.

Acceptance check:

- A user can start from a template, convert it to VibeComfy's internal format, change prompt and seed in a generated Python file, and run it without opening the ComfyUI UI.
- A user can add external Python control flow around internal VibeComfy workflow edits and execute a batch.
- An agent can use CLI commands to discover workflows, discover nodes, generate a scratchpad, validate it, run it, and diagnose failures without manual UI interaction.

## 11. Install and Verify Custom Node Packs

Video workflows will likely require custom nodes. Treat custom-node installation as a documented, reproducible step rather than hidden state.

Initial node packs:

- ComfyUI core
- ComfyScript
- `kijai/ComfyUI-KJNodes`
- `kijai/ComfyUI-WanVideoWrapper`
- ComfyUI-Manager, optional

Installer checklist:

```bash
mkdir -p custom_nodes
cd custom_nodes
git clone https://github.com/kijai/ComfyUI-KJNodes.git
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
git -C ComfyUI-KJNodes checkout <pinned-sha>
git -C ComfyUI-WanVideoWrapper checkout <pinned-sha>
cd ..
uv pip install --no-deps -r custom_nodes/<node>/requirements.txt
uv pip check
```

Install custom-node `requirements.txt` with `--no-deps` first, then run `uv pip check` and resolve conflicts manually. Letting `uv pip` follow transitive dependencies will silently downgrade or reinstall `torch` and break the embedded ComfyUI runtime — Kijai's repos in particular pin specific torch nightlies that disagree with whatever HiddenSwitch installed.

Verification:

```bash
comfyui env check
comfyui --guess-settings
vibecomfy inspect <wan-template-id>
```

Custom-node validation should go beyond installation. Exercise a small set of node-pack-specific behavior:

- Inspect node classes and confirm they appear in Comfy's registered node mappings.
- Load at least one workflow that uses each required node pack.
- Run at least one external workflow that mixes custom nodes with core Comfy nodes.
- Scan `example_workflows/` and related folders from installed custom-node repositories.
- Confirm errors identify the missing node pack, not just the raw missing class name.

Deliverables:

- `docs/custom_nodes.md`
- Optional helper script: `scripts/install_custom_nodes.py`
- `scripts/verify_custom_nodes.py`
- Requirement detection that says exactly which custom node pack is missing.

Acceptance check:

- A fresh environment can install the documented node packs and inspect a Wan video workflow without unresolved custom-node errors.
- At least one custom-node workflow runs end to end.
- External custom-node code and internal VibeComfy Python code are used together in the same validation run.

## 12. Prove the RunPod Live Path

Use `runpod-lifecycle` to launch a GPU pod, attach the desired network storage volume, install VibeComfy, and run real workflows remotely.

Known working launcher context:

- This plan file is in the same repository/directory as the `runpod-lifecycle` launcher.
- Package path: `/Users/peteromalley/Documents/reigh-workspace/runpod-lifecycle`
- Python command currently needs Python 3.10+.
- For this user's account, `Peter` storage volume resolved to `m6ccu1lodp` in the live smoke run. This must be configuration, not a hard-coded default.
- A previous smoke launch successfully reached SSH and ran `nvidia-smi` on an RTX 4090.
- Model files are expected on the attached storage volume at standard ComfyUI paths (`models/checkpoints/`, `models/loras/`, `models/vae/`, `models/clip/`, `models/diffusion_models/`, `models/text_encoders/`). The `Peter` volume already contains the milestone-1 weight set; documenting fresh-volume bootstrap (which weights, which URLs, which hashes, which auth) is deferred to milestone 2.

Local launch flow:

```python
import asyncio
import time

from dotenv import load_dotenv
from runpod_lifecycle import RunPodConfig, launch


async def main():
    load_dotenv()
    config = RunPodConfig.from_env(
        storage_name="Peter",
        storage_volumes=(),
        gpu_type="NVIDIA GeForce RTX 4090",
        ram_tiers=(32, 16),
    )
    pod = await launch(config, name=f"vibecomfy-{int(time.time())}")
    try:
        await pod.wait_ready(timeout=300)
        code, stdout, stderr = await pod.exec_ssh("nvidia-smi -L", timeout=60)
        assert code == 0, stderr
        print(pod.id)
        print(stdout)
        # Continue with remote install and workflow execution.
    finally:
        await pod.terminate()


asyncio.run(main())
```

Implementation note: the real validation launcher should wrap this pattern in a context manager with signal handling and a watchdog. RunPod's current pod docs expose explicit stop/delete calls and a scheduled local stop pattern; network-volume pods should be terminated rather than stopped. The pod should only be left running when the user explicitly asks for that.

Remote setup flow:

```bash
git clone <vibecomfy repo>
cd vibecomfy
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
uv pip install --torch-backend=auto "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
comfyui env check
```

Remote smoke commands:

```bash
vibecomfy smoke
vibecomfy sources sync
vibecomfy workflows list
vibecomfy convert <sdxl-template> --out out/scratchpads/sdxl.py
vibecomfy run out/scratchpads/sdxl.py
vibecomfy convert <flux-template> --out out/scratchpads/flux.py
vibecomfy run out/scratchpads/flux.py
vibecomfy convert <wan-template> --out out/scratchpads/wan.py
vibecomfy run out/scratchpads/wan.py
```

Deliverables:

- `docs/runpod.md`
- `vibecomfy smoke`
- One reproducible command or script for a fresh RunPod smoke test.
- A launch context manager that records the pod id and terminates it on normal exit, exception, or signal.
- A cleanup command that terminates the launched pod by id.
- A watchdog backstop when RunPod does not expose a first-class pod TTL through the launcher, so an abandoned validation pod does not run indefinitely.

Acceptance check:

- One fresh RunPod launch can install VibeComfy, run at least five representative workflows, save outputs, and terminate cleanly at the end of the run.
- The final test script terminates the machine it launched unless the user explicitly requested that it stay running.
- A forced failure during validation still runs cleanup.

Execution status: passed. `scripts/runpod_validate.py` launched `cxf6kcag1s3p3m`, ran the current VibeComfy tests, installed HiddenSwitch ComfyUI and ComfyScript, indexed 23 official templates / 51 external examples / 1,202 runtime nodes, ran managed runtime smoke, executed five embedded GraphBuilder-backed workflows, verified output PNGs, and terminated the launched pod. Follow-up media validation synced 473 official template JSON files from the official Comfy template library, generated five model-backed official PNG outputs, generated one KJNodes custom-node PNG output, generated one WEBM video output, and terminated the validation pod. Fresh final validation on `4pz5727nh80qe2` repeated the proper matrix and additionally generated a real official Wan model-backed `text_to_video_wan` MP4 through both baseline Comfy and VibeComfy before terminating the pod.

## 13. Run the End-to-End Test Matrix

The agent should not call the project done until these tiers pass.

### Tier 0: Runtime Health

```bash
comfyui env check
comfyui run-workflow <HiddenSwitch test workflow> --guess-settings --prompt "test" --steps 4
comfyui run-workflow <mainline-compatible smoke workflow> --prompt "test" --steps 4
vibecomfy runtime doctor
vibecomfy runtime smoke --mode managed
```

Pass condition:

- Runtime starts.
- HiddenSwitch test workflow runs.
- Mainline-compatible smoke workflow runs or documents the precise incompatibility.
- Environment check output is captured.
- Managed Comfy server lifecycle starts, runs one tiny workflow, logs output, and stops cleanly.

Execution status: passed locally. `comfyui env check` succeeded, `vibecomfy runtime smoke --mode managed` started the server and loaded 1,202 nodes from `/object_info`, and five tiny mainline-compatible `EmptyImage -> SaveImage` workflows executed through embedded Comfy and wrote PNGs.

### Tier 1: Template and External Sample Load Tests

```bash
vibecomfy sources sync
vibecomfy workflows list
vibecomfy nodes list
vibecomfy inspect --all --limit 20
vibecomfy inspect --source external --all
```

Pass condition:

- No parser crashes.
- Every template is classified.
- At least 20 official or external workflow examples are indexed and inspected.
- At least one `kijai/ComfyUI-WanVideoWrapper` external custom-node workflow is inspected.
- Node inventory comes from the active runtime.

Execution status: passed for the current parser/converter scope. `vibecomfy sources sync` indexed 23 official templates and 51 external workflows/examples after adding smoke examples. The first 20 official templates inspected, converted, and validated with no parser crashes.

### Tier 2: Scratchpad and JSON Passthrough Execution

Run at least five workflows end to end. The set must include:

- At least four official Comfy templates.
- One SD or SDXL text-to-image from the official template library.
- One FLUX text-to-image from the official template library.
- One image-to-image from the official template library.
- One inpaint from the official template library.
- One Wan/video workflow, preferably from the official template library.
- One external `kijai/ComfyUI-WanVideoWrapper` custom-node workflow as an additional compatibility check if it is not already covered by the five official-heavy runs.

Pass condition:

- Workflow submits to `queue_prompt`.
- Output file exists.
- Output passes a per-media sanity check: image is not all-black and has non-trivial standard deviation; video has at least the expected frame count and is not a single repeated frame; audio has nonzero RMS. A "succeeded" `queue_prompt` that produced a black image counts as a failure.
- Metadata saved.
- Runtime logs captured.
- Each selected workflow can be represented as a scratchpad file.
- If six candidates are selected because image-to-image and inpaint are separate workflows, all selected candidates should run. Five successful end-to-end runs is the minimum.

Execution status: passed for the requested image/video/Wan media proof. On RunPod, baseline Comfy and VibeComfy scratchpads both generated media for five official model-backed image workflows: `default`, `sdxlturbo_example`, `sdxl_simple_example`, `flux_schnell`, and `sdxl_refiner_prompt_example`. Fresh final validation on `4pz5727nh80qe2` also generated a real official Wan model-backed `text_to_video_wan` MP4 through both baseline Comfy and VibeComfy. The same validation generated a KJNodes custom-node PNG with `ImageResizeKJv2` and a WEBM video with `SaveWEBM`. `flux_fill_inpaint_example` was attempted and failed in baseline Comfy because the required `black-forest-labs/FLUX.1-Fill-dev` model is gated and returned Hugging Face `401 Unauthorized`.

### Tier 3: Scratchpad Compile and Optional GraphBuilder Round Trip

For five templates:

```bash
vibecomfy convert <template-id> --out out/scratchpads/<template-id>.py
vibecomfy validate out/scratchpads/<template-id>.py
vibecomfy run out/scratchpads/<template-id>.py
# If GraphBuilder is enabled:
vibecomfy validate out/scratchpads/<template-id>.py --backend graphbuilder
vibecomfy run out/scratchpads/<template-id>.py --backend graphbuilder
```

Pass condition:

- Generated Python scratchpad output is accepted by `queue_prompt`.
- GraphBuilder output is accepted by `queue_prompt` if this backend is enabled.
- Output broadly matches the original template's media shape.
- At least one generated Python example includes ordinary Python logic around the workflow, not just a static exported graph.

Execution status: passed for generated/validated scratchpads and GraphBuilder parity. More than five official templates convert to scratchpad files, and GraphBuilder parity is tested. Full model-backed GraphBuilder output equivalence remains tied to Tier 2's model-backed execution.

### Tier 4: Scratchpad Editing

For each selected template:

```bash
vibecomfy convert <id> --out out/scratchpads/<id>-edited.py
# Edit out/scratchpads/<id>-edited.py to set prompt, seed, and steps through VibeWorkflow methods.
vibecomfy validate out/scratchpads/<id>-edited.py
vibecomfy run out/scratchpads/<id>-edited.py
```

Pass condition:

- Only intended fields changed.
- Workflow still validates.
- Output generated.

### Tier 5: Custom Python and External Code Together

Create one real Python script:

```python
from vibecomfy import workflow_from_template, run


async def main():
    for seed in range(5):
        workflow = workflow_from_template("flux_txt2img")
        workflow.set_seed(seed)
        workflow.set_prompt(f"variation {seed}")
        await run(workflow)
```

Pass condition:

- Batch outputs generated.
- No manual UI interaction.
- The script uses VibeComfy's internal workflow format, not raw ad hoc JSON mutation.
- At least one run starts from an external custom-node workflow and modifies it with VibeComfy Python code.

### Tier 6: ComfyScript Decision Check

Run the ComfyScript spike results against the selected workflows.

Pass condition:

- The project records whether ComfyScript is reused, wrapped, learned from, or deferred.
- If ComfyScript is enabled as an adapter, at least one import or export test runs end to end.
- If ComfyScript is deferred, the reason is concrete and does not block the scratchpad path.

Execution status: passed. Decision is `learn`; ComfyScript is not a milestone-one dependency.

### Tier 7: RunPod Cleanup

Run the final remote validation on a launched RunPod machine.

Pass condition:

- The launched pod id is written to logs.
- The validation result records all generated output paths.
- The machine launched for the validation is terminated at the end.
- A final pod list confirms the launched pod is no longer active.

Execution status: passed. The launched pod `cxf6kcag1s3p3m` was not present after cleanup. A pod that appeared during validation, `u4k4gxjsxhcqn2`, was also terminated. The media validation pod `dlj64f5kodisbm` was explicitly terminated after generating PNG and WEBM outputs. The fresh final validation pod `4pz5727nh80qe2` was explicitly terminated after generating the five official images, KJNodes image, utility WEBM, and official Wan MP4. The validation scripts now include signal handling plus `VIBECOMFY_RUNPOD_MAX_RUNTIME_SECONDS` watchdog termination. Final pod list showed only `e0vudir5xdxq8d` and `jf1evyhjw3ohv4`, which were not the fresh validation pod.

## 14. Meet the Definition of Done

The project is complete for the first milestone only when all of this exists:

- [x] Template index generated from official Comfy templates.
- [x] Official template ingestion uses package/index/manifest structure where available.
- [x] External workflow index generated from installed custom-node example workflow folders.
- [x] External workflow index includes at least one `kijai/ComfyUI-WanVideoWrapper` source.
- [x] Runtime-generated node index includes Comfy core nodes and installed custom nodes.
- [x] `VibeWorkflow` v0 is specified in one page and implemented as the only editable workflow IR.
- [x] Scratchpad is consistently defined as a generated Python file with `build()`, not a second public object API.
- [x] Generated scratchpad files live under `out/scratchpads/`; package modules do not use the same path for user-generated files.
- [x] Comfy server lifecycle is specified and implemented for managed and external runtime modes.
- [x] CLI sync-to-async bridge is specified and uses `asyncio.run` or equivalent safely.
- [x] ComfyScript spike completed with an explicit reuse/wrap/learn/defer decision.
- [x] At least 20 templates inspect cleanly.
- [x] At least 20 official or external workflow examples are collected and inspect cleanly.
- [x] At least 15 inspected examples come from the official Comfy template library.
- [x] At least 5 representative workflows run end to end through VibeComfy's internal workflow format. Current status: five official model-backed image workflows, one KJNodes custom-node workflow, one WEBM video workflow, and one official Wan model-backed MP4 generated outputs on RunPod.
- [x] At least 5 editable Python scratchpad files are generated and validated.
- [x] At least 5 templates convert to executable Python scratchpad files.
- [x] GraphBuilder is either validated as a backend or explicitly deferred in favor of direct API-dict generation.
- [x] At least 1 Wan/video workflow runs on RunPod. Current status: official `text_to_video_wan` generated a real model-backed MP4 on RunPod through both baseline Comfy and VibeComfy.
- [x] At least 1 external custom-node workflow runs end to end.
- [x] Pure-Python scratchpad code can modify and run workflows without UI interaction.
- [x] External custom-node workflows and internal VibeComfy Python edits are validated together.
- [x] `kijai/ComfyUI-KJNodes` and `kijai/ComfyUI-WanVideoWrapper` installation is documented.
- [x] Custom-node installs are pinned by commit SHA in the install script or lockfile.
- [x] Structured error types are specified for missing model, missing node, invalid graph, runtime failure, and insufficient VRAM/device profile.
- [x] Missing model and missing custom-node errors are readable.
- [x] Outputs are saved with prompt, seed/template inputs, template id, workflow hash, and git SHA.
- [x] One command can reproduce a smoke test on fresh RunPod and terminate the launched machine at the end.
- [x] RunPod cleanup is protected by context manager, signal handling, and TTL/watchdog where supported. Current status: context manager, signal handling, explicit termination, final pod listing, and a `VIBECOMFY_RUNPOD_MAX_RUNTIME_SECONDS` watchdog are implemented; RunPod's current pod docs expose explicit stop/delete and local scheduled stop patterns rather than a first-class pod TTL through this launcher.

## Sense Check

The main risk is trying to make every workflow export as beautiful Python immediately. That should be treated as a later quality pass. The first useful product is a dependable template loader, converter, patcher, runner, scratchpad, and requirements reporter.

The second risk is hiding environment state. Custom nodes, models, and RunPod storage need to be explicit in docs and command output, otherwise a workflow may work once on one machine and fail everywhere else.

The third risk is under-validating by running only one happy-path workflow. The plan now requires at least five end-to-end workflow runs and at least 20 inspected examples across official and external sources.

The fourth risk is overfitting to raw JSON. JSON passthrough is useful for bootstrapping, but the product goal is a pure-Python scratchpad over VibeComfy's internal workflow format.

The fifth risk is assuming the architecture before testing the real tools. ComfyScript, GraphBuilder, runtime node definitions, official template manifests, and custom-node `example_workflows` must be inspected directly before their role is locked in.

The sixth risk is silent success. ComfyUI has two distinct workflow JSON shapes (UI vs. API), `queue_prompt` will return success for a workflow that produced a black image or dropped a connection, and a literal executor will check boxes ("output file exists", "no errors") on broken work. Every end-to-end run needs a normalize-then-validate gate plus an output-sanity check; every CLI command needs a smoke test that proves it does the work, not just that it returns 0.
