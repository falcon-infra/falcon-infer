# Agent Instructions for vLLM

## Current fork: P5 and P6 joint qualification (2026-10-05)

The user authorized freezing native-layout and preparing P5/P6 together.
Work on p5, then p6 inheriting all p5 changes. Keep native-layout-frozen-20261005
and earlier refs immutable. Change release/validation/operations tooling only;
preserve frozen runtime, native implementations, ABI and the 910B3/GLM-5.2 scope.
Use one final intranet functionality/performance campaign on the p6 pair.
Do not claim qualification, switch live services, or retire original checkouts
before evidence and recovery gates pass. No native builds on this source host.
This supersedes earlier working-branch instructions below.

Current installation/deployment instructions are in
`docs/design/baseline-validation.md`. Keep the native-layout command interface
and validated serving arguments. Paired strict editable remains valid for this
baseline comparison; wheel/image release evidence is a separate requirement.
Historical phase notes below are provenance, not current installation steps.

## Current fork: native repository layout (2026-10-04)

Work on `refactor/native-layout` from immutable `p4-frozen-20261004`.
The user approved removing the repository-root `ascend/` donor tree. Integrate
native sources into `csrc/` and `cmake/`, and use one root tests/tools/examples
tree. Preserve native Python owners, algorithms, ABI names, pinned materials
and the P4 910B3/GLM-5.2 scope. Do not change frozen P4 or earlier refs. This
overrides historical branch/plugin instructions below. Host/source checks only;
native builds and 2P2D acceptance remain in the intranet.

## Current fork: P4 profile pruning (2026-10-03)

The user authorized freezing P3 and implementing P4. Work on `p4`; the paired
`p3-frozen-20261003` tags and all earlier phase branches are immutable inputs.
Only GLM-5.2 native text generation and its shared DSA/MTP implementation remain
supported. Remove other vendors/models, multimodal, training, LoRA and pooling.
Keep native NPU runtime, CPU KV storage, P/D, RemoteFill, checkpoint and recovery.
This supersedes the P3/P2 branch restrictions below, not the source-only boundary.

## Current fork: P3 paired LMCache integration

On 2026-09-28 the user authorized starting P3 from the complete P2 code and
validating P2/P3 together in the intranet. Work on `p3`; retain `p2` at
`f1be323571e3ca2aab53992234045dd064d1967f`, and keep `p1`/`main` unchanged.
P3 establishes paired package identity and native LMCache owners. The formal
LMCacheConnectorV1 now owns DSA/checkpoint/event-handoff/RemoteFill lifecycle
delegation without an imported LMCache plugin mutating its class. Keep LMCache
an optional lazy dependency when KV transfer is not configured.
P4 pruning is not part of this batch. The source-only/NPU boundary below remains.

The user authorized preserving P1 and starting P2 on 2026-09-27. Retain `p1`
for later baseline comparisons and repairs.
Existing `main` and the original four repositories are not the P2 work area.
Preserve behavior when moving platform, registration, Runner, graph, and kernel
code, and document each migration batch. P1 acceptance remains incomplete.

On this source workstation, use the available Python tooling for static and
host tests. Do not install torch/CANN or build native artifacts. Record NPU/ABI
validation as pending for the intranet environment. The phase workflow takes
precedence over the generic environment installation instructions below.

> These instructions apply to **all** AI-assisted contributions to `vllm-project/vllm`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

### Environment setup

Use the existing intranet Python 3.11.14/aarch64, CANN 8.5.1 and pinned
torch/torch-npu environment documented in the baseline guide. Do not create a
Python 3.12 runtime, install torch or run an upstream CUDA setup command.
Use available tooling for host-only checks on this workstation.

### Installing dependencies

In a dedicated intranet test container, prepare pinned materials, run
`p1_dev.py doctor`, then use `p1_dev.py editable --isolated-env --output <new-dir>`
for both repositories. Reinstall both after a phase/version change; preserve
live editable build directories. Do not enable `VLLM_USE_PRECOMPILED=1`.
For ordinary wheels use the same helper's build/install subcommands.

### Running tests

The current host subset is explicit; the historical full suite is not a
declaration that removed devices/models are supported. Use the prepared test
dependencies without upgrading the intranet framework environment.

```bash
python -B tools/run_layout_host_checks.py --list
python -B tools/run_layout_host_checks.py
# Run specific test from specific test file
pytest tests/path/to/test.py -v -s -k test_name

# Run all tests in directory
pytest tests/path/to/dir -v -s
```

### Running linters

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files

# Run mypy as it is in CI:
pre-commit run mypy-3.10 --all-files --hook-stage manual
```

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```
