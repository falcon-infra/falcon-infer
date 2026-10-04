# SPDX-License-Identifier: Apache-2.0
"""Layout, frozen implementation and source manifest contracts without compilers."""

# Standard
import hashlib
import importlib.util
import json
from pathlib import Path

# Third Party
import pytest
from setuptools.command.egg_info import FileList

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "native_layout_gate", ROOT / "tools/check_native_layout.py"
)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def test_real_checkout_has_frozen_implementations_and_no_donor_root() -> None:
    report = GATE.audit(ROOT, source_only=True)
    assert report["passed"], report["errors"]


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    (tmp_path / "docs/design").mkdir(parents=True)
    (tmp_path / "cmake").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "native.py").write_text("value = 1\n")
    (tmp_path / "tests/test_contract.py").write_text("def test_contract(): pass\n")
    for name in (
        "CMakeLists.txt",
        "MANIFEST.in",
        ".gitmodules",
        "p1_build.py",
        "p1_dev.py",
        "cmake/npu_extensions.cmake",
    ):
        (tmp_path / name).write_text("")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="fixture"\nversion="1+layout1"\n'
    )
    manifest = {
        "runtime_files": {
            "native.py": hashlib.sha256(
                (tmp_path / "native.py").read_bytes()
            ).hexdigest()
        },
        "native_files": {},
        "layout_version": "1+layout1",
        "entries": [
            {
                "action": "move",
                "destination": "tests/test_contract.py",
                "mode": "100644",
                "test_contracts": ["test_contract"],
            }
        ],
    }
    (tmp_path / "docs/design/layout-migration.json").write_text(json.dumps(manifest))
    return tmp_path


def test_gate_detects_source_drift(checkout: Path) -> None:
    assert GATE.audit(checkout, source_only=True)["passed"]
    (checkout / "native.py").write_text("value = 2\n")
    assert not GATE.audit(checkout, source_only=True)["passed"]


@pytest.mark.parametrize("linked", [False, True])
def test_gate_rejects_donor_root_even_when_empty_or_linked(
    checkout: Path, linked: bool
) -> None:
    if linked:
        (checkout / "ascend").symlink_to("cmake", target_is_directory=True)
    else:
        (checkout / "ascend").mkdir()
    assert not GATE.audit(checkout, source_only=True)["passed"]


def test_gate_detects_lost_test_contract(checkout: Path) -> None:
    (checkout / "tests/test_contract.py").write_text("# mistakenly lost test\n")
    assert not GATE.audit(checkout, source_only=True)["passed"]


def test_gate_detects_stale_builder_path(checkout: Path) -> None:
    (checkout / "p1_build.py").write_text(
        'manifest = ROOT / "ascend/submodule-materials.json"\n'
    )
    assert not GATE.audit(checkout, source_only=True)["passed"]


def test_imported_test_contracts_are_not_lost_in_colliding_wrappers() -> None:
    source = (
        "from tests.common import test_common\n"
        "from tests.extra import test_extra as test_npu_extra\n"
        "def test_local(): pass\n"
    )
    assert GATE.test_contracts(source) == [
        "test_common",
        "test_local",
        "test_npu_extra",
    ]


@pytest.mark.parametrize("damage", ["changed", "extra", "escape"])
def test_registered_material_gate_detects_drift(checkout: Path, damage: str) -> None:
    directory = checkout / "third_party/pinned"
    directory.mkdir(parents=True)
    (directory / "kernel.h").write_text("pinned material")
    manifest_path = checkout / "docs/design/layout-migration.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"].append(
        {
            "action": "move",
            "destination": "third_party/pinned",
            "mode": "160000",
            "blob": "fixed-material-commit",
        }
    )
    manifest_path.write_text(json.dumps(manifest))
    assert not GATE.audit(checkout)["passed"]
    registered = {
        "path": "third_party/pinned",
        "commit": "fixed-material-commit",
        "files": {
            "kernel.h": {
                "sha256": hashlib.sha256(b"pinned material").hexdigest(),
                "symlink": None,
            }
        },
    }
    (checkout / "submodule-materials.json").write_text(json.dumps(registered))
    assert GATE.audit(checkout)["passed"]
    if damage == "changed":
        (directory / "kernel.h").write_text("changed material")
    elif damage == "extra":
        (directory / "extra.h").write_text("extra material")
    else:
        (directory / "escape.h").symlink_to(checkout / "native.py")
    assert not GATE.audit(checkout)["passed"]


@pytest.mark.parametrize("git_control", [".git", ".git/config", ".git/objects/hash"])
def test_sdist_manifest_keeps_native_inputs_but_not_old_tree_or_git_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git_control: str
) -> None:
    # Exercise the actual setuptools MANIFEST rules, without package builds.
    candidates = (
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "setup.py",
        "p1_build.py",
        "p1_dev.py",
        "CMakeLists.txt",
        ".gitmodules",
        "csrc/kernel.cpp",
        "cmake/npu_extensions.cmake",
        "submodule-materials.json",
        "csrc/third_party/catlass/include/header.h",
        f"csrc/third_party/catlass/{git_control}",
        "third_party/kvcache-ops/kernel.cpp",
        f"third_party/kvcache-ops/{git_control}",
        "docs/design/layout-migration.json",
        "ascend/old.py",
        "build/old.so",
        "dist/old.whl",
        "tests/test_example.py",
        "csrc/build/generated.cpp",
        "csrc/output/generated.cpp",
    )
    for name in candidates:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    monkeypatch.chdir(tmp_path)
    files = FileList()
    files.findall()
    for line in (ROOT / "MANIFEST.in").read_text().splitlines():
        if line.strip():
            files.process_template_line(line)
    names = set(files.files)
    assert {
        "csrc/kernel.cpp",
        "cmake/npu_extensions.cmake",
        "submodule-materials.json",
        "docs/design/layout-migration.json",
    } <= names
    if ROOT.name == "LMCache":
        assert "third_party/kvcache-ops/kernel.cpp" in names
    else:
        assert "csrc/third_party/catlass/include/header.h" in names
    assert not any(
        name.startswith(
            ("ascend/", "build/", "dist/", "tests/", "csrc/build/", "csrc/output/")
        )
        or name.endswith("/.git")
        or "/.git/" in name
        for name in names
    )
