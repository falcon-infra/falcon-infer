#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check layout/protected source contracts without imports, compilers or devices."""

from __future__ import annotations

# Standard
import argparse
import ast
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path


def test_contracts(source: str) -> list[str]:
    """Return qualified test names, preserving both colliding test modules."""
    result = []

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.ImportFrom, ast.Import)):
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name.startswith("test_"):
                        result.append(prefix + name)
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name.startswith("test_"):
                result.append(prefix + node.name)

    visit(ast.parse(source).body)
    return sorted(result)


def audit(root: Path, *, source_only: bool = False) -> dict:
    """Audit a checkout or materialized sdist; source_only tolerates absent payloads."""
    manifest = json.loads((root / "docs/design/layout-migration.json").read_text())
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    errors = []
    if (root / "ascend").exists() or (root / "ascend").is_symlink():
        errors.append(
            "Repository-root ascend/ must not exist, including compatibility links"
        )
    for path, digest in manifest["runtime_files"].items():
        target = root / path
        if (
            not target.is_file()
            or hashlib.sha256(target.read_bytes()).hexdigest() != digest
        ):
            errors.append(f"Runtime differs from the frozen P4 input: {path}")
    for path, digest in manifest["native_files"].items():
        target = root / path
        if (
            not target.is_file()
            or hashlib.sha256(target.read_bytes()).hexdigest() != digest
        ):
            errors.append(f"Native implementation differs from frozen P4: {path}")
    missing_materials = []
    for row in manifest["entries"]:
        if row["action"] != "move":
            continue
        dest = root / row["destination"]
        if row["mode"] == "160000":
            if not dest.is_dir() or not any(p.name != ".git" for p in dest.iterdir()):
                missing_materials.append(row["destination"])
            if (root / ".git").exists():
                entry = subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(root),
                        "ls-files",
                        "--stage",
                        "--",
                        row["destination"],
                    ],
                    text=True,
                ).split()
                if entry[:2] != ["160000", row["blob"]]:
                    errors.append(f"Pinned gitlink changed: {row['destination']}")
        elif not dest.is_file():
            # sdists intentionally exclude tests, examples and developer tooling.
            if (root / ".git").exists() or row["destination"].startswith(
                ("csrc/", "cmake/")
            ):
                errors.append(f"Migrated file missing: {row['destination']}")
        else:
            if bool(dest.stat().st_mode & 0o111) != (row["mode"] == "100755"):
                errors.append(f"Executable mode changed: {row['destination']}")
            if "test_contracts" in row:
                actual = test_contracts(dest.read_text())
                missing = set(row["test_contracts"]) - set(actual)
                if missing:
                    errors.append(
                        f"Test contracts lost: {row['destination']}: {sorted(missing)}"
                    )
    if missing_materials and not source_only:
        errors.append(
            "Register pinned materials before building: " + ", ".join(missing_materials)
        )
    if not source_only and not missing_materials:
        for row in manifest["entries"]:
            if row["mode"] != "160000":
                continue
            record = root / "submodule-materials.json"
            if not record.is_file():
                errors.append("Missing root submodule-materials.json")
                continue
            registered = json.loads(record.read_text())
            directory = root / row["destination"]
            files = {}
            for path in sorted(directory.rglob("*")):
                relative = path.relative_to(directory)
                if relative.parts[0] == ".git":
                    continue
                if path.is_symlink():
                    if not path.resolve().is_relative_to(directory.resolve()):
                        errors.append(f"Material symlink escapes its tree: {relative}")
                        continue
                    link = str(path.readlink())
                    digest = hashlib.sha256(link.encode()).hexdigest()
                elif path.is_file():
                    link = None
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                else:
                    continue
                files[str(relative)] = {"sha256": digest, "symlink": link}
            if (
                registered.get("path") != row["destination"]
                or registered.get("commit") != row["blob"]
                or not files
                or files != registered.get("files")
            ):
                errors.append(
                    f"Missing or changed material provenance: {row['destination']}"
                )
    for relative in (
        "CMakeLists.txt",
        "MANIFEST.in",
        ".gitmodules",
        "p1_build.py",
        "p1_dev.py",
        "cmake/npu_extensions.cmake",
    ):
        source = (root / relative).read_text()
        for old in (
            "ascend/csrc/",
            "ascend/cmake/",
            "ascend/third_party/",
            "ascend/submodule-materials.json",
            "add_subdirectory(ascend)",
            'ROOT / "ascend"',
        ):
            if old in source:
                errors.append(f"Stale build path in {relative}: {old}")
    release_path = root / "release-profile.json"
    expected_version = manifest["layout_version"]
    if release_path.is_file():
        release = json.loads(release_path.read_text())
        expected_version = release["versions"][project["name"]]
    if project["version"] != expected_version:
        errors.append("Project version does not identify this layout")
    return {
        "scope": "source_layout_not_native_build_ABI_or_NPU",
        "primary": project["name"],
        "version": project["version"],
        "runtime_files_checked": len(manifest["runtime_files"]),
        "native_files_checked": len(manifest["native_files"]),
        "migration_entries": len(manifest["entries"]),
        "missing_materials": missing_materials,
        "errors": errors,
        "passed": not errors,
    }


def main() -> None:
    """Print machine-readable results; fail closed on layout or source drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Allow uninitialized pinned materials; not a build gate",
    )
    args = parser.parse_args()
    report = audit(args.root.resolve(), source_only=args.source_only)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
