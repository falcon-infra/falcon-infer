#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check layout/protected source contracts without imports, compilers or devices."""

from __future__ import annotations

# Standard
import argparse
import ast
import fnmatch
import hashlib
import json
import subprocess
import tomllib
import zipfile
from email.parser import BytesParser
from importlib import metadata, util
from importlib.machinery import PathFinder
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project() -> tuple[str, str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return config["name"], config["version"]


def build_contracts():
    """Read setup.py's artifact contract without running setup or a build."""
    spec = util.spec_from_file_location("_ascend_setup_audit", ROOT / "setup.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_installed_versions() -> None:
    """Reject retired plugins and mixed release versions, without uninstalling."""
    versions = json.loads((ROOT / "release-profile.json").read_text())["versions"]
    for name in ("vllm-ascend", "lmcache-ascend", *versions):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        if name not in versions or installed != versions[name]:
            raise RuntimeError(f"Old/conflicting distribution: {name}=={installed}")


def wheel_info(path: Path) -> dict:
    """Validate this project's native wheel identity, resources and build provenance."""
    primary, version = project()
    builder = build_contracts()
    addon = builder.resource_namespace(primary)
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        metas = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metas) != 1 or len(names) != len(set(names)) or wheel.testzip():
            raise ValueError("Invalid wheel metadata or duplicate/corrupt entries")
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("Unsafe wheel member")
        if any(
            name.startswith(
                (
                    "vllm_ascend/",
                    "lmcache_ascend/",
                    "ascend/legacy_patches/",
                    "ascend/legacy_plugin/",
                    "ascend/legacy-p3/",
                )
            )
            for name in names
        ):
            raise ValueError(
                "P4 wheel contains a retired plugin namespace or patch archive"
            )
        meta = BytesParser().parsebytes(wheel.read(metas[0]))
        if meta["Name"].lower() != primary or meta["Version"] != version:
            raise ValueError("Wheel identity does not match this checkout")
        dist_info = metas[0].rsplit("/", 1)[0]
        tags = BytesParser().parsebytes(wheel.read(dist_info + "/WHEEL"))
        if tags["Root-Is-Purelib"] != "false" or tags.get_all("Tag") != [
            "cp311-cp311-linux_aarch64"
        ]:
            raise ValueError("Expected a native cp311-cp311-linux_aarch64 wheel")
        info = json.loads(wheel.read(addon + "/p1_build_info.json"))
        if info.get("install_mode") != "wheel":
            raise ValueError(
                "Only a regular wheel is accepted here, not editable metadata"
            )
        required = builder.required_artifacts(primary, info)
        required.setdefault(primary, []).extend(["__init__.py", "_version.py"])
        required.setdefault(addon, []).extend(
            ["__init__.py", "_version.py", "_build_info.py"]
        )
        for namespace, patterns in required.items():
            for pattern in patterns:
                if not any(
                    fnmatch.fnmatchcase(name, namespace + "/" + pattern)
                    for name in names
                ):
                    raise ValueError(f"Missing wheel resource: {namespace}/{pattern}")
    return {
        "distribution": primary,
        "version": version,
        "wheel": str(path),
        "sha256": digest(path),
        "ABI_tested": False,
    }


def verify(mode: str) -> dict:
    """Check distribution/import paths and native files without loading an NPU."""
    primary, version = project()
    builder = build_contracts()
    addon = builder.resource_namespace(primary)
    check_installed_versions()
    distribution = metadata.distribution(primary)
    if distribution.version != version:
        raise RuntimeError("Unexpected installed distribution version")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    editable = bool(direct.get("dir_info", {}).get("editable"))
    if editable != (mode == "editable"):
        raise RuntimeError(
            "Installed wheel/editable mode does not match the requested mode"
        )
    paths = {}
    for namespace in dict.fromkeys((primary, addon)):
        spec = PathFinder.find_spec(namespace)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Missing namespace: {namespace}")
        directory = Path(spec.origin).absolute().parent
        expected = (
            ROOT / "build"
            if editable
            else Path(distribution.locate_file(namespace)).resolve()
        )
        if (editable and not directory.is_relative_to(expected)) or (
            not editable and directory.resolve() != expected
        ):
            raise RuntimeError(
                f"Import shadowed by another checkout: {namespace}: {directory}"
            )
        paths[namespace] = directory
    info = json.loads((paths[addon] / "p1_build_info.json").read_text())
    expected_mode = "strict-editable" if editable else "wheel"
    if info.get("install_mode") != expected_mode:
        raise RuntimeError("Generated build metadata does not match installed mode")
    for namespace, patterns in builder.required_artifacts(primary, info).items():
        for pattern in patterns:
            if not any(path.is_file() for path in paths[namespace].glob(pattern)):
                raise RuntimeError(
                    f"Missing installed native resource: {namespace}/{pattern}"
                )
    for namespace in dict.fromkeys((primary, addon)):
        if not (paths[namespace] / "_version.py").is_file():
            raise RuntimeError(f"Missing generated version: {namespace}")
    if not (paths[addon] / "_build_info.py").is_file():
        raise RuntimeError("Missing generated Ascend build metadata")
    return {
        "scope": "installation_paths_and_files_not_ABI_or_NPU",
        "distribution": primary,
        "version": version,
        "mode": mode,
        "namespaces": {key: str(value) for key, value in paths.items()},
        "passed": True,
    }


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
        "setup.py",
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
    for retired in ("p1_build.py", "p1_dev.py"):
        if (root / retired).exists():
            errors.append(f"Retired packaging entry must not exist: {retired}")
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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--installed", choices=("editable", "wheel"))
    selection.add_argument(
        "--wheel", type=Path, help="Inspect a regular wheel without installing"
    )
    args = parser.parse_args()
    if args.source_only and (args.installed or args.wheel):
        parser.error("--source-only applies only to the source audit")
    global ROOT
    ROOT = args.root.resolve()
    try:
        if args.installed:
            report = verify(args.installed)
        elif args.wheel:
            report = {"passed": True, **wheel_info(args.wheel)}
        else:
            report = audit(ROOT, source_only=args.source_only)
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        zipfile.BadZipFile,
        metadata.PackageNotFoundError,
    ) as exc:
        report = {"passed": False, "errors": [str(exc)]}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
