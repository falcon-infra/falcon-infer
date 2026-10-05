# SPDX-License-Identifier: Apache-2.0
"""Host-only packaging contracts; synthetic native files, no compiler or NPU.

Run directly: python -B tests/standalone/test_standard_packaging.py -v
Native build commands in this suite are mocks producing synthetic file fixtures.
"""

from __future__ import annotations

# Standard
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import tomllib
import unittest
import zipfile
from importlib import util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Third Party
from setuptools import Distribution

ROOT = Path(__file__).resolve().parents[2]


def load(path: Path, name: str) -> object:
    """Load a build helper, never the inference framework."""
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = load(ROOT / "setup.py", "setup_contract_test")
AUDIT = load(ROOT / "tools/check_native_layout.py", "install_audit_test")
VERSIONS = json.loads((ROOT / "release-profile.json").read_text())["versions"]


class DevelopmentContracts(unittest.TestCase):
    """Exercise both distribution layouts through their independent helpers."""

    def test_sdist_verifies_root_materials_without_environment_probe(self):
        command = BUILD.AscendSdist(self.dist)
        with (
            patch.object(
                BUILD, "verify_materials", return_value={"files": 1}
            ) as verify,
            patch.object(BUILD.sdist, "run") as run,
            patch.object(
                BUILD,
                "check_environment",
                side_effect=AssertionError("device/build probe"),
            ),
        ):
            command.run()
        verify.assert_called_once_with(self.primary)
        run.assert_called_once()

    def test_sdist_refuses_missing_materials_before_creating_archive(self):
        command = BUILD.AscendSdist(self.dist)
        with (
            patch.object(
                BUILD, "verify_materials", side_effect=RuntimeError("missing material")
            ),
            patch.object(BUILD.sdist, "run") as run,
            self.assertRaisesRegex(RuntimeError, "missing material"),
        ):
            command.run()
        run.assert_not_called()

    def setUp(self) -> None:
        """Create only disposable synthetic source/material/artifact files."""
        temporary = tempfile.TemporaryDirectory(prefix="p1-contract-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.primary, self.version = AUDIT.project()
        self.addon = BUILD.resource_namespace(self.primary)
        for module in (BUILD, AUDIT):
            self.enterContext(patch.object(module, "ROOT", self.root))
        self.enterContext(patch.object(AUDIT, "build_contracts", return_value=BUILD))
        (self.root / "release-profile.json").write_text(
            json.dumps({"versions": VERSIONS})
        )
        self.dist = Distribution({"name": self.primary, "version": self.version})
        self.command = BUILD.AscendBuildExt(self.dist)
        self.command.build_lib = str(self.root / "pip-temporary/lib")
        self.source = self.root / "source"
        self.source.mkdir()
        self.staging = self.root / "retained-native/install"
        self.info = {"cann_version": "8.5.1", "use_hixl": True, "build_mooncake": False}
        (self.root / "pyproject.toml").write_text(
            f'[project]\nname = "{self.primary}"\nversion = "{self.version}"\n'
            '[build-system]\nrequires = ["setuptools>=77.0.3,<81"]\n'
        )
        (self.root / "requirements").mkdir()
        (self.root / "requirements/ascend.txt").write_text("torch==2.9.0\n")

    def populate(self, base: Path, mode: str = "wheel") -> list[str]:
        """Create synthetic native bytes and real generated metadata."""
        paths = []
        for namespace, patterns in BUILD.required_artifacts(
            self.primary, self.info
        ).items():
            for pattern in patterns:
                relative = namespace + "/" + pattern.replace("*", ".fixture")
                path = base / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"SYNTHETIC TEST DATA, NOT AN ELF LIBRARY")
                paths.append(relative)
        for namespace in (self.primary, self.addon):
            path = base / namespace / "__init__.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic package\n")
        BUILD.write_build_metadata(
            base,
            self.primary,
            self.addon,
            self.version,
            {**self.info, "install_mode": mode},
        )
        return paths

    def test_build_names_are_unique_without_creating_directories(self) -> None:
        first, second = BUILD.AscendBuild(self.dist), BUILD.AscendBuild(self.dist)
        self.assertNotEqual(first.build_base, second.build_base)
        self.assertTrue(Path(first.build_base).is_relative_to(self.root / "build"))
        self.assertFalse((self.root / "build").exists())

    def test_editable_only_maps_python_without_environment_probe(self) -> None:
        command = BUILD.AscendBuildPy(self.dist)
        command.editable_mode = True
        with patch.object(BUILD, "check_environment") as check:
            command.run()
        check.assert_not_called()

    def test_editable_enforces_strict_mode(self) -> None:
        command = BUILD.AscendEditableWheel(self.dist)
        with patch.object(BUILD.editable_wheel, "run") as backend:
            command.run()
            self.assertEqual(command.mode, "strict")
            backend.assert_called_once()
        for mode in ("lenient", "compat"):
            command.mode = mode
            with self.assertRaisesRegex(RuntimeError, "strict"):
                command.run()

    def test_editable_maps_all_resources_outside_pip_temporary_tree(self) -> None:
        resources = self.populate(self.staging, "strict-editable")
        self.command.editable_mode = True
        self.command.publish_outputs(self.staging)
        mapping = self.command.get_output_mapping()
        self.assertEqual(set(self.command.get_outputs()), set(mapping))
        expected = (
            set(resources)
            | {
                f"{namespace}/{name}"
                for namespace in {self.primary, self.addon}
                for name in ("__init__.py", "_version.py")
            }
            | {f"{self.addon}/_build_info.py", f"{self.addon}/p1_build_info.json"}
        )
        self.assertEqual(
            {str(Path(path).relative_to(self.command.build_lib)) for path in mapping},
            expected,
        )
        self.assertFalse(Path(self.command.build_lib).exists())
        for output, source in mapping.items():
            self.assertTrue(Path(output).is_relative_to(Path(self.command.build_lib)))
            self.assertTrue(Path(source).is_relative_to(self.staging))
            self.assertTrue(Path(source).is_file())
        self.assertTrue(any("_build_info.py" in path for path in mapping))
        self.assertTrue(any("p1_build_info.json" in path for path in mapping))
        self.assertFalse((self.root / self.primary).exists())
        self.assertFalse((self.root / "ascend" / self.addon).exists())

    def test_wheel_copies_every_native_resource(self) -> None:
        self.populate(self.staging)
        self.command.publish_outputs(self.staging)
        self.assertEqual(self.command.get_output_mapping(), {})
        for output in self.command.get_outputs():
            relative = Path(output).relative_to(self.command.build_lib)
            self.assertEqual(
                Path(output).read_bytes(), (self.staging / relative).read_bytes()
            )

    def test_external_artifact_symlink_is_rejected(self) -> None:
        self.staging.mkdir(parents=True)
        outside = self.root / "outside.so"
        outside.write_bytes(b"not an artifact")
        (self.staging / "escape.so").symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            self.command.publish_outputs(self.staging)

    def test_material_inventory_ignores_git_control_but_not_extra_payload(self) -> None:
        (self.source / "kernel.cpp").write_text("// fixture")
        expected = BUILD.material_inventory(self.source)
        (self.source / ".git").write_text("gitdir: /irrelevant/control")
        self.assertEqual(BUILD.material_inventory(self.source), expected)
        (self.source / "unexpected.o").write_bytes(b"old executable")
        self.assertNotEqual(BUILD.material_inventory(self.source), expected)

    def make_wheel(self, path: Path, *, omit: str = "", mode: str = "wheel") -> None:
        """Write a synthetic ZIP; never invoke a wheel backend."""
        self.populate(self.staging, mode)
        metadata_dir = f"{self.primary}-{self.version}.dist-info"
        with zipfile.ZipFile(path, "w") as wheel:
            for item in self.staging.rglob("*"):
                if item.is_file() and item.name != omit:
                    wheel.write(item, str(item.relative_to(self.staging)))
            wheel.writestr(
                metadata_dir + "/METADATA",
                f"Metadata-Version: 2.4\nName: {self.primary}\n"
                f"Version: {self.version}\n",
            )
            wheel.writestr(
                metadata_dir + "/WHEEL",
                "Root-Is-Purelib: false\nTag: cp311-cp311-linux_aarch64\n",
            )

    def test_wheel_identity_and_resources_are_checked_without_loading(self) -> None:
        wheel = self.root / "fixture.whl"
        self.make_wheel(wheel)
        report = AUDIT.wheel_info(wheel)
        self.assertEqual(report["distribution"], self.primary)
        self.assertFalse(report["ABI_tested"])

    def test_missing_wheel_resource_and_editable_snapshot_are_rejected(self) -> None:
        wheel = self.root / "fixture.whl"
        self.make_wheel(wheel, omit="_build_info.py")
        with self.assertRaisesRegex(ValueError, "Missing wheel resource"):
            AUDIT.wheel_info(wheel)
        self.make_wheel(wheel, mode="strict-editable")
        with self.assertRaisesRegex(ValueError, "regular wheel"):
            AUDIT.wheel_info(wheel)

    def test_native_wheel_rejects_old_namespace_and_patch_archives(self) -> None:
        for member in (
            "vllm_ascend/__init__.py",
            "lmcache_ascend/__init__.py",
            "ascend/legacy-p3/lmcache_ascend/__init__.py",
            "ascend/legacy_patches/worker/patch_eagle.py",
            "ascend/legacy_plugin/platform.py",
        ):
            with self.subTest(member=member):
                wheel = self.root / "fixture.whl"
                self.make_wheel(wheel)
                with zipfile.ZipFile(wheel, "a") as archive:
                    archive.writestr(member, "# must not be installed")
                with self.assertRaisesRegex(ValueError, "retired plugin namespace"):
                    AUDIT.wheel_info(wheel)

    def test_mocked_native_rebuild_uses_fresh_work_and_install_directories(
        self,
    ) -> None:
        # Cover both the device-object relink and private ACLNN source paths.
        for relative, _ in BUILD.MATERIALS.values():
            material = self.root / relative
            (material / "include").mkdir(parents=True)
            (material / "CMakeLists.txt").write_text("# fixture")
        cann = self.root / "sdk"
        ini = cann / "aarch64-linux/data/platform_config/Ascend910B3.ini"
        ini.parent.mkdir(parents=True)
        ini.write_text("[version]\nAIC_version=AscendC-220\n")
        info = {
            **self.info,
            "cann": str(cann),
            "torch_npu_path": "/fixture/npu",
            "torch": {"path": "/fixture/torch", "cmake": "/fixture/cmake", "abi": 1},
        }
        directories = []
        install = None
        selected = None

        def native(command: list[str], **kwargs: object) -> None:
            nonlocal install, selected
            if command[0] == "bash":
                resource = (
                    Path(command[2])
                    / "vllm/_cann_ops_custom/vendors/vllm-ascend"
                    / "op_api/lib/fixture.so"
                )
                resource.parent.mkdir(parents=True)
                resource.write_bytes(b"SYNTHETIC ACLNN")
            if command[:2] == ["cmake", "-S"]:
                prefix = next(
                    value.split("=", 1)[1]
                    for value in command
                    if value.startswith("-DCMAKE_INSTALL_PREFIX=")
                )
                install = Path(prefix).parent
                selected = Path(prefix).name.removesuffix("_ascend")
                directories.append((command[command.index("-B") + 1], selected))
            if command[:2] == ["cmake", "--install"]:
                for namespace, patterns in BUILD.required_artifacts(
                    selected, info
                ).items():
                    for pattern in patterns:
                        path = install / namespace / pattern.replace("*", ".fixture")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b"SYNTHETIC")

        with (
            patch.object(BUILD, "check_environment", side_effect=lambda: dict(info)),
            patch.object(BUILD, "verify_materials", return_value={"files": 1}),
            patch.object(BUILD.platform, "machine", return_value="aarch64"),
            patch.object(BUILD.metadata, "version", return_value=BUILD.TRITON_VERSION),
            patch.object(
                BUILD.subprocess, "check_output", return_value="/fixture/pybind"
            ),
            patch.object(BUILD.subprocess, "run", side_effect=native),
            patch.dict(
                os.environ, {"MAX_JOBS": "1", "BUILD_MOONCAKE": "0"}, clear=True
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for primary in ("lmcache", "vllm"):
                for _ in range(2):
                    command = BUILD.AscendBuildExt(
                        Distribution({"name": primary, "version": VERSIONS[primary]})
                    )
                    command.build_lib = str(self.root / "wheel-lib")
                    command.run()
        self.assertEqual(len({directory for directory, _ in directories}), 4)
        for directory, primary in directories:
            for namespace, patterns in BUILD.required_artifacts(primary, info).items():
                for pattern in patterns:
                    self.assertTrue(
                        list((Path(directory) / "install" / namespace).glob(pattern))
                    )
        for relative, _ in BUILD.MATERIALS.values():
            self.assertFalse((self.root / relative / "build").exists())

    def test_invalid_job_count_fails_before_native_commands(self) -> None:
        with (
            patch.object(BUILD, "check_environment", return_value={}),
            patch.dict(os.environ, {"MAX_JOBS": "0"}),
            patch.object(BUILD.subprocess, "run") as process,
            self.assertRaisesRegex(RuntimeError, "MAX_JOBS"),
        ):
            self.command.run()
        process.assert_not_called()

    def test_native_failure_does_not_publish_or_reuse_old_payload(self) -> None:
        relative, _ = BUILD.MATERIALS["lmcache"]
        material = self.root / relative
        material.mkdir(parents=True)
        (material / "CMakeLists.txt").write_text("# fixture")
        ini = self.root / "sdk/aarch64-linux/data/platform_config/Ascend910B3.ini"
        ini.parent.mkdir(parents=True)
        ini.write_text("[version]\nAIC_version=AscendC-220\n")
        info = {
            **self.info,
            "cann": str(self.root / "sdk"),
            "torch_npu_path": "/fixture/npu",
            "torch": {"path": "/fixture/torch", "cmake": "/fixture/cmake", "abi": 1},
        }
        command = BUILD.AscendBuildExt(
            Distribution({"name": "lmcache", "version": "0.4.3+ascend.p1"})
        )
        command.build_lib = str(self.root / "lib")
        with (
            patch.object(BUILD, "check_environment", return_value=info),
            patch.object(BUILD, "verify_materials", return_value={"files": 1}),
            patch.object(BUILD.platform, "machine", return_value="aarch64"),
            patch.object(
                BUILD.subprocess, "check_output", return_value="/fixture/pybind"
            ),
            patch.object(
                BUILD.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(2, ["cmake"]),
            ),
            patch.dict(
                os.environ, {"MAX_JOBS": "1", "BUILD_MOONCAKE": "0"}, clear=True
            ),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            command.run()
        self.assertEqual(command.native_output_mapping, {})
        self.assertFalse(Path(command.build_lib).exists())
        self.assertEqual(len(list((self.root / "build/p1-native").glob("run-*"))), 1)

    def test_verify_accepts_strict_link_tree_and_rejects_wrong_mode(self) -> None:
        tree = self.root / "build/__editable__.fixture"
        self.populate(tree, "strict-editable")
        distribution = SimpleNamespace(
            version=self.version,
            read_text=lambda _: json.dumps({"dir_info": {"editable": True}}),
        )
        with (
            patch.object(AUDIT, "check_installed_versions"),
            patch.object(AUDIT.metadata, "distribution", return_value=distribution),
            patch.object(
                AUDIT.PathFinder,
                "find_spec",
                side_effect=lambda name: SimpleNamespace(
                    origin=str(tree / name / "__init__.py")
                ),
            ),
        ):
            self.assertTrue(AUDIT.verify("editable")["passed"])
            with self.assertRaisesRegex(RuntimeError, "mode"):
                AUDIT.verify("wheel")

    def test_install_path_verification_rejects_source_shadowing(self) -> None:
        paths = {
            namespace: self.root / "site-packages" / namespace
            for namespace in (self.primary, self.addon)
        }
        self.populate(self.root / "site-packages")
        distribution = SimpleNamespace(
            version=self.version,
            read_text=lambda _: "{}",
            locate_file=lambda name: paths[name],
        )
        with (
            patch.object(AUDIT, "check_installed_versions"),
            patch.object(AUDIT.metadata, "distribution", return_value=distribution),
            patch.object(
                AUDIT.PathFinder,
                "find_spec",
                side_effect=lambda name: SimpleNamespace(
                    origin=str(paths[name] / "__init__.py")
                ),
            ),
        ):
            self.assertTrue(AUDIT.verify("wheel")["passed"])
        with (
            patch.object(AUDIT, "check_installed_versions"),
            patch.object(AUDIT.metadata, "distribution", return_value=distribution),
            patch.object(
                AUDIT.PathFinder,
                "find_spec",
                return_value=SimpleNamespace(
                    origin=str(self.root / self.primary / "__init__.py")
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "shadowed"),
        ):
            AUDIT.verify("wheel")

    def pinned_material(self):
        """Create a real local Git fixture; never contact a remote."""
        relative, _ = BUILD.MATERIALS[self.primary]
        target = self.root / relative
        target.mkdir(parents=True, exist_ok=True)
        (target / "kernel.cpp").write_text("// pinned fixture\n")
        for args in (
            ("init", "-q"),
            ("add", "kernel.cpp"),
            (
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-q",
                "-m",
                "fixture",
            ),
        ):
            subprocess.run(["git", "-C", str(target), *args], check=True)
        commit = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        self.enterContext(
            patch.dict(BUILD.MATERIALS, {self.primary: (relative, commit)})
        )
        return target, commit

    def test_materials_auto_register_only_clean_pin_and_verify_sdist_payload(self):
        target, commit = self.pinned_material()
        report = BUILD.verify_materials(self.primary)
        self.assertEqual(report["commit"], commit)
        self.assertEqual(report["files"], 1)
        shutil.rmtree(target / ".git")  # disposable fixture only
        self.assertEqual(BUILD.verify_materials(self.primary), report)
        (target / "kernel.cpp").write_text("// changed")
        with self.assertRaisesRegex(RuntimeError, "missing or changed"):
            BUILD.verify_materials(self.primary)

    def test_materials_reject_dirty_untracked_ignored_or_wrong_pin(self):
        target, commit = self.pinned_material()
        for relative in ("kernel.cpp", "extra.cpp", ".gitignore"):
            old = (
                (target / relative).read_bytes()
                if (target / relative).exists()
                else None
            )
            (target / relative).write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "clean material"):
                BUILD.verify_materials(self.primary)
            if old is None:
                (target / relative).unlink()
            else:
                (target / relative).write_bytes(old)
        (target / ".git/info/exclude").write_text("ignored.o\n")
        (target / "ignored.o").write_bytes(b"stale native")
        with self.assertRaisesRegex(RuntimeError, "extra or changed"):
            BUILD.verify_materials(self.primary)
        (target / "ignored.o").unlink()
        with (
            patch.dict(
                BUILD.MATERIALS,
                {self.primary: (str(target.relative_to(self.root)), "0" * 40)},
            ),
            self.assertRaisesRegex(RuntimeError, "clean material"),
        ):
            BUILD.verify_materials(self.primary)
        self.assertFalse((self.root / "submodule-materials.json").exists())

    def test_materials_fail_closed_without_git_or_registered_manifest(self):
        relative, _ = BUILD.MATERIALS[self.primary]
        target = self.root / relative
        target.mkdir(parents=True)
        (target / "kernel.cpp").write_text("// unproven")
        with self.assertRaises(RuntimeError):
            BUILD.verify_materials(self.primary)
        self.assertFalse((self.root / "submodule-materials.json").exists())

    def test_install_audit_rejects_old_plugins_and_mixed_versions(self):
        installed = dict(VERSIONS)

        def version(name):
            if name in installed:
                return installed[name]
            raise AUDIT.metadata.PackageNotFoundError(name)

        with patch.object(AUDIT.metadata, "version", side_effect=version):
            AUDIT.check_installed_versions()
            for name in ("vllm", "lmcache", "vllm-ascend", "lmcache-ascend"):
                previous = installed.get(name)
                installed[name] = "old"
                with self.assertRaisesRegex(RuntimeError, "Old/conflicting"):
                    AUDIT.check_installed_versions()
                if previous:
                    installed[name] = previous
                else:
                    del installed[name]

    def test_setup_entry_is_self_contained_and_build_requirements_are_mirrored(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        requirements = [
            line.split("#", 1)[0].strip()
            for line in (ROOT / "requirements/build.txt").read_text().splitlines()
            if line.split("#", 1)[0].strip()
        ]
        self.assertEqual(set(config["build-system"]["requires"]), set(requirements))
        self.assertEqual(
            config["build-system"]["build-backend"], "setuptools.build_meta"
        )
        for retired in ("p1_build.py", "p1_dev.py"):
            self.assertFalse((ROOT / retired).exists())
            self.assertNotIn(retired, (ROOT / "setup.py").read_text())
        if self.primary == "vllm":
            self.assertIn("triton-ascend==" + BUILD.TRITON_VERSION, requirements)

    def test_standard_setup_commands_and_pep660_backend_with_synthetic_native(self):
        """Real setuptools CLI/backend; only native compilation is replaced."""
        target, commit = self.pinned_material()
        package = self.root / self.primary
        package.mkdir()
        (package / "__init__.py").write_text("# synthetic import-free package\n")
        (package / "example.py").write_text("value = 1\n")
        (self.root / "README.md").write_text("Synthetic packaging fixture\n")
        (self.root / "MANIFEST.in").write_text(
            "include setup.py pyproject.toml README.md submodule-materials.json\n"
            "graft requirements\n"
            f"graft {target.relative_to(self.root)}\n"
            f"graft {self.primary}\n"
            "global-exclude .git .git/**\n"
            f"prune {target.relative_to(self.root)}/.git\n"
        )
        # No real framework or native dependency is imported by this fixture.
        script = (ROOT / "setup.py").read_text()
        relative, original_commit = json.loads(
            json.dumps(
                load(ROOT / "setup.py", "setup_original_pin").MATERIALS[self.primary]
            )
        )
        script = script.replace(original_commit, commit)
        patch_native = textwrap.dedent("""
            def _synthetic_native(self):
                info = {"cann_version": "8.5.1", "use_hixl": True,
                        "build_mooncake": False,
                        "install_mode": (
                            "strict-editable" if self.editable_mode else "wheel")}
                primary = self.distribution.get_name()
                verify_materials(primary)
                staging = ROOT / "build" / ("fixture-" + uuid4().hex)
                for namespace, patterns in required_artifacts(primary, info).items():
                    for pattern in patterns:
                        output = staging / namespace / pattern.replace("*", ".fixture")
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_bytes(b"SYNTHETIC NOT ELF")
                write_build_metadata(
                    staging, primary, primary, self.distribution.get_version(), info)
                self.publish_outputs(staging)
            AscendBuildExt.run = _synthetic_native
            check_environment = lambda: {"cann_version": "8.5.1"}
        """)
        script = script.replace(
            'if __name__ == "__main__":', patch_native + '\nif __name__ == "__main__":'
        )
        (self.root / "setup.py").write_text(script)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("PYTHONPATH", None)

        def execute(*args):
            process = subprocess.run(
                [sys.executable, "-B", *args],
                cwd=self.root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(process.returncode, 0, process.stdout[-15000:])
            return process.stdout

        execute("setup.py", "--name")
        self.assertFalse((self.root / "submodule-materials.json").exists())
        self.assertFalse((self.root / "build").exists())
        execute("setup.py", "sdist", "--dist-dir", "sdist")
        archive = next((self.root / "sdist").glob("*.tar.gz"))
        with tarfile.open(archive) as stream:
            names = stream.getnames()
            self.assertTrue(
                any(name.endswith("/submodule-materials.json") for name in names)
            )
            self.assertTrue(any(name.endswith("/kernel.cpp") for name in names))
            self.assertFalse(
                any("/.git/" in name or name.endswith("/.git") for name in names)
            )
            self.assertFalse(
                any(name.endswith(("p1_build.py", "p1_dev.py")) for name in names)
            )
            stream.extractall(self.root / "unpacked", filter="data")
        execute("setup.py", "bdist_wheel", "--dist-dir", "wheels")
        self.assertEqual(len(list((self.root / "wheels").glob("*.whl"))), 1)
        # Backend metadata must not need NPU or a native compilation.
        (self.root / "metadata").mkdir()
        execute(
            "-c",
            "from setuptools import build_meta; "
            "build_meta.prepare_metadata_for_build_wheel('metadata')",
        )
        (self.root / "editable").mkdir()
        execute(
            "-c",
            "from setuptools import build_meta; build_meta.build_editable('editable')",
        )
        self.assertEqual(len(list((self.root / "editable").glob("*.whl"))), 1)
        trees = list((self.root / "build").glob("__editable__.*"))
        self.assertEqual(len(trees), 1)
        tree = trees[0] / self.primary
        self.assertEqual((tree / "example.py").read_text(), "value = 1\n")
        (package / "example.py").write_text("value = 2\n")
        self.assertEqual((tree / "example.py").read_text(), "value = 2\n")
        info = json.loads((tree / "p1_build_info.json").read_text())
        self.assertEqual(info["install_mode"], "strict-editable")
        for namespace, patterns in BUILD.required_artifacts(self.primary, info).items():
            for pattern in patterns:
                self.assertTrue(any((trees[0] / namespace).glob(pattern)), pattern)
        # Building the sdist extraction must not depend on the original Git tree.
        unpacked = next((self.root / "unpacked").iterdir())
        self.assertFalse((unpacked / relative / ".git").exists())
        process = subprocess.run(
            [sys.executable, "-B", "setup.py", "bdist_wheel", "--dist-dir", "rebuilt"],
            cwd=unpacked,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(process.returncode, 0, process.stdout[-15000:])
        self.assertEqual(len(list((unpacked / "rebuilt").glob("*.whl"))), 1)


if __name__ == "__main__":
    unittest.main()
