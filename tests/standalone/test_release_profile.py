# SPDX-License-Identifier: Apache-2.0
"""Release metadata must not broaden scope or alter frozen implementation hashes."""

# Standard
import json
import tomllib
from pathlib import Path


def test_candidate_pair_and_scope() -> None:
    """Both checkouts advertise one candidate; the historical map stays immutable."""
    root = Path(__file__).resolve().parents[2]
    peer = root.parent / "LMCache"
    release = json.loads((root / "release-profile.json").read_text())
    assert release == json.loads((peer / "release-profile.json").read_text())
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    assert project["version"] == release["versions"][project["name"]]
    assert release["runtime_change"] is False
    assert release["scope"]["c8"] is False
    assert release["scope"]["dsa_two_groups"] is True
    assert release["scope"]["mtp"] is True
    assert release["environment"]["soc"] == "Ascend910B3"
    assert release["status"] == "qualification_pending_intranet"
    assert "include release-profile.json" in (root / "MANIFEST.in").read_text()
    migration = json.loads((root / "docs/design/layout-migration.json").read_text())
    assert migration["layout_version"].endswith("+ascend.layout1")
