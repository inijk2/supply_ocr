from pathlib import Path

from pipeline import _repo_root


def test_repo_root_contains_assets():
    root = _repo_root()
    assert (root / "a" / "supply_frame.png").exists()
