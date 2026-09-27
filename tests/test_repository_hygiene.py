"""Repository tracked-file hygiene contracts."""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_repository_hygiene.py"


def test_repository_hygiene_script_exists() -> None:
    assert SCRIPT.is_file()


def _write(root: Path, relative: str, contents: bytes = b"test") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)


def test_hygiene_rejects_forced_tracked_prohibited_files_but_allows_fixtures(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    for path in (
        "docs/PROJECT_PLAN.md",
        "AGENTS.md",
        "runs/example/last.pt",
        "metrics.jsonl",
        "src/lmforge/example.py",
        "tests/fixtures/model.pt",
        "tests/_snapshots/expected.npz",
    ):
        _write(tmp_path, path)
    subprocess.run(["git", "add", "-f", "."], cwd=tmp_path, check=True)

    result = subprocess.run(
        [sys.executable, SCRIPT, "--root", tmp_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 1
    assert "docs/PROJECT_PLAN.md" in result.stdout
    assert "AGENTS.md" in result.stdout
    assert "runs/example/last.pt" in result.stdout
    assert "metrics.jsonl" in result.stdout
    assert "tests/fixtures/model.pt" not in result.stdout
    assert "tests/_snapshots/expected.npz" not in result.stdout


def test_hygiene_rejects_oversized_tracked_files(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("repository_hygiene", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _write(tmp_path, "reports/oversized.bin", b"12345")

    violations = module.find_violations(
        tmp_path,
        ["reports/oversized.bin"],
        max_bytes=4,
    )

    assert violations == ["reports/oversized.bin: tracked file exceeds 4 bytes"]


def test_current_repository_passes_hygiene_check() -> None:
    result = subprocess.run(
        [sys.executable, SCRIPT, "--root", ROOT],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
