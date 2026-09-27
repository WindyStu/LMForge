"""GitHub Actions CI contract."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def test_ci_workflow_exists() -> None:
    assert WORKFLOW.is_file()


def test_ci_uses_locked_uv_cpu_checks_and_repository_hygiene() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'UV_PYTHON: "3.12"' in workflow
    assert 'CUDA_VISIBLE_DEVICES: ""' in workflow
    assert "astral-sh/setup-uv@" in workflow
    assert "uv sync --locked --dev" in workflow
    assert "scripts/check_repository_hygiene.py" in workflow
    assert "uv run ruff check" in workflow
    assert "uv run ruff format --check" in workflow
    assert "uv run lmforge train --help" in workflow
    assert "uv run pytest -q" in workflow
    assert "curl " not in workflow
    assert "wget " not in workflow
