"""Fail when Git tracks local coordination, AI, or runtime artifact files."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path, PurePosixPath

MAX_TRACKED_BYTES = 10 * 1024 * 1024

LOCAL_DOCUMENTS = {
    "docs/PROJECT_PLAN.md",
    "docs/ARCHITECTURE.md",
    "docs/BENCHMARKS.md",
    "docs/DECISIONS.md",
    "docs/PROFILING.md",
}
FORBIDDEN_PREFIXES = (
    ".agents/",
    ".codex/",
    ".claude/",
    ".cursor/",
    ".windsurf/",
    ".continue/",
    "docs/superpowers/",
    "chat-history/",
    "data/",
    "artifacts/",
    "runs/",
    "checkpoints/",
    "wandb/",
)
FORBIDDEN_NAMES = {"AGENTS.md", "CLAUDE.md", "GEMINI.md"}
FORBIDDEN_SUFFIXES = (
    ".prompt.md",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".npy",
    ".npz",
    ".jsonl",
    ".log",
)
TEST_ARTIFACT_ALLOWLIST = ("tests/fixtures/", "tests/_snapshots/")


def tracked_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    return [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]


def find_violations(
    root: Path,
    paths: list[str],
    *,
    max_bytes: int = MAX_TRACKED_BYTES,
) -> list[str]:
    violations = []
    for raw_path in paths:
        path = PurePosixPath(raw_path).as_posix()
        allowed_test_artifact = path.startswith(TEST_ARTIFACT_ALLOWLIST)
        reason = None
        if path in LOCAL_DOCUMENTS:
            reason = "local cross-session document"
        elif path.startswith(FORBIDDEN_PREFIXES):
            reason = "local tooling or runtime artifact path"
        elif PurePosixPath(path).name in FORBIDDEN_NAMES:
            reason = "AI agent instruction file"
        elif not allowed_test_artifact and path.endswith(FORBIDDEN_SUFFIXES):
            reason = "runtime data, checkpoint, log, or experiment artifact"
        else:
            file_path = root / Path(path)
            if file_path.is_file() and file_path.stat().st_size > max_bytes:
                reason = f"tracked file exceeds {max_bytes} bytes"
        if reason is not None:
            violations.append(f"{path}: {reason}")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    violations = find_violations(root, tracked_paths(root))
    if violations:
        print("Repository hygiene violations:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
