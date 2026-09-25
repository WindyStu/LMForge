# LMForge Packaging Contract

## Purpose

LMForge uses an installable `src` layout. Production Python code lives under
`src/lmforge` and consumers import it through the `lmforge` namespace.

This migration changes package paths and build metadata only. It must not
change model mathematics, tokenizer behavior, checkpoint formats, training
semantics, or numerical snapshots.

## Repository layout

```text
src/lmforge/
├── __init__.py
├── heap.py
├── pretokenization_example.py
├── nn/
├── tokenization/
└── training/

configs/tinystories/
docs/
reports/
scripts/
tests/
```

The `cs336_basics` import namespace is removed without a compatibility shim.
The root-level tokenizer re-export is also removed; callers use
`lmforge.tokenization.tokenizer` directly.

Package-internal imports are relative. Tests, module entry points, and other
external consumers use absolute `lmforge.*` imports.

## Non-production assets

- Package-local tests move to the root `tests/` directory.
- Existing benchmark and profiling reports move to root `reports/`.
- The TinyStories smoke configuration moves to
  `configs/tinystories/small.json`.
- The training guide moves to `docs/TRAINING.md`.
- The WSL profiling shell entry point moves to root `scripts/`.

The migration does not split the training engine or reorganize BPE trainer
implementations. Those changes belong to later Phase 1 tasks.

## Build configuration

The project distribution and import package are both named `lmforge`.
`pyproject.toml` configures `uv_build` with module root `src` and module name
`lmforge`. The lockfile is regenerated with uv under WSL Ubuntu.

## Correctness contract

The migration follows a test-first sequence:

1. Add a package test that initially fails because `lmforge` is unavailable.
2. Require the installed distribution and imported module to be named
   `lmforge` and loaded from `src/lmforge`.
3. Require the removed `cs336_basics` namespace to be undiscoverable.
4. Migrate paths and imports without changing implementation bodies.
5. Run focused model snapshot, tokenizer parity, checkpoint, and training
   workflow tests before the full suite.

Numerical snapshots must never be regenerated to make this migration pass.
Any parity difference is a migration defect.

## Acceptance checks

All Python and test commands run in the WSL `Ubuntu` distribution through uv.
The migration is complete only when:

- `uv sync` succeeds;
- `uv run python -c "import lmforge"` succeeds;
- tests import the installed `lmforge` distribution;
- production Python files contain no `cs336_basics` import;
- the old `cs336_basics` directory is absent;
- model snapshots, tokenizer parity, checkpoint tests, and training workflow
  tests pass;
- the full suite has no failures and does not regress below the P1-02
  baseline of 66 ordinary passing tests.

The inherited non-strict tokenizer memory xfail may report XFAIL or XPASS; its
actual result is recorded without changing the test.
