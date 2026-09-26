# Production Path and Correctness Contract

This document records the Phase 1 P1-04/P1-05 design for keeping LMForge's
assignment compatibility tests on production code paths while preserving the
verified numerical baseline.

## Scope

The change is limited to three known correctness gaps:

1. Assignment BPE tests must exercise the production BPE trainer.
2. Explicit transformer token positions must reach every RoPE attention layer.
3. Test fixture text I/O must use UTF-8 explicitly.

Tokenizer performance work, SDPA, `torch.compile`, training-engine refactors,
and later-phase features are out of scope.

## BPE adapter boundary

`tests.adapters.run_train_bpe` is a compatibility boundary, not a BPE
implementation. It converts the Assignment API arguments and calls
`lmforge.tokenization.train_bpe.train_bpe`. Validation, pre-tokenization,
parallel execution, pair selection, and merging remain owned by the production
trainer. Supported optional arguments are forwarded to that API; unsupported
arguments follow its normal `TypeError` behavior.

Consequently, the Assignment BPE tests and the optimized BPE tests cover the
same production implementation. The separate legacy trainer remains only as a
benchmark baseline and is not an Assignment test path.

## Token-position data flow

`TransformerLM.forward` accepts optional `token_positions`, crops it to the same
context window as the input token IDs, and passes it to every
`TransformerBlock`. Each block passes the positions to its attention module,
which applies the existing RoPE implementation. The default `None` path remains
unchanged and therefore preserves existing numerical snapshots.

## UTF-8 fixture contract

Every text-mode file operation in the Python test suite specifies
`encoding="utf-8"`. Binary file operations remain unchanged. A structural test
checks `open`, `Path.open`, `read_text`, and `write_text` calls so future tests do
not silently reintroduce dependence on the ambient default encoding.

## Verification strategy

Implementation follows test-driven development:

1. Add and observe a failing adapter delegation regression.
2. Add and observe a failing explicit-token-position regression using real
   production model components.
3. Add and observe a failing UTF-8 text-I/O contract test.
4. Make only the production and fixture changes required for those tests.
5. Run targeted tests, then the complete suite under WSL Ubuntu with
   `PYTHONUTF8=1 uv run pytest -q`.

Completion also requires preserving numerical snapshots, tokenizer parity,
checkpoint tests, training workflow tests, ignored local coordination docs, and
the prohibition on tracked AI-assistant files.
