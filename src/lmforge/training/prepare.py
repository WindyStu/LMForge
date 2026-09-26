"""Stream text records to a memory-mappable token .npy file."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def tokenizer_state(tokenizer):
    return {"vocab": tokenizer.vocab, "merges": tokenizer.merges,
            "special_tokens": tokenizer.special_tokens}


def tokenizer_fingerprint(tokenizer):
    data = {"vocab": [(k, v.hex()) for k, v in sorted(tokenizer.vocab.items())],
            "merges": [(a.hex(), b.hex()) for a, b in tokenizer.merges],
            "special_tokens": tokenizer.special_tokens}
    return hashlib.sha256(json.dumps(data, ensure_ascii=True).encode()).hexdigest()


def prepare_tokens(input_path, output_path, tokenizer):
    """Each input line is encoded independently, matching encode_iterable().

    Memory is bounded by the longest line and a 1M-token copy buffer. A temporary
    raw file permits learning the final array size without retaining all tokens.
    No newline normalization or implicit EOS insertion is performed.
    """
    source, destination = Path(input_path), Path(output_path)
    if destination.suffix != '.npy':
        raise ValueError("output must be a .npy file")
    if source.resolve() == destination.resolve():
        raise ValueError("input and output must differ")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        raw = Path(temporary) / 'tokens.bin'
        with source.open(encoding='utf-8', newline='') as text, raw.open('wb') as binary:
            for line in text:
                ids = np.asarray(tokenizer.encode(line), dtype=np.int64)
                binary.write(ids.tobytes())
                count += len(ids)
        if count == 0:
            raise ValueError("input contains no tokens")
        mapped = np.memmap(raw, dtype=np.int64, mode='r', shape=(count,))
        staged = Path(temporary) / 'tokens.npy'
        output = np.lib.format.open_memmap(staged, mode='w+', dtype=np.int64, shape=(count,))
        for start in range(0, count, 1_000_000):
            output[start:start + 1_000_000] = mapped[start:start + 1_000_000]
        output.flush()
        del output, mapped
        os.replace(staged, destination)
    metadata = {"tokens": count, "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
                "source": str(source.resolve()), "encoding": "independent-lines"}
    destination.with_suffix('.meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def main():
    from ..cli import legacy_main

    return legacy_main("prepare")


if __name__ == '__main__':
    raise SystemExit(main())
