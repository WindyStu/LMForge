#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=0

# Run this script from the assignment repository root inside WSL.
corpus_path="tests/fixtures/tinystories_sample_5M.txt"
report_dir="cs336_basics/reports"
profile_dir="${report_dir}/profiles"

mkdir -p "${profile_dir}"

{
  date --iso-8601=seconds
  uname -a
  uv run python --version
  lscpu
  free -h
} > "${report_dir}/wsl_environment.txt"

uv run python -m cs336_basics.tokenization.benchmark_bpe \
  --input "${corpus_path}" \
  --vocab-size 1000 \
  --special-token '<|endoftext|>' \
  --process-counts 1,4 \
  --repeat 3 \
  --output "${report_dir}/bpe_5mb_wsl.json"

for implementation in baseline optimized; do
  uv run python -m cProfile \
    -o "${profile_dir}/${implementation}_p1.prof" \
    -m cs336_basics.tokenization.benchmark_bpe \
    --worker \
    --implementation "${implementation}" \
    --input "${corpus_path}" \
    --vocab-size 1000 \
    --special-token '<|endoftext|>' \
    --num-processes 1

  uv run python -c \
    "import pstats; pstats.Stats('${profile_dir}/${implementation}_p1.prof').strip_dirs().sort_stats('cumtime').print_stats(40)" \
    > "${profile_dir}/${implementation}_p1_cumtime.txt"

  uv run scalene \
    run \
    -o "${profile_dir}/${implementation}_p1_scalene.json" \
    cs336_basics/tokenization/benchmark_bpe.py \
    --- \
    --worker \
    --implementation "${implementation}" \
    --input "${corpus_path}" \
    --vocab-size 1000 \
    --special-token '<|endoftext|>' \
    --num-processes 1

  uv run scalene view \
    --cli \
    -r \
    "${profile_dir}/${implementation}_p1_scalene.json" \
    > "${profile_dir}/${implementation}_p1_scalene.txt"
done

echo "Benchmark: ${report_dir}/bpe_5mb_wsl.json"
echo "Profiles:  ${profile_dir}"
