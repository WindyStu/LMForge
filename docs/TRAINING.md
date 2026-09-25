# 训练与自回归解码

以下命令在 WSL 的 `assignment1-basics` 仓库根目录运行。测试和示例使用 CPU float32；支持 CUDA float32、float16 和 bfloat16（取决于设备）。

## 1. 准备 tokenizer

使用此前 `save_tokenizer_files()` 保存的 JSON/hex 词表和 merges。若还没有保存文件，可用现有 BPE 实现创建一套演示词表：

```bash
cd /mnt/e/develop/code/CS336/assignment1-basics
uv run python - <<'PY'
from lmforge.tokenization.train_bpe import train_bpe
from lmforge.tokenization.serialization import save_tokenizer_files

vocab, merges = train_bpe(
    "tests/fixtures/tinystories_sample_5M.txt", 1000,
    ["<|endoftext|>"], num_processes=1,
)
save_tokenizer_files(vocab, merges,
    "artifacts/demo-tokenizer/vocab.json",
    "artifacts/demo-tokenizer/merges.json")
PY
```

此命令会写入指定的两个文件，已有 tokenizer 时直接复用你的路径。`vocab_size` 必须等于最终实际词表大小（含 special tokens），样本过小时不一定能学满目标词表。

## 2. 文本转换为 token 数组

```bash
uv run python -m lmforge.training.prepare \
  --input tests/fixtures/tinystories_sample_5M.txt \
  --output artifacts/demo-data/train.npy \
  --vocab artifacts/demo-tokenizer/vocab.json \
  --merges artifacts/demo-tokenizer/merges.json \
  --special-token '<|endoftext|>'

uv run python -m lmforge.training.prepare \
  --input tests/fixtures/tinystories_sample.txt \
  --output artifacts/demo-data/valid.npy \
  --vocab artifacts/demo-tokenizer/vocab.json \
  --merges artifacts/demo-tokenizer/merges.json \
  --special-token '<|endoftext|>'
```

这两份 fixture 用于验证流程；不要把它们当成已经验证互不重叠的正式 train/validation split。实验报告应换成独立的验证集，BPE 也只能在训练集上学习。

输出为一维 int64 `.npy`，训练通过 `np.load(..., mmap_mode='r')` 随机读取窗口。每份数据必须至少有 `context_length + 1` 个 token。伴随的 `.meta.json` 保存 tokenizer 指纹，CLI 会检查词表/merge 是否一致。

准备阶段逐行编码，沿用 `encode_iterable()` 的独立记录语义；不规范化换行、不自动添加 EOS。内存主要取决于最长一行及复制缓冲区，硬盘临时需要保存 raw tokens 和 `.npy` 两份数据。特殊 token 应完整地位于同一行。输出已经存在时会报错，避免误覆盖预处理数据。

## 3. 训练

提供的 `configs/tinystories/small.json` 是 2 层、128 hidden、128 context 的小模型配置。有效 batch size 为 `batch_size × grad_accum_steps`，默认 `2 × 8 = 16` 个窗口。

先进行 5 步 CPU 流程检查：

```bash
uv run python -m lmforge.training.train \
  --config configs/tinystories/small.json \
  --train-data artifacts/demo-data/train.npy \
  --val-data artifacts/demo-data/valid.npy \
  --output-dir artifacts/demo-run \
  --vocab artifacts/demo-tokenizer/vocab.json \
  --merges artifacts/demo-tokenizer/merges.json \
  --special-token '<|endoftext|>' \
  --device cpu --max-steps 5
```

训练迭代执行：

```text
采样 inputs / 右移一位的 targets
→ forward → 将 [batch, seq, vocab] 展平后计算交叉熵
→ loss / grad_accum_steps → backward（每个 microbatch）
→ AMP unscale → 全局梯度裁剪 → AdamW step → 清空下一轮梯度
```

- AdamW、交叉熵和 cosine schedule 使用作业自己的实现。
- 学习率按零基 step 调用已有调度函数。warmup > 0 时第一步学习率为 0；5 步测试只是检查流程。
- 参数保持 float32；`precision` 控制 autocast。CUDA fp16 使用 GradScaler；溢出时跳过更新并在日志标记。`max_steps` 计数包括这种尝试。
- 验证使用独立采样器、eval 模式和 no-grad，不消耗训练采样器的随机序列。每 `eval_interval` 步执行，默认 100；短跑不到间隔时不会验证。
- 完整 checkpoint 保存模型、optimizer、scaler、训练步数、随机状态、数据指纹、模型配置和 tokenizer。
- `last.pt` 按保存间隔及最终步保存；`best.pt` 只在验证 loss 创新低时保存。文件通过临时文件后替换的方式写入。
- `metrics.jsonl` 每步记录 loss、lr、训练耗时和 tokens/s；训练耗时不含验证和保存。`config.json` 保存运行配置。
- 新训练若输出目录已有 checkpoint 或日志会报错；请续训或使用新目录。

## 4. 断点恢复

从上述 5 步接着训练到总计 1000 步：

```bash
uv run python -m lmforge.training.train \
  --config configs/tinystories/small.json \
  --train-data artifacts/demo-data/train.npy \
  --val-data artifacts/demo-data/valid.npy \
  --output-dir artifacts/demo-run \
  --resume artifacts/demo-run/last.pt \
  --device cpu --max-steps 1000
```

tokenizer 从 checkpoint 恢复，无需重复传文件。`max_steps` 是训练总步数，不是追加步数。允许更改总步数、device、日志及保存间隔；其余训练参数、tokenizer 和数据需与 checkpoint 一致。`lr_decay_steps` 固定调度时间轴，不能为了续训重设 warmup。

CPU float32 测试验证了连续 4 步与 2 步后恢复到 4 步得到完全相同参数。CUDA 受内核确定性等因素影响，不承诺跨设备逐位相同。

如果要从更早 checkpoint 分叉，使用新输出目录，避免和新于该 checkpoint 的日志混在一起。

## 5. 模型 decode / 生成文本

```bash
uv run python -m lmforge.training.generate \
  --checkpoint artifacts/demo-run/last.pt \
  --prompt 'Once upon a time' \
  --max-new-tokens 128 \
  --temperature 0.8 --top-p 0.9 --seed 42 --device cpu
```

`temperature=0` 使用 greedy argmax；正数时使用 temperature + top-p 采样。top-p 保留累计概率首次达到阈值的最小前缀，包括跨越阈值的 token。默认遇到 `<|endoftext|>` 停止，文本输出隐藏生成末尾的 EOS，但保留 prompt。

每步只输入最近 `context_length` 个 token，并将该窗口的位置从 0 重新编号，避免 RoPE 表越界。当前一次生成一条序列，使用完整窗口重算，没有 KV cache。进入 inference/eval 模式后会恢复调用前模式。`generate()` 返回包含 prompt/EOS 的一维 token ID Tensor，`generate_text()` 返回文本；CLI 自动加载 checkpoint 中的配置和 tokenizer。

tokenizer 的 `decode()` 本来已有实现：先拼接所有 token bytes，再统一 UTF-8 解码。已增加中文、多字节字符、不合法 UTF-8、空输入及未知 ID 的测试。只训练几步后生成文本可能乱码，这是训练量不足；该短跑不代表语言模型质量实验。

## 6. 3050 Laptop / WSL

先确认 WSL 中的 Python 可使用 CUDA：

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

返回 True 后可在上述命令使用 `--device cuda`。需要 fp16 时，在**新训练开始前**把配置中的 `precision` 改为 `float16`。显存不足时先减小 `batch_size` 或 `context_length`，可增加 `grad_accum_steps` 保持有效 batch size。提供的小配置用于入门训练，未在你的显卡上测量峰值显存。

## 7. 测试

从仓库根目录运行：

```bash
uv run pytest tests/test_workflow.py -q
uv run pytest tests/test_model.py tests/test_nn_utils.py tests/test_data.py tests/test_optimizer.py tests/test_serialization.py -q
```

新测试包含实际 Python 子进程中的 prepare → train → resume → generate 命令行流程、精确 CPU 断点恢复、梯度累积等价性、EOS、滑动窗口和 nucleus 采样；CUDA fp16 测试在没有 CUDA 时自动跳过。
