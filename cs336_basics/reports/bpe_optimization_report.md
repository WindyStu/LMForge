# CS336 BPE 训练优化报告

> 状态：实现、正确性验证、正式 WSL 三次基准和热点分析已完成。Scalene 2.3 在本次 WSL 运行中未产生结果文件，内存结论采用独立子进程的进程树峰值 RSS；CPU 结论采用 cProfile 与训练器阶段计时交叉验证。

## 1. 基准定义

- 数据：`tests/fixtures/tinystories_sample_5M.txt`（5,242,880 bytes）
- 词表大小：1,000
- 特殊 token：`<|endoftext|>`
- 进程数：1、4
- 每组独立运行：3 次，报告中位数
- 正确性门槛：baseline 与 optimized 的完整 vocab/merges SHA-256 必须一致
- 时间：独立子进程端到端 wall time
- 内存：采样根进程及全部递归子进程的 RSS 总和峰值
- 环境：WSL、Python、CPU 和内存信息写入 `reports/wsl_environment.txt`

## 2. 原实现的瓶颈

### 2.1 数据预处理与多进程通信

原实现由父进程读取并解码所有 chunk，再把大字符串发送给 worker；worker 返回 `list[bytes]`。重复 pre-token 会重复占用列表槽位、bytes 对象以及 pickle/IPC 负载，父进程随后又物化完整 `corpus`。

Profiler 重点观察：`_legacy_pre_token`、multiprocessing pickle/管道等待、列表构造，以及 `worker_result_items`。

### 2.2 Pair 初始化

原实现遍历完整 `corpus`，即使同一 pre-token 出现很多次，也重复切分和统计相邻 Pair。内存同时保留完整 corpus、`word_count`、`word_splits` 和 Pair 反向索引。

### 2.3 Pair 更新

原实现已经有 Pair→word 的反向索引，但数据从重复 corpus 建立，更新通过可变列表和邻接分支逐项修补，状态较难验证；惰性堆条目也没有上界，长训练可能积累大量失效节点。

## 3. 优化实现

### 3.1 Counter 化的预分词结果

worker 直接读取自己的文件字节区间，并返回 `Counter[pretoken_bytes, frequency]`。父进程使用 `imap_unordered` 到达一个合并一个，不保留所有 worker 结果。

优化目标：把通信和父进程语料表示从“总 token 数量”降为“各 chunk 的唯一 token 数量”。

### 3.2 唯一 word + 频率

每个不同 pre-token 只建立一次 `word_splits`，出现次数单独保存在 `word_frequency`。Pair 全局计数满足：

```text
global_pair_count[pair]
  = Σ(pair_in_word_count[pair][word] × word_frequency[word])
```

其中 `pair_in_word_count` 保存某 Pair 在某唯一 word 当前切分中的局部出现次数，不能只保存布尔 membership。

### 3.3 局部差分更新

选出 Pair 后，只访问反向索引中的受影响 word。对每个 word 分别计算 merge 前后的局部邻接 Pair Counter，并用差值乘 word frequency 更新全局计数；新局部计数为零时才从反向索引移除该 word。

这样同时处理了以下边界：

- 一个 Pair 在同一 word 中出现多次；
- merge 拆掉一个旧邻接关系，但同一 Pair 在 word 的其他位置仍存在；
- `(a, a)` 等重叠候选必须按非重叠规则合并。

### 3.4 惰性最大堆与有界重建

堆按 `(count, pair)` 取最大值，从而保留 CS336 要求的确定性 tie-breaking。旧条目弹出时与当前全局计数核对；失效则丢弃。当堆超过 `max(50_000, 4 × live_pair_types)` 时按当前有效计数重建，避免惰性条目无界增长。

## 4. 正确性验证

- 小语料与每轮全量扫描的慢速参考实现逐项相等；
- 1 进程与 2 进程输出相等；
- benchmark 冻结 baseline 与 optimized 输出相等；
- `corpus.en`、vocab=500 上两套实现的 vocab 与 merges 完全相等；
- 二进制词表通过 JSON/hex 往返，覆盖 `0x00`、`0xff` 等非 UTF-8 token；
- `BPE_tokenizer.from_files()` 加载后可 encode/decode，并支持新增 special token。

## 5. 正式 WSL 结果

| 进程 | baseline 中位时间 | optimized 中位时间 | 时间提升 | baseline 峰值 RSS | optimized 峰值 RSS | 内存下降 | worker 返回项下降 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.874 s | 1.045 s | 63.64% | 127.30 MiB | 41.91 MiB | 67.08% | 99.36% |
| 4 | 2.684 s | 0.761 s | 71.66% | 273.52 MiB | 121.92 MiB | 55.43% | 98.36% |

环境：WSL2 Linux 6.18.33.2、Python 3.12.13、Intel Core i7-12700H、20 个逻辑 CPU、WSL 可用内存约 7.6 GiB。每个单元格是三个独立子进程运行的中位数。

所有 12 次运行均生成 1,000 个 vocab entry 和 743 个 merge，artifact SHA-256 均为 `adfb849561a427fe2f1ec55e4adc4444441ee31076f0ab5ee998c4be517cd8f8`。

百分比定义：

```text
时间提升% = (baseline_time - optimized_time) / baseline_time × 100
内存下降% = (baseline_peak_rss - optimized_peak_rss) / baseline_peak_rss × 100
通信项下降% = (baseline_items - optimized_items) / baseline_items × 100
```

若时间提升为负数，应原样报告为性能回退，不对数字取绝对值。

### Windows 单次 smoke test（仅验证趋势，不作为正式结论）

| 进程 | baseline 时间 | optimized 时间 | 时间提升 | baseline 峰值 RSS | optimized 峰值 RSS | 内存下降 | worker 返回项下降 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.701 s | 1.482 s | 68.47% | 128.5 MiB | 51.2 MiB | 60.13% | 99.36% |
| 4 | 4.444 s | 0.998 s | 77.54% | 281.9 MiB | 176.0 MiB | 37.57% | 98.36% |

两组输出的 artifact SHA-256 均相同。这里只运行了一次，且平台是 Windows 11 / Python 3.13，因此不能替代三次 WSL 中位数。

## 6. Profiler 与组件级结果

### 6.1 cProfile 定位

WSL baseline cProfile 共记录约 550.8 万次函数调用、总采样时间 4.260 秒；`train_bpe_legacy` 累计 3.908 秒，其中 `_legacy_pre_token` 累计 0.972 秒。堆相关操作也很显著：41,038 次 pop、77,606 次 push，并产生 40,295 次失效条目弹出。

优化版的 Windows cProfile 预检查与正式 WSL 阶段计时方向一致：最大热点已经从“完整 corpus 的 Pair 初始化”转移到 `count_pretokens_in_text`、正则匹配结果编码和 `Counter` 构建。Profiler 会显著改变绝对耗时，因此这里只用它识别函数热点，不用 profiler 时间计算性能提升。

### 6.2 阶段耗时中位数

| 进程 | 阶段 | baseline | optimized | 变化 |
|---:|---|---:|---:|---:|
| 1 | 预分词 | 0.514 s | 0.565 s | -9.88%（回退） |
| 1 | Pair 初始化 | 1.845 s | 0.023 s | +98.78% |
| 1 | Pair merge | 0.213 s | 0.195 s | +8.87% |
| 4 | 预分词 | 0.334 s | 0.279 s | +16.53% |
| 4 | Pair 初始化 | 1.829 s | 0.024 s | +98.69% |
| 4 | Pair merge | 0.204 s | 0.196 s | +3.91% |

最主要的收益不是 merge 循环本身，而是只为 8,126 个唯一 pre-token 建立切分和 Pair，而不是为 1,263,131 个出现实例重复初始化。单进程预分词慢约 9.88%，原因是 Counter 哈希计数比直接 append list 多做了工作；但它换来了后续初始化和内存的数量级下降。四进程时，因为 IPC 负载变小，预分词阶段反而快 16.53%。

### 6.3 Pair 更新与堆

| 指标 | baseline | optimized | 减少 |
|---|---:|---:|---:|
| affected word updates | 33,141 | 28,249 | 14.76% |
| heap pushes | 78,402 | 15,561 | 80.15% |
| stale heap pops | 40,295 | 2,925 | 92.74% |

局部 Counter 差分不仅更容易维持计数不变量，也避免同一个 Pair 更新过程中反复向堆压入中间状态。当前 743 次 merge 没有触发堆重建，说明有界重建是大规模训练的保护机制，而不是本次 5MB 结果的主要收益来源。

### 6.4 多进程通信与内存

单进程 worker 返回项由 1,263,131 降到 8,126，减少 99.36%；四进程按 chunk 分别去重后合计返回 20,678 项，减少 98.36%。这直接解释了四进程预分词加速以及父进程常驻数据下降。

从 1 进程切换到 4 进程，baseline 仅获得约 1.07 倍端到端加速，optimized 获得约 1.37 倍；但多进程会复制解释器、正则和 worker 状态，因此 optimized 峰值 RSS 从 41.91 MiB 上升到 121.92 MiB。若数据只有 5MB，1 进程更节省内存；追求速度时使用 4 进程。

### 6.5 Scalene 运行限制

预期的 `baseline_p1_scalene.json` 和 `optimized_p1_scalene.json` 均未生成，脚本在第一轮 Scalene 后停止，因此不能声称获得了 Scalene 的逐行内存分配结果。报告中的内存数字来自 benchmark 对根进程及递归子进程 RSS 的持续采样，它适合比较真实进程树峰值，但不能回答某一行代码具体分配了多少内存。

这一限制不影响正式时间、峰值内存、IPC 项数和输出正确性结论；若后续必须取得逐行内存归因，应把 Scalene 单独作为环境兼容性问题处理，而不是重新运行正式 benchmark。

## 7. 平台说明

正式报告只比较同一 WSL 环境中的结果。Windows 使用 `spawn`，WSL/Linux 通常使用 `fork`；进程启动、模块导入和数据复制语义不同，跨平台时间不能直接计算优化百分比。Windows smoke test只用于确认命令和输出格式可运行。
