# Candor 四种强制对齐模式实验

新增 Qwen＋Nyra、Qwen＋MFA＋Nyra。本轮重新运行四种模式，均从 Qwen 全量推理开始；没有复用预测结果来缩短任一方案的计时。

## 数据、模式与验收

- 同一批 4 段 Candor 原始音频，共 70 个片段、8083.8125 秒、18616 个字词、1549 句话。复用已有 ASR，不重新识别。
- 输入：`data/candor_full/manifest.jsonl`，SHA-256：`42fcf5700b50e8859321a61cf690edd203f5e0591f8954a2c6874ed3252800b7`。
- 四种模式的 Qwen 阶段逐词结果完全一致；两次 MFA 阶段逐词结果也完全一致，已自动核验。
- 任一字词零时长即触发完整句子回退；相邻触发句按原规则合为句组，组内全部词一起替换。所有模式使用相同 0.35 秒裁剪余量和边界门槛。
- Qwen＋Nyra：380 个 Qwen 异常句组直接送 Nyra。Qwen＋MFA＋Nyra：先完成 MFA，只将被 MFA 验收拒绝的完整句组送 Nyra，保留已接受的 MFA 结果。
- 候选须文本覆盖及词序一致、无 skipped、未解析词或漏词、每个词为正时长，且不与未修改邻词重叠、不越出切片。Nyra 可通过 espeak 为词典外词生成发音。失败不伪造时长，保持上一阶段原结果，记录到单独的 `unresolved.json`。
- Nyra 0.2.2，英文 standard 权重，FP32；WavLM 在第一次对齐前完整加载。独立的 um/uh 映射到官方 [UM]/[UH] 输入，输出恢复原文本。模型文件及 SHA-256 见 `models/nyra-forced-aligner/checkpoints/download-provenance.json`。
- Qwen 为本地 0.6B、BF16、SDPA、public backend、batch size 4；MFA 3.4.3、English US ARPA、8 CPU jobs、beam 100/retry 400。

## 零时长程度与错误率

错误率为时间戳结构异常率，没有人工词级时间真值，不能解释为边界准确率。零时长阈值为 `abs(end-start) <= 1e-7`；字词率分母是全部输出字词，句子/片段率分母分别是全部非空句子/全部片段。异常词按词索引去重。

| 指标 | Qwen | Qwen＋MFA | Qwen＋Nyra | Qwen＋MFA＋Nyra |
|---|---:|---:|---:|---:|
| 零时长字词数 | 1201 | 118 | 106 | 47 |
| 零时长字词率 | 6.4514% | 0.6339% | 0.5694% | 0.2525% |
| 全部异常字词数 | 1203 | 119 | 107 | 48 |
| 全部异常字词率 | 6.4622% | 0.6392% | 0.5748% | 0.2578% |
| 含错误句子数 | 681 | 78 | 51 | 29 |
| 句子错误率 | 43.9638% | 5.0355% | 3.2924% | 1.8722% |
| 含错误片段数 | 65 | 28 | 24 | 13 |
| 片段错误率 | 92.8571% | 40.0000% | 34.2857% | 18.5714% |
| 推理失败片段 | 0 | 0 | 0 | 0 |
| out_of_bounds | 2 | 1 | 1 | 1 |
| negative_duration | 0 | 0 | 0 | 0 |
| nonfinite | 0 | 0 | 0 | 0 |
| nonmonotonic | 0 | 0 | 0 | 0 |
| overlap | 0 | 0 | 0 | 0 |
| 含零时长词的句子数 | 680 | 77 | 50 | 28 |
| 含零时长词的片段数 | 65 | 28 | 23 | 12 |
| 最长连续零时长词数 | 6 | 6 | 6 | 6 |

## RTF 与阶段耗时

RTF = 每个进程实测墙钟 / 同一份音频总时长。端到端包括输入检查、全部模型加载、Qwen、MFA/Nyra、被拒绝的尝试、验证与输出，不含共用切片准备和首次权重下载。处理 RTF 排除 Qwen 输入/加载与 Nyra 显式加载；保留各后端首条推理及 Nyra JIT、MFA CLI 生命周期，不是稳定热推理速度。

| 指标 | Qwen | Qwen＋MFA | Qwen＋Nyra | Qwen＋MFA＋Nyra |
|---|---:|---:|---:|---:|
| 总墙钟（秒） | 62.132581 | 182.434938 | 87.977116 | 186.799203 |
| Qwen 输入/加载（秒） | 48.990617 | 47.913464 | 44.383883 | 43.284198 |
| Qwen 阶段（秒） | 13.034359 | 13.064861 | 12.924326 | 12.921616 |
| Nyra 模型加载（秒） | 0.000000 | 0.000000 | 4.348670 | 4.072649 |
| 端到端 RTF | 0.007686 | 0.022568 | 0.010883 | 0.023108 |
| 排除显式神经模型加载的处理 RTF | 0.001626 | 0.016641 | 0.004855 | 0.017250 |

运行两轮：先 Qwen（GPU 0）和 Qwen＋MFA（GPU 1），再 Qwen＋Nyra（GPU 0）和 Qwen＋MFA＋Nyra（GPU 1）；每个神经方案只使用一张 H100。GPU 映射依次为：`['0', '1', '0', '1']`。这是单轮测量，共享 CPU/文件系统和缓存会影响耗时，不表示统计显著性。

## Nyra 新方案阶段统计

| 指标 | Qwen＋Nyra 的 Nyra 阶段 | Qwen＋MFA＋Nyra 的 Nyra 阶段 |
|---|---:|---:|
| 请求句组 | 380 | 39 |
| 请求完整句子 | 680 | 77 |
| 接受句组 | 353 | 25 |
| 拒绝句组 | 27 | 14 |
| 修复完整句子 | 630 | 49 |
| 修复零时长字词 | 1095 | 71 |
| 替换全部字词 | 10823 | 541 |
| Nyra 显式加载秒数 | 4.348670 | 4.072649 |
| Nyra 对齐和验收秒数 | 16.274325 | 2.257824 |

三级方案零时长字词轨迹：Qwen 1201 → MFA 后 118 → Nyra 后 47；Nyra 在 MFA 基础上再减少 71 个。

Nyra 拒绝原因：

- Qwen＋Nyra：`{"Sentence is invalid after boundary clamping": 23, "Nyra skipped input tokens: ['2005', '2005', '2008']": 1, "Nyra skipped input tokens: ['2008', '250']": 1, "Nyra skipped input tokens: ['250']": 1, "No positive interval between unchanged neighbors": 1}`。
- Qwen＋MFA＋Nyra：`{"Sentence is invalid after boundary clamping": 13, "No positive interval between unchanged neighbors": 1}`。

## 两个新方案的实际例子

以下分别展示 Nyra 直接回退和 MFA 拒绝后由 Nyra 接手的实际完整句子。为便于阅读，文档时间戳显示至 6 位小数，原始文件保留完整精度。

### qwen+nyra

```json
{
  "id": "0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a:c00000",
  "source_audio": "/mnt/data/L202500721/stevezhuang/candor/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a.mp3",
  "text": "Wait, I can't hear you at all",
  "timestamp_origin": "source recording",
  "status": "repaired",
  "input_zero_words": 1,
  "words": [
    {
      "text": "Wait",
      "start": 85.48,
      "end": 85.74,
      "source": "nyra"
    },
    {
      "text": "I",
      "start": 85.88,
      "end": 86.01,
      "source": "nyra"
    },
    {
      "text": "can't",
      "start": 86.01,
      "end": 86.45,
      "source": "nyra"
    },
    {
      "text": "hear",
      "start": 86.45,
      "end": 86.6,
      "source": "nyra"
    },
    {
      "text": "you",
      "start": 86.6,
      "end": 86.76,
      "source": "nyra"
    },
    {
      "text": "at",
      "start": 86.76,
      "end": 86.86,
      "source": "nyra"
    },
    {
      "text": "all",
      "start": 86.86,
      "end": 87.09,
      "source": "nyra"
    }
  ],
  "boundary_limits": {
    "left": 85.28,
    "right": 88.08
  },
  "reason": null,
  "mfa_rejection_reason": null
}
```

### qwen+mfa+nyra

```json
{
  "id": "0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a:c00009",
  "source_audio": "/mnt/data/L202500721/stevezhuang/candor/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a.mp3",
  "text": "Do you guys or have you come across this in your studies?",
  "timestamp_origin": "source recording",
  "status": "repaired",
  "input_zero_words": 1,
  "words": [
    {
      "text": "Do",
      "start": 1104.8,
      "end": 1104.859938,
      "source": "nyra"
    },
    {
      "text": "you",
      "start": 1104.859938,
      "end": 1104.949938,
      "source": "nyra"
    },
    {
      "text": "guys",
      "start": 1104.949938,
      "end": 1105.449938,
      "source": "nyra"
    },
    {
      "text": "or",
      "start": 1105.779937,
      "end": 1105.859938,
      "source": "nyra"
    },
    {
      "text": "have",
      "start": 1105.859938,
      "end": 1106.109938,
      "source": "nyra"
    },
    {
      "text": "you",
      "start": 1106.109938,
      "end": 1106.649938,
      "source": "nyra"
    },
    {
      "text": "come",
      "start": 1106.959938,
      "end": 1107.319937,
      "source": "nyra"
    },
    {
      "text": "across",
      "start": 1107.319937,
      "end": 1107.719938,
      "source": "nyra"
    },
    {
      "text": "this",
      "start": 1107.719938,
      "end": 1107.889938,
      "source": "nyra"
    },
    {
      "text": "in",
      "start": 1107.889938,
      "end": 1107.999937,
      "source": "nyra"
    },
    {
      "text": "your",
      "start": 1107.999937,
      "end": 1108.099938,
      "source": "nyra"
    },
    {
      "text": "studies",
      "start": 1108.099938,
      "end": 1108.589938,
      "source": "nyra"
    }
  ],
  "boundary_limits": {
    "left": 1104.8,
    "right": 1109.04
  },
  "reason": null,
  "mfa_rejection_reason": "Sentence is invalid after boundary clamping"
}
```

## 结果与复现

本轮结果根目录：`results/candor_four_modes/`。四个模式各有独立目录：`qwen-fa/`、`qwen3-fa+mfa/`、`qwen3-fa+nyra/`、`qwen3-fa+mfa+nyra/`。旧的双方案结果及记录保留。

- 每个模式有 `predictions.jsonl`、`qwen_predictions.jsonl`、`error_stats.json`、`runtime.json`、`metadata.json`、`unresolved.json`。
- 两个新增模式各有 `nyra/tasks.jsonl`、`nyra/predictions.jsonl`（原始裁剪内时间）、`nyra/repairs.jsonl` 和 `nyra/runtime.json`；`example.json` 是上文实际例子。
- 三级方案另存 `mfa_predictions.jsonl`、`mfa_error_stats.json`、`mfa/unresolved.json` 和 MFA 的全部输入/输出及拒绝原因，能追溯每次转交。
- 汇总：`comparison.json`；全量整句替换、边界、词序及未解决记录校验：`validation.json`。

下载与启动命令见 [design.md](design.md)。运行 `python src/compare_four_modes.py` 可重新核验并生成本记录。`results/nyra-tests.log` 保存 16 项回归检查。
