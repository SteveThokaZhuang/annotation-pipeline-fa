# Candor v5 预处理、ASR 与两种 JSON 粒度灰度

日期：2026-11-09。此次目的为核验标注 pipeline 能输出约定格式，所记录的小样本耗时不作为吞吐性能结论。

输入仍为原选定的前 3 条 120 秒 WAV，总计 360 秒：`0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a_c00000.wav`、`_c00001.wav`、`_c00002.wav`。它们来自同一原始 MP3 的 0–120、120–240、240–360 秒，属于 3 个实际输入文件，不是前 3 个完整会话。输入与父录音区间见 [input_windows.jsonl](../data/candor_v5_smoke/input_windows.jsonl)。本次没有使用原先 manifest 中的 ASR 文本。

旧的 Candor 格式灰度及其诊断结果在重跑前删除；历史完整消融实验保留。speaker_b 的 35 个已有文件保留，只按后续要求给整条音轨 JSON 的 440 个 `annotated_segments` 补充各自文本，词和时间戳均未修改。见 [清理记录](../results/smoke-test/candor_cleanup.json)、[speaker_b 文本补充记录](../results/smoke-test/speaker_b_text_update.json)。

使用的原生预处理与 ASR 仓库为 `/hbt_share02/202610/ob/ob_ppl_env_data/audio_pipeline_optimized/Pipeline_github_v5`。顺序为 ingest → DNSMOS → VAD → DeepFilterNet3 增强 → Qwen ASR → FunASR → disagreement → 按需 FireRed → ROVER。原生 `align` 阶段不运行，ROVER 文本随后送入本项目的 Qwen＋MFA、Qwen＋MFA＋Nyra，二者依次执行。输入目录不附带旧转写；参数配置保存在 [config](../data/candor_v5_smoke/config/)，完整命令保存在 [job.json](../results/smoke-test/candor_v5/job.json)。预处理使用启动时检查为空闲的 GPU 3；每种 FA 开始前重新选卡，要求连续两次计算利用率为 0%，且空闲显存至少 32 GiB，快照写入各模式的 `.command.json`。

本次 MFA 的实际组成如下；`mfa` 与 `mfa-english` 是程序环境和语言模型资源的分工，并非两套对齐后端混用。Qwen＋MFA 和 Qwen＋MFA＋Nyra 使用相同配置：

| 组成 | 本次实际使用 |
| --- | --- |
| 对齐程序 | Montreal Forced Aligner **3.4.3** |
| Python / MFA 环境 | `models/mfa/bin/python` |
| 英文声学模型 | `models/mfa-english/english_us_arpa.zip` |
| 英文发音词典 | `models/mfa-english/dictionary/english_us_arpa.dict` |

因此可准确称为「MFA 3.4.3 ＋ English US ARPA 声学模型和词典」。MFA 触发时，代码显式把上述模型和词典路径传给 `align`；不自动切换成另一套默认模型。三 FA 方案仅在该 MFA 整句候选仍未通过检查时再调用 Nyra。资源绝对路径与 SHA256 见 [mfa_resources.json](../results/smoke-test/candor_v5/mfa_resources.json)，实际命令也会写入各块的 MFA runtime 记录。

首轮原生 Qwen ASR 遇到 `AttributeError: AudioPrefetch ... spool`，34 个 chunk 均失败。保留失败记录，停止该轮后通过本次私有配置关闭跨 bundle 预取与相关 ASR 准备优化，复用已成功的预处理并重试。原生仓库源码未改。初始配置保存为 `resources.initial.yaml`，实际成功重试配置为 `resources.yaml`，见 [重试说明](../results/smoke-test/candor_v5/preprocess_retry.json)。失败转写不用于 FA。

v5 增强音频为 48 kHz，VAD 原采样坐标为 16 kHz。读取增强音频时使用 `audio_start_sample/audio_end_sample` 与实际采样率，另保留原 VAD 采样坐标；测试覆盖该转换。最终 JSON 的音频路径指向本次持久化的增强 WAV，不依赖临时目录。`asr/sources.json` 记录音频 SHA256、原输入和父 MP3 区间。

两种保存方式共享同一份 FA 结果，不重复推理：

| 目录 | 文件粒度 | FA 时间原点 |
| --- | --- | --- |
| `<模式>/audio_json/` | 每个实际 WAV 一个 JSON | 整条 WAV 起点 |
| `<模式>/chunk_json/` | 每个 v5 VAD/ASR chunk 一个 JSON | 该 chunk 起点 |

每个 audio JSON 包含 `annotated_segments: [{chunk_id, start, end, text, word_start_index, word_end_index}, ...]`；起止时间相对完整 WAV，索引对应 `force_align` 数组切片，右端不包含。每个 chunk JSON 保留自身完整 `text` 和 WAV 内区间 `audio_start_sec/audio_end_sec`。两个版本均保持 FA 词元、词序和时间精度；异常原样保留并标记复核。v5 chunk 来自 VAD，不是 turn，没有依据填充 user/assistant，因此保留 `turns: []`、`turns_status: "not_available"`。

可直接在推理入口指定 `--export-unit chunk`、`--export-unit audio` 或 `--export-unit both`。不指定时保持原 JSONL 保存方式。已有结果使用以下命令导出，`--input` 可为模式目录、含 `blocks/` 的 worker 目录，或标准 `predictions.jsonl`：

```bash
/mnt/data/L202500293/obed/envs/qwenasr/bin/python src/export_annotations.py \
  --input <FA结果目录> --output <新导出目录> --unit both
```

单模式目录保留 `predictions.jsonl`、完整审计 `alignment_predictions.jsonl`、`unresolved.json`、路由记录、统计和运行时间。`export_index.json` 映射实际 JSON 文件、chunk ID 和音频文件；不向 JSON 文件名直接写入任意路径，也不覆盖已存在的同类导出目录。

本次后处理/FA/导出脚本为 [finish_candor_v5.py](../src/finish_candor_v5.py)，复核已有结果使用 `--finalize-only`，不会重新推理。`--preprocess` 可先运行/续跑 `job.json` 中保存的原生 v5 命令，再执行 FA；本次原生预处理启动日志为 `preprocess.log` 和 `preprocess_retry.log`。

当前运行结果与格式检查写入 [summary.json](../results/smoke-test/candor_v5/summary.json)。最终导出验证要求：两种粒度覆盖全部 chunk，各区间文本一致；词数组与审计输出相同；chunk 相对时间加区间起点能恢复 audio 时间；整句替换和边缘核验通过。

本次原生 v5 重试已成功：Qwen ASR、FunASR、disagreement 和 ROVER 各处理 34 个 chunk，FireRed 按需处理其中 18 个，所有阶段最终失败数均为 0。3 个 WAV 的原始总时长为 360 秒，VAD/ROVER 区间合计 **274.38 秒**。两种 FA 的 RTF 以这 274.38 秒实际输入语音区间为分母，包含模型加载、整句回退、检查和 JSON 导出；原生预处理与 ASR 时间不计入 FA RTF。

## 完成结果与查看入口

两方案均完成；每套 **3 个 audio JSON＋34 个 chunk JSON**，合计 74 个标注 JSON。逐 chunk 文本、数组词序、全音频时间偏移及整句替换核验均通过，见 [最终检查](../results/smoke-test/candor_v5/final_validation.json)。33 项自动检查通过，见 [tests.log](../results/smoke-test/candor_v5/tests.log)。两方案的 Qwen 初始输出 SHA256 一致，均使用 GPU 3；共用同一组新 ROVER 文本。

| 方案 | 按音频文件 | 按 chunk | 索引 |
| --- | --- | --- | --- |
| Qwen＋MFA | [audio_json](../results/smoke-test/candor_v5/qwen3-fa+mfa/audio_json/) | [chunk_json](../results/smoke-test/candor_v5/qwen3-fa+mfa/chunk_json/) | [export_index.json](../results/smoke-test/candor_v5/qwen3-fa+mfa/export_index.json) |
| Qwen＋MFA＋Nyra | [audio_json](../results/smoke-test/candor_v5/qwen3-fa+mfa+nyra/audio_json/) | [chunk_json](../results/smoke-test/candor_v5/qwen3-fa+mfa+nyra/chunk_json/) | [export_index.json](../results/smoke-test/candor_v5/qwen3-fa+mfa+nyra/export_index.json) |

| 阶段/方案 | 词数 | 零时长词/比例 | 异常词/比例 | 异常 chunk/比例 | FA 耗时 | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 初始 Qwen（两次一致） | 750 | 34 / 4.5333% | 36 / 4.8000% | 15 / 44.1176% | — | — |
| Qwen＋MFA | 750 | 0 / 0.0000% | 2 / 0.2667% | 2 / 5.8824% | 142.600 秒 | 0.519718 |
| Qwen＋MFA＋Nyra | 750 | 0 / 0.0000% | 2 / 0.2667% | 2 / 5.8824% | 148.478 秒 | 0.541139 |

两套均由 MFA 接受 13/13 组整句候选，替换 621 个词，修复全部 34 个零时长词；含零时长的 chunk 从 13/34 降至 0/34，最长连续零时长词从 3 降至 0。本次 ROVER 文本没有句末标点，因此每个 VAD/ROVER chunk 的完整可用文本作为一个句组处理，不据此推断 turn。

三 FA 中 **Nyra 请求数为 0**：MFA 已解决全部零时长触发句组，未加载或运行 Nyra。本次证明三 FA 路由与格式能够完整输出；不能据此衡量 Nyra 的额外修复收益。

两方案各剩 **2 个原始 Qwen 越界词**，分布在 c00001 WAV 的两个 chunk：`yeah` 的输出超过该短 chunk 右边界，`again` 的结束时间为 120.035 秒，超过 WAV 的 120 秒结尾。它们没有零时长触发条件，保留原始时间与 `needs_review` 标记，记录在各模式的 `unresolved.json`。上表异常词率按去重异常词数/总词数计算，chunk 错误率按异常 chunk/全部 chunk 计算；未使用人工词级真值，未计算 WER 或词边界准确率。

RTF 包含单次模型加载、实际音频读取/重采样、Qwen、回退、QA、归档与两种 JSON 导出，分母为 274.38 秒；原生预处理/ASR、两模式之间的调度和最终 Markdown 报告生成不在此指标内。该灰度样本小且模型加载占比较大，两模式分别执行 MFA 导致耗时有所波动，不用于推导吞吐优劣。

新 ASR 文本见 [transcripts.jsonl](../results/smoke-test/candor_v5/asr/transcripts.jsonl)，实际增强音频及来源映射见 [sources.json](../results/smoke-test/candor_v5/asr/sources.json)。原生成功重试总耗时为 617.901 秒，启动前首轮失败另保留日志，不混入 FA RTF。


### Qwen＋MFA 实际例子

[完整 chunk JSON](../results/smoke-test/candor_v5/qwen3-fa+mfa/chunk_json/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a_c00000_c0_1298560_1342720_646a2db8d86b.json)；[对应整条 WAV JSON](../results/smoke-test/candor_v5/qwen3-fa+mfa/audio_json/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a_c00000_43d769c0f799.json)。以下为该 chunk 的完整词级数组：

```json
{
  "chunk_id": "0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a_c00000_c0_1298560_1342720",
  "audio_start_sec": 81.16,
  "audio_end_sec": 83.92,
  "text": "hi i can't hear you can you hear me",
  "force_align": {
    "timestamps": [
      [
        0.3700000047683716,
        0.4399999976158142
      ],
      [
        0.4699999988079071,
        0.7799999713897705
      ],
      [
        1.1299999952316284,
        1.25
      ],
      [
        1.25,
        1.3700000047683716
      ],
      [
        1.3700000047683716,
        1.5299999713897705
      ],
      [
        1.5299999713897705,
        1.6799999475479126
      ],
      [
        1.6799999475479126,
        1.7999999523162842
      ],
      [
        1.7999999523162842,
        2.009999990463257
      ],
      [
        2.009999990463257,
        2.4100000858306885
      ]
    ],
    "words": [
      "hi",
      "i",
      "can't",
      "hear",
      "you",
      "can",
      "you",
      "hear",
      "me"
    ]
  }
}
```
同一段在 audio JSON 中的区间项含完整文本；其词级数组位于上述左右索引之间：

```json
{
  "chunk_id": "0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a_c00000_c0_1298560_1342720",
  "start": 81.16,
  "end": 83.92,
  "text": "hi i can't hear you can you hear me",
  "word_start_index": 8,
  "word_end_index": 17,
  "channel": 0
}
```

### Qwen＋MFA＋Nyra 实际例子

[完整 chunk JSON](../results/smoke-test/candor_v5/qwen3-fa+mfa+nyra/chunk_json/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a_c00002_c0_21120_72320_58d41fb1c365.json)；[对应整条 WAV JSON](../results/smoke-test/candor_v5/qwen3-fa+mfa+nyra/audio_json/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a_c00002_e6e5d2eaa18b.json)。以下为该 chunk 的完整词级数组：

```json
{
  "chunk_id": "0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a_c00002_c0_21120_72320",
  "audio_start_sec": 1.32,
  "audio_end_sec": 4.5200000000000005,
  "text": "do you guys have lab work to do or no",
  "force_align": {
    "timestamps": [
      [
        0.3800000131130219,
        0.4500000059604645
      ],
      [
        0.4500000059604645,
        0.529999989271164
      ],
      [
        0.529999989271164,
        0.7499999880790711
      ],
      [
        0.7499999880790711,
        0.900000023841858
      ],
      [
        0.900000023841858,
        1.3700000524520874
      ],
      [
        1.3700000524520874,
        1.5799999713897706
      ],
      [
        1.5799999713897706,
        1.709999966621399
      ],
      [
        1.709999966621399,
        2.1399999141693113
      ],
      [
        2.169999885559082,
        2.2600000381469725
      ],
      [
        2.2600000381469725,
        2.7500000476837156
      ]
    ],
    "words": [
      "do",
      "you",
      "guys",
      "have",
      "lab",
      "work",
      "to",
      "do",
      "or",
      "no"
    ]
  }
}
```
同一段在 audio JSON 中的区间项含完整文本；其词级数组位于上述左右索引之间：

```json
{
  "chunk_id": "0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a_c00002_c0_21120_72320",
  "start": 1.32,
  "end": 4.5200000000000005,
  "text": "do you guys have lab work to do or no",
  "word_start_index": 3,
  "word_end_index": 13,
  "channel": 0
}
```
