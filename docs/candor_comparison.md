# Candor：Qwen FA 与整句 MFA 回退对比

本次按最终要求执行：任一字词的时长为 0，则该字词所在的整句话全部交给 MFA；接受时整句所有词统一替换，不混用该句的 Qwen 时间戳。

## 数据与运行条件

- 音频：`/mnt/data/L202500721/stevezhuang/candor` 下的全部 4 个 MP3。
- ASR：复用 `../annotation_pipeline/results/v4/candor_v4_0_1/asr.jsonl`，没有重新识别。
- 两方案共用 70 个 16 kHz 单声道切片，总时长 8083.8125 秒；manifest：`data/candor_full/manifest.jsonl`。
- 纯 Qwen 使用物理 GPU 0；混合使用物理 GPU 1。各进程设置 CUDA_VISIBLE_DEVICES 后内部设备均为 cuda:0。MFA 在 CPU 上使用 8 个 job。
- Qwen3-ForcedAligner-0.6B，public 后端、bfloat16、SDPA、batch size 4；MFA 3.4.3，官方 English US ARPA 模型和配套词典。
- manifest SHA-256：`42fcf5700b50e8859321a61cf690edd203f5e0591f8954a2c6874ed3252800b7`；两次 Qwen 阶段的全部词及时间戳完全一致：True。
- 时间戳单位为秒，原点为原始录音；切片偏移只添加一次。准备切片和下载模型不计入 FA RTF，准备耗时见 `data/candor_full/preparation.json`。

## 错误统计与错误率

这些是可程序检测的时间戳结构错误率。没有人工词级时间戳真值，因此不代表边界准确率、WER 或人工听审准确率。中文按模型输出的字、英文按模型输出的词统计。

| 指标 | 纯 Qwen（GPU 0） | Qwen＋MFA（GPU 1） |
|---|---:|---:|
| 字词总数 | 18616 | 18616 |
| 零时长字词数 | 1201 | 118 |
| 零时长字词率 | 6.4514% | 0.6339% |
| 异常字词数（去重） | 1203 | 119 |
| 异常字词率 | 6.4622% | 0.6392% |
| 句子总数 | 1549 | 1549 |
| 含错误句子数 | 681 | 78 |
| 句子错误率 | 43.9638% | 5.0355% |
| 片段总数 | 70 | 70 |
| 含错误片段数 | 65 | 28 |
| 片段错误率 | 92.8571% | 40.0000% |
| 推理失败片段数 | 0 | 0 |
| 文本非空但无输出片段数 | 0 | 0 |
| negative_duration | 0 | 0 |
| nonfinite | 0 | 0 |
| out_of_bounds | 2 | 1 |
| nonmonotonic | 0 | 0 |
| overlap | 0 | 0 |

分母：字词率 = 对应异常字词数 / 全部输出字词数；句子率 = 含至少一个异常或推理失败的句子数 / 全部非空句子数；片段率 = 含异常、失败或非空文本无输出的片段数 / 全部片段数。异常字词数按索引去重，各类型数可重叠。零时长阈值为 `abs(end-start) <= 1e-7`，其他结构校验同样使用 1e-7 秒容差。

零时长字词减少 1083 个，减少比例 90.1749%。

## MFA 整句回退

- 触发 680 句话，合并为 380 个相邻句组；相邻触发句一起送入 MFA，组内每句都完整替换。
- 成功替换 603 句话、341 个句组、11223 个词，整句修复成功率 88.6765%。
- 修复原始零时长词 1083 个，拒绝 39 个句组。MFA 退出码：`0`。
- 拒绝原因：`{"Sentence is invalid after boundary clamping": 38, "No positive interval between unchanged neighbors": 1}`。
- 未解决项：`results/qwen3-fa+mfa/unresolved.json`，含句子、录音、裁剪范围、原始坏词、原因和最终残留异常。完整修复记录：`mfa/tasks.jsonl`、`mfa/repairs.jsonl`；MFA 原始结果在 `mfa/aligned/`。
- 接受条件：词序和覆盖完整、整句所有字词为正时长、顺序合法；将裁剪相对时间恢复至原录音，再限制在切片及未修改邻词的边界内。边界裁剪造成任一词无时长则拒绝整句，不人为填充时长。

## RTF

RTF = 本方案耗时 / 同一份 8,083.8125 秒音频。墙钟包含模型加载、音频读取、Qwen、可选 MFA、校验与输出；处理 RTF 排除输入检查及 Qwen 加载。两个进程分别测量自身耗时，不能把并行总墙钟当成单方案耗时。单次实验会受文件缓存及共享 CPU/I/O 影响。

| 指标 | 纯 Qwen | Qwen＋MFA |
|---|---:|---:|
| 总墙钟（秒） | 59.493197 | 177.801809 |
| 输入检查和模型加载（秒） | 45.677694 | 45.703201 |
| Qwen 阶段墙钟（秒） | 13.700842 | 13.662616 |
| 回退和输出阶段（秒） | 0.114677 | 118.436008 |
| 端到端 RTF | 0.007360 | 0.021995 |
| 排除加载后的处理 RTF | 0.001709 | 0.016341 |

MFA 子进程耗时 107.362023 秒；含裁剪和替换的整个修复阶段 118.312952 秒。
`runtime.json` 另有 CUDA event 包围 Qwen 调用的 stream 计时；它可能包含 CPU 预处理造成的 GPU 等待，不作为纯 GPU kernel 时间或混合方案总耗时。

## 同一句话的两个实际结果

片段 `0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a:c00000`；原文：Wait, I can't hear you at all。以下为整句全部字词，时间戳相对于原始录音。

纯 Qwen（同时保存为 `results/qwen-fa/example.json`）：

```json
{
  "id": "0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a:c00000",
  "source_audio": "/mnt/data/L202500721/stevezhuang/candor/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a.mp3",
  "text": "Wait, I can't hear you at all",
  "timestamp_origin": "source recording",
  "words": [
    {
      "text": "Wait",
      "start": 85.52,
      "end": 85.68,
      "source": "qwen"
    },
    {
      "text": "I",
      "start": 85.92,
      "end": 85.92,
      "source": "qwen"
    },
    {
      "text": "can't",
      "start": 86.16,
      "end": 86.48,
      "source": "qwen"
    },
    {
      "text": "hear",
      "start": 86.48,
      "end": 86.56,
      "source": "qwen"
    },
    {
      "text": "you",
      "start": 86.56,
      "end": 86.72,
      "source": "qwen"
    },
    {
      "text": "at",
      "start": 86.8,
      "end": 86.88,
      "source": "qwen"
    },
    {
      "text": "all",
      "start": 86.88,
      "end": 87.04,
      "source": "qwen"
    }
  ]
}
```

整句 MFA 替换后（同时保存为 `results/qwen3-fa+mfa/example.json`）：

```json
{
  "id": "0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a:c00000",
  "source_audio": "/mnt/data/L202500721/stevezhuang/candor/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a.mp3",
  "text": "Wait, I can't hear you at all",
  "timestamp_origin": "source recording",
  "words": [
    {
      "text": "Wait",
      "start": 85.48000000238419,
      "end": 85.69999997138977,
      "source": "mfa"
    },
    {
      "text": "I",
      "start": 85.91000000953674,
      "end": 85.9999999833107,
      "source": "mfa"
    },
    {
      "text": "can't",
      "start": 86.20999996185303,
      "end": 86.44999997138977,
      "source": "mfa"
    },
    {
      "text": "hear",
      "start": 86.44999997138977,
      "end": 86.5799999666214,
      "source": "mfa"
    },
    {
      "text": "you",
      "start": 86.5799999666214,
      "end": 86.7600000333786,
      "source": "mfa"
    },
    {
      "text": "at",
      "start": 86.7600000333786,
      "end": 86.83999995708466,
      "source": "mfa"
    },
    {
      "text": "all",
      "start": 86.83999995708466,
      "end": 87.06999997615814,
      "source": "mfa"
    }
  ]
}
```

## 复现与校验

完整启动命令与算法说明见 [design.md](design.md)。运行 `python src/compare_results.py` 可从结果重新生成本记录和 `results/comparison.json`。

运行 `python -m unittest discover -s tests -v` 检查整句原子替换、零时长拒绝、邻句边界、词序/覆盖、缩写分词映射、录音时间偏移、空批次和错误率分母。

本次 10 项回归检查通过。比较脚本另对全部 70 个片段、380 个句组验证：整句替换完整、接受词全部为正时长且不越界、拒绝句与邻词未被修改、词数词序不变、未解决项记录完整。结果见 `results/validation.json`。

边界限制：句子以已有 ASR 标点划分；标点和 120 秒 ASR 分块本身的语义准确性未经人工审校。输入发生词元映射歧义时保守地回退整个 ASR 片段。MFA 对重叠讲话、漏词、ASR 幻觉或词典外词仍可能失败；失败保持原结果并记录，正时长也不能保证声学边界正确。
