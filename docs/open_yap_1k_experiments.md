# OpenYap ROVER 四卡流式强制对齐实验

输入共 1,796 个 ROVER 批结果，919,459 个唯一片段，3,205 个录音通道，标注区间总长 3992338.070771 秒（1108.983 小时）。文本及原音频不修改。

原始输入：`/hbt_share02/202610/ob/ob_ppl_env_data/audio_pipeline_optimized/Pipeline_github_v2/outputs/production/open_yap_1k/rover/rover/batch_results`。四卡清单及来源文件 SHA-256 位于 `data/open_yap_1k_rover/preparation.json`。

本轮按要求依次执行三级方案、MFA 双路由消融、Nyra 双路由消融；当前完整完成 3/3 组。每组都独立使用 GPU 0–3，四卡按标注时长分片，分片清单相同。

## RTF 与最终零时长情况

主 RTF 包含四卡加载模型、读取/重采样、全部成功和失败回退、边界检查、验证、输出归档与汇总。分母为全部唯一标注区间秒数之和。RTF 用协调器真实墙钟计算，不能用各卡耗时相加替代。共享的输入清点、磁盘空间恢复及分片导出不计入各组 FA 墙钟。

| 方案 | 完成片段 | 总字词 | 零时长字词 | 零时长率 | 四卡墙钟秒 | 四卡 RTF |
|---|---:|---:|---:|---:|---:|---:|
| Qwen＋MFA＋Nyra | 919,459 | 10,301,121 | 186 | 0.00181% | 5351.198 | 0.00134037 |
| Qwen＋MFA | 919,459 | 10,301,121 | 747 | 0.00725% | 5176.028 | 0.00129649 |
| Qwen＋Nyra | 919,459 | 10,301,121 | 3,361 | 0.03263% | 4129.453 | 0.00103434 |

错误率为时间戳结构异常率，没有人工词级时间真值，因此不当作声学边界准确率。零时长阈值为 `abs(end-start) <= 1e-7`。字词/句子/片段百分比分别用全量字词/非空句子/片段计数重算，不平均各卡或各块的比例。

本次三级方案零时长残留最少：比 Qwen＋MFA 少 561 个（降低 75.10%），比 Qwen＋Nyra 少 3,175 个（降低 94.47%）。Qwen＋Nyra 的四卡 RTF 最小，比三级方案墙钟短 1221.745 秒。

三级中的 Nyra 仅收到 67 个回退组，接受 52 个；直接 Qwen＋Nyra 收到 303,055 个回退组。三组 Qwen 输出的逐卡 SHA-256 一致：`True`。

这些耗时来自各方案的一次全量顺序运行；共享主机负载可能变化，墙钟差额不能完全归因于单一回退模型。

## 分阶段及残留程度

### Qwen＋MFA＋Nyra

| 阶段 | 总词数 | 零时长词 | 零时长率 | 含零词句子 | 含零词片段 | 最长连续零词 | 全部异常词率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen | 10,301,121 | 535,259 | 5.19612% | 303,060 | 303,054 | 85 | 5.21130% |
| mfa | 10,301,121 | 747 | 0.00725% | 69 | 66 | 85 | 0.01492% |
| final | 10,301,121 | 186 | 0.00181% | 17 | 14 | 85 | 0.00940% |

- 推理失败 0 个片段；非空文本无输出 0 个片段。
- 最终各类异常：`{"out_of_bounds": 784, "zero_duration": 186}`。
- 请求及接受数：`{"mfa_requested": 303055, "mfa_accepted": 302988, "nyra_requested": 67, "nyra_accepted": 52}`。
- 四卡阶段耗时之和（秒）：`{"mfa_command_seconds": 10822.247941785958, "nyra_model_load_seconds": 23.48278250801377, "nyra_inference_seconds": 13.04846029705368, "qwen_seconds": 9168.882265037159, "repair_seconds": 11475.490253239637}`。
- 四卡 worker 总耗时 21280.641 秒，归一化工作量 RTF 0.00533037；它不是四卡吞吐 RTF。
- Qwen 输出与第一组逐卡 SHA-256 一致：`True`。

三级零时长轨迹：**535,259 → 747 → 186**；Nyra 在 MFA 后再修复 561 个。

仍含零时长的片段共 14 个；回退拒绝原因前五项：`{"Nyra transcript coverage mismatch (missing/OOV/changed words)": 8, "alignment failed at all beams (85 tokens, 34 frames)": 1, "No positive interval between unchanged neighbors": 1, "Sentence is invalid after boundary clamping": 1, "Nyra skipped input tokens: ['51', '51']": 1}`。

零时长字词最多的残留片段（完整原因和原时间戳在单独未解决记录中）：

```json
[
  {
    "id": "f27e49dee2dd9e0dc4c1784f_c0_100802640_101927040",
    "text": "cash it ain't i'm fuckin safety if she wanna fuck me she gotta pay you go dc just to give me a cut cut that bitch all she ain't making me cut cut that bitch all she ain't making rock it me park it some fuck it we're arms up now i flex keep all your clothes i just want me to net promote pay me i don't want me no check cash money i don't want me no check cash money like i'm you",
    "zero_words": 85,
    "source_audio": "/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing/open_yap_1k/conv_f32324299a39/speaker_b.wav",
    "record": "/tmp/fa_open_yap_results/qwen3-fa+mfa+nyra/gpu_0/blocks/b00006/unresolved.json"
  },
  {
    "id": "9f693c4db5b895f18fe1c67d_c0_105660960_106809360",
    "text": "it's some gaga gaga came with twenty one i just bought my thirty six for a block twenty one ain't got thirty eight that fool ain't near twenty three my nigga if the boy did this it look like a combo my twenty six looks like a hundred some brothers sent us that for another and that's a brand",
    "zero_words": 59,
    "source_audio": "/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing/open_yap_1k/conv_9267fbe11895/speaker_b.wav",
    "record": "/tmp/fa_open_yap_results/qwen3-fa+mfa+nyra/gpu_2/blocks/b00003/unresolved.json"
  },
  {
    "id": "48d2e0e20d13ac87578f04e3_c0_51990240_53123280",
    "text": "and the 51 year old black boyfriend is not special needs is he preying on her i don't care what you say because law and order mariska her heart her heart's crazy she said i remember they had episodes like that and she made it clear if someone is special needs and no matter how old they are they have a mind of a toddler and you are a grown person and you're 51 years old",
    "zero_words": 15,
    "source_audio": "/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing/open_yap_1k/conv_24c4af577de5/speaker_a.wav",
    "record": "/tmp/fa_open_yap_results/qwen3-fa+mfa+nyra/gpu_2/blocks/b00000/unresolved.json"
  }
]
```

实际完整回退例子（原始录音秒坐标）：

```json
{
  "id": "261064be9d67c061602ad814_c0_15208800_15692160",
  "text": "yikes damn yikes okay",
  "source_audio": "/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing/open_yap_1k/conv_1556519a4556/speaker_a.wav",
  "backend": "nyra",
  "crop_start": 316.85,
  "crop_end": 326.2400625,
  "original_words": [
    {
      "text": "yikes",
      "start": 317.17,
      "end": 317.73,
      "source": "qwen"
    },
    {
      "text": "damn",
      "start": 320.69,
      "end": 320.69,
      "source": "qwen"
    },
    {
      "text": "yikes",
      "start": 324.53000000000003,
      "end": 325.09000000000003,
      "source": "qwen"
    },
    {
      "text": "okay",
      "start": 325.41,
      "end": 325.89000000000004,
      "source": "qwen"
    }
  ],
  "replacement_words": [
    {
      "text": "yikes",
      "start": 317.16,
      "end": 317.71000000000004,
      "source": "nyra"
    },
    {
      "text": "damn",
      "start": 320.66,
      "end": 321.19,
      "source": "nyra"
    },
    {
      "text": "yikes",
      "start": 324.48,
      "end": 325.07000000000005,
      "source": "nyra"
    },
    {
      "text": "okay",
      "start": 325.43,
      "end": 325.96000000000004,
      "source": "nyra"
    }
  ],
  "boundary_limits": {
    "left": 316.85,
    "right": 326.92
  },
  "record": "/tmp/fa_open_yap_results/qwen3-fa+mfa+nyra/gpu_2/blocks/b00000/nyra/repairs.jsonl",
  "task_id": "s006033"
}
```

### Qwen＋MFA

| 阶段 | 总词数 | 零时长词 | 零时长率 | 含零词句子 | 含零词片段 | 最长连续零词 | 全部异常词率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen | 10,301,121 | 535,259 | 5.19612% | 303,060 | 303,054 | 85 | 5.21130% |
| mfa | 10,301,121 | 747 | 0.00725% | 69 | 66 | 85 | 0.01492% |
| final | 10,301,121 | 747 | 0.00725% | 69 | 66 | 85 | 0.01492% |

- 推理失败 0 个片段；非空文本无输出 0 个片段。
- 最终各类异常：`{"out_of_bounds": 841, "zero_duration": 747}`。
- 请求及接受数：`{"mfa_requested": 303055, "mfa_accepted": 302988}`。
- 四卡阶段耗时之和（秒）：`{"mfa_command_seconds": 10797.601556180976, "qwen_seconds": 8591.896814194042, "repair_seconds": 11339.886061596684}`。
- 四卡 worker 总耗时 20515.402 秒，归一化工作量 RTF 0.00513869；它不是四卡吞吐 RTF。
- Qwen 输出与第一组逐卡 SHA-256 一致：`True`。

仍含零时长的片段共 66 个；回退拒绝原因前五项：`{"MFA has no unique output (exit=0)": 45, "MFA transcript coverage mismatch (missing/OOV/changed words)": 9, "Sentence is invalid after boundary clamping": 8, "MFA returned an unknown word": 4, "No positive interval between unchanged neighbors": 1}`。

另有未触发零时长回退的原始异常记录：`{"qwen_error_without_zero_trigger": 778}`；它们不计入后端回退拒绝数。

零时长字词最多的残留片段（完整原因和原时间戳在单独未解决记录中）：

```json
[
  {
    "id": "f27e49dee2dd9e0dc4c1784f_c0_100802640_101927040",
    "text": "cash it ain't i'm fuckin safety if she wanna fuck me she gotta pay you go dc just to give me a cut cut that bitch all she ain't making me cut cut that bitch all she ain't making rock it me park it some fuck it we're arms up now i flex keep all your clothes i just want me to net promote pay me i don't want me no check cash money i don't want me no check cash money like i'm you",
    "zero_words": 85,
    "source_audio": "/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing/open_yap_1k/conv_f32324299a39/speaker_b.wav",
    "record": "/tmp/fa_open_yap_results/qwen3-fa+mfa/gpu_0/blocks/b00006/unresolved.json"
  },
  {
    "id": "9f693c4db5b895f18fe1c67d_c0_105660960_106809360",
    "text": "it's some gaga gaga came with twenty one i just bought my thirty six for a block twenty one ain't got thirty eight that fool ain't near twenty three my nigga if the boy did this it look like a combo my twenty six looks like a hundred some brothers sent us that for another and that's a brand",
    "zero_words": 59,
    "source_audio": "/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing/open_yap_1k/conv_9267fbe11895/speaker_b.wav",
    "record": "/tmp/fa_open_yap_results/qwen3-fa+mfa/gpu_2/blocks/b00003/unresolved.json"
  },
  {
    "id": "f27e49dee2dd9e0dc4c1784f_c0_106545120_107718960",
    "text": "do it for the club like cool that twitching my lane like jet jet i'm hot i'm tired to run down when he got i said he love me like i'm bigger your fuck back get it's then i dropped took a little break had to get to the month now he's on back now this is a background check my background i remember niggas ain't had no hope but i'm a mac",
    "zero_words": 54,
    "source_audio": "/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing/open_yap_1k/conv_f32324299a39/speaker_b.wav",
    "record": "/tmp/fa_open_yap_results/qwen3-fa+mfa/gpu_0/blocks/b00006/unresolved.json"
  }
]
```

实际完整回退例子（原始录音秒坐标）：

```json
{
  "id": "b0b80b393fc0e4b538955e4b_c0_1349760_1390080",
  "text": "i was i love",
  "source_audio": "/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing/open_yap_1k/conv_0084a98344ee/speaker_b.wav",
  "backend": "mfa",
  "crop_start": 28.12,
  "crop_end": 28.96,
  "original_words": [
    {
      "text": "i",
      "start": 28.12,
      "end": 28.12,
      "source": "qwen"
    },
    {
      "text": "was",
      "start": 28.36,
      "end": 28.44,
      "source": "qwen"
    },
    {
      "text": "i",
      "start": 28.6,
      "end": 28.68,
      "source": "qwen"
    },
    {
      "text": "love",
      "start": 28.84,
      "end": 28.92,
      "source": "qwen"
    }
  ],
  "replacement_words": [
    {
      "text": "i",
      "start": 28.12,
      "end": 28.14999999932945,
      "source": "mfa"
    },
    {
      "text": "was",
      "start": 28.339999998807908,
      "end": 28.439999992847444,
      "source": "mfa"
    },
    {
      "text": "i",
      "start": 28.809999997615815,
      "end": 28.84000002861023,
      "source": "mfa"
    },
    {
      "text": "love",
      "start": 28.84000002861023,
      "end": 28.930000002384187,
      "source": "mfa"
    }
  ],
  "boundary_limits": {
    "left": 28.12,
    "right": 28.96
  },
  "record": "/tmp/fa_open_yap_results/qwen3-fa+mfa/gpu_0/blocks/b00000/mfa/repairs.jsonl",
  "task_id": "s000002"
}
```

### Qwen＋Nyra

| 阶段 | 总词数 | 零时长词 | 零时长率 | 含零词句子 | 含零词片段 | 最长连续零词 | 全部异常词率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen | 10,301,121 | 535,259 | 5.19612% | 303,060 | 303,054 | 85 | 5.21130% |
| final | 10,301,121 | 3,361 | 0.03263% | 683 | 680 | 85 | 0.04063% |

- 推理失败 0 个片段；非空文本无输出 0 个片段。
- 最终各类异常：`{"out_of_bounds": 936, "zero_duration": 3361}`。
- 请求及接受数：`{"nyra_requested": 303055, "nyra_accepted": 302374}`。
- 四卡阶段耗时之和（秒）：`{"nyra_model_load_seconds": 20.3883193298243, "nyra_inference_seconds": 6592.5036230382975, "qwen_seconds": 8703.037866341881, "repair_seconds": 7115.77539938665}`。
- 四卡 worker 总耗时 16373.890 秒，归一化工作量 RTF 0.00410133；它不是四卡吞吐 RTF。
- Qwen 输出与第一组逐卡 SHA-256 一致：`True`。

仍含零时长的片段共 680 个；回退拒绝原因前五项：`{"Nyra skipped input tokens: ['1']": 19, "alignment failed at all beams (5 tokens, 130 frames)": 14, "alignment failed at all beams (6 tokens, 146 frames)": 13, "Nyra skipped input tokens: ['2018']": 10, "Nyra skipped input tokens: ['180']": 9}`。

另有未触发零时长回退的原始异常记录：`{"qwen_error_without_zero_trigger": 778}`；它们不计入后端回退拒绝数。

零时长字词最多的残留片段（完整原因和原时间戳在单独未解决记录中）：

```json
[
  {
    "id": "f27e49dee2dd9e0dc4c1784f_c0_100802640_101927040",
    "text": "cash it ain't i'm fuckin safety if she wanna fuck me she gotta pay you go dc just to give me a cut cut that bitch all she ain't making me cut cut that bitch all she ain't making rock it me park it some fuck it we're arms up now i flex keep all your clothes i just want me to net promote pay me i don't want me no check cash money i don't want me no check cash money like i'm you",
    "zero_words": 85,
    "source_audio": "/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing/open_yap_1k/conv_f32324299a39/speaker_b.wav",
    "record": "/tmp/fa_open_yap_results/qwen3-fa+nyra/gpu_0/blocks/b00006/unresolved.json"
  },
  {
    "id": "9f693c4db5b895f18fe1c67d_c0_105660960_106809360",
    "text": "it's some gaga gaga came with twenty one i just bought my thirty six for a block twenty one ain't got thirty eight that fool ain't near twenty three my nigga if the boy did this it look like a combo my twenty six looks like a hundred some brothers sent us that for another and that's a brand",
    "zero_words": 59,
    "source_audio": "/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing/open_yap_1k/conv_9267fbe11895/speaker_b.wav",
    "record": "/tmp/fa_open_yap_results/qwen3-fa+nyra/gpu_2/blocks/b00003/unresolved.json"
  },
  {
    "id": "ce14e09ef4992c7baddc192f_c0_14344800_14647200",
    "text": "哈哈哈哈哈 the 哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈",
    "zero_words": 37,
    "source_audio": "/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing/open_yap_1k/conv_658ab4a69158/speaker_a.wav",
    "record": "/tmp/fa_open_yap_results/qwen3-fa+nyra/gpu_0/blocks/b00002/unresolved.json"
  }
]
```

实际完整回退例子（原始录音秒坐标）：

```json
{
  "id": "b0b80b393fc0e4b538955e4b_c0_1349760_1390080",
  "text": "i was i love",
  "source_audio": "/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing/open_yap_1k/conv_0084a98344ee/speaker_b.wav",
  "backend": "nyra",
  "crop_start": 28.12,
  "crop_end": 28.96,
  "original_words": [
    {
      "text": "i",
      "start": 28.12,
      "end": 28.12,
      "source": "qwen"
    },
    {
      "text": "was",
      "start": 28.36,
      "end": 28.44,
      "source": "qwen"
    },
    {
      "text": "i",
      "start": 28.6,
      "end": 28.68,
      "source": "qwen"
    },
    {
      "text": "love",
      "start": 28.84,
      "end": 28.92,
      "source": "qwen"
    }
  ],
  "replacement_words": [
    {
      "text": "i",
      "start": 28.12,
      "end": 28.25,
      "source": "nyra"
    },
    {
      "text": "was",
      "start": 28.290000000000003,
      "end": 28.46,
      "source": "nyra"
    },
    {
      "text": "i",
      "start": 28.46,
      "end": 28.490000000000002,
      "source": "nyra"
    },
    {
      "text": "love",
      "start": 28.77,
      "end": 28.94,
      "source": "nyra"
    }
  ],
  "boundary_limits": {
    "left": 28.12,
    "right": 28.96
  },
  "record": "/tmp/fa_open_yap_results/qwen3-fa+nyra/gpu_0/blocks/b00000/nyra/repairs.jsonl",
  "task_id": "s000002"
}
```

## 配置、数据适配和记录位置

- 设计与全部代码入口：[open_yap_streaming_design.md](open_yap_streaming_design.md)。Qwen batch 128，块大小 32,768，块内按时长分桶，结果恢复 manifest 顺序；每卡 MFA 32 jobs。Qwen 和 Nyra 在一组实验内常驻，跨块复用。
- 测量在共享主机进行，期间观察到其他 CPU/I/O 数据转换作业；RTF 为该实际运行环境的结果。
- ROVER 去标点片段缺少可靠自然句界时，回退整个已有 VAD/ROVER 片段；不改写融合文本。缺失的单系统 ASR hypothesis 允许为 null，266 条相关数据已完整补回。
- 校验包括整句/片段原子替换、正时长与邻词边缘、文本/词数不变、未解决记录覆盖。各块 `validation.json` 保存实际校验结果。
- 每组目录含 `summary.json`、`results.json` 和四个 GPU 目录。`results.json` 列出全部最终预测分片；块内保留 Qwen、中间 MFA、最终预测及 `unresolved.json`。
- 大结果实际在 `/tmp/fa_open_yap_results/`，项目 `results/open_yap_1k_rover_4gpu` 是访问链接。输入大索引也在本地磁盘。原因是共享项目盘出现 ENOSPC；这些本机文件不是共享盘持久归档，迁移前请保留所在机器。
- 完整最终预测与未解决记录另压缩导出到项目 `results/open_yap_1k_summary/predictions/`，逐组实际存储路径与复制状态见下面的归档表。
- 小体积的汇总、输入元数据和真实例子保存在项目 `results/open_yap_1k_summary/`，代码保存在 `src/`；结果根目录还保存了代码快照及 SHA-256。
- 最终归档核验代码为 `src/verify_rover_outputs.py`；[final_audit.json](../results/open_yap_1k_summary/final_audit.json) 保存覆盖回执、路由校验、代码快照与归档完整性复核。
- `results/open_yap_4gpu_suite.log` 记录三组先后顺序；`status.json` 为当前进度，`suite_summary.json` 为完成组的机器可读汇总。

## 最终压缩预测导出

以下是完整最终预测的 gzip JSONL。该步骤在三组基准全部结束后执行，单独记录压缩和共享盘复制耗时，未混入主表的 FA 墙钟。共享复制若失败，会保留本地完整压缩文件并记录错误。

| 方案 | 片段数 | 压缩字节 | 导出秒数 | 加上导出后的 RTF |
|---|---:|---:|---:|---:|
| qwen+mfa+nyra | 919,459 | 271,634,380 | 50.818 | 0.00135310 |

文件：`/mnt/data/L202500721/stevezhuang/fa/results/open_yap_1k_summary/predictions/qwen3-fa+mfa+nyra.predictions.jsonl.gz`；SHA-256：`0d8375bb524acee7c3ac1724604298819f661df091cfb4324a13eb97da2a8312`；共享复制错误：`None`。

导出核验：`{"status": "passed", "manifest_coverage": "all rows, order, IDs, source samples and unchanged text", "words": 10301121, "zero_duration_words": 186, "blocks": 30}`。

单独的未解决记录：`/mnt/data/L202500721/stevezhuang/fa/results/open_yap_1k_summary/predictions/qwen3-fa+mfa+nyra.unresolved.jsonl.gz`；数量：`{'rejected_sentences': 15, 'residual_chunks': 792, 'backend_rejected_groups': 15, 'nontriggered_issue_records': 0}`。

| qwen+mfa | 919,459 | 271,624,980 | 51.288 | 0.00130934 |

文件：`/mnt/data/L202500721/stevezhuang/fa/results/open_yap_1k_summary/predictions/qwen3-fa+mfa.predictions.jsonl.gz`；SHA-256：`7d3f542fd8cd05298c189c8faf4f626f7059bbcd48c8215892efeef253f34b09`；共享复制错误：`None`。

导出核验：`{"status": "passed", "manifest_coverage": "all rows, order, IDs, source samples and unchanged text", "words": 10301121, "zero_duration_words": 747, "blocks": 30}`。

单独的未解决记录：`/mnt/data/L202500721/stevezhuang/fa/results/open_yap_1k_summary/predictions/qwen3-fa+mfa.unresolved.jsonl.gz`；数量：`{'rejected_sentences': 845, 'residual_chunks': 844, 'backend_rejected_groups': 67, 'nontriggered_issue_records': 778}`。

| qwen+nyra | 919,459 | 247,901,831 | 46.367 | 0.00104596 |

文件：`/mnt/data/L202500721/stevezhuang/fa/results/open_yap_1k_summary/predictions/qwen3-fa+nyra.predictions.jsonl.gz`；SHA-256：`756a99b7398378329e0bc49aca850b16d4b3f676bcd16fbd2ae0b87d0f6cae3e`；共享复制错误：`None`。

导出核验：`{"status": "passed", "manifest_coverage": "all rows, order, IDs, source samples and unchanged text", "words": 10301121, "zero_duration_words": 3361, "blocks": 30}`。

单独的未解决记录：`/mnt/data/L202500721/stevezhuang/fa/results/open_yap_1k_summary/predictions/qwen3-fa+nyra.unresolved.jsonl.gz`；数量：`{'rejected_sentences': 1459, 'residual_chunks': 1458, 'backend_rejected_groups': 681, 'nontriggered_issue_records': 778}`。

归档兼容字段 `rejected_sentences` 含后端拒绝组及没有零时长触发的原始异常记录；实际后端拒绝数与未触发记录数已另外列出。
