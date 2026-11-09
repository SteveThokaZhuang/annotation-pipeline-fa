# Qwen FA 与整句 MFA 回退实现

最终策略：对 Qwen 每个输出字词判断时长；只要某个字词 `abs(end-start) <= 1e-7`，就将它所在的**完整句子全部交给 MFA**。只有整句的 MFA 输出及替换边界都合法，才替换该句全部字词。失败保留原结果并在独立 JSON 记录。

## 入口与复用

- `src/qwen-fa/process.py`：按要求提供的纯 Qwen 入口，调用 `src/qwen3-fa/process.py`。
- `src/qwen3-fa/core.py`：复用现有 Qwen 模型加载、public/native 后端和录音时间偏移；修正空批次返回结构、缺省 ID 和空语言字段。新增 `strict=False` 让 process 保留原始异常作逐词统计，原有调用默认仍严格校验。
- `src/qwen3-fa/process.py`：公共运行流程、单条失败隔离、逐字词校验、错误率、RTF 和输出。
- `src/qwen3-fa/sentences.py`：按 ASR 句末标点划句，再使用同一 Qwen tokenizer 验证词元覆盖；若拆句改变分词，保守地把整个 ASR chunk 当上下文，避免猜测索引。
- `src/qwen3-fa+mfa/process.py`、`repair.py`：独立加载 Qwen 后执行整句 MFA 回退。
- `src/prepare_candor.py`：只复用已有 ASR，读取指定原始 MP3，统一为 16 kHz 单声道 PCM 切片。
- `src/setup_mfa_models.py`：从官方 `MontrealCorpusTools/english_us_arpa` 仓库下载固定 revision 的声学模型和配套词典，保存来源与文件 SHA-256。
- `src/compare_results.py`：校验两次 Qwen 原始输出完全一致，生成比较 JSON、两份实际例子和实验记录。

没有重跑 ASR。已有 `annotation_pipeline` 的源代码、标注和录音不修改。

## 整句回退步骤

1. 先完成 Qwen process，保存 `qwen_predictions.jsonl`；逐字词检查零时长、负时长、非有限值、越界、时间倒退、重叠，保留真实异常，不用最小时长填充。
2. 将零时长词映射到完整 ASR 句子。一个句子多个坏词只回退一次；相邻触发句合并为一个包含完整句子的区域，一起对齐，避免在连续坏区之间制造边界。
3. 从原音频切片中读取区域，按 Qwen 整句时间范围裁剪，左右各加 0.35 秒上下文，并限制在原切片内。MFA `.lab` 包含这些完整句子的全部 Qwen 字词，保持输入文本覆盖一致。
4. 将所有受影响区域批量交给一次 MFA `align`，输出原始 JSON。MFA 使用 CPU，Qwen 按指定 GPU 执行。
5. 对 MFA 词表做字符归一化比较，要求文本完整、顺序一致、没有 OOV/漏词。允许一个 Qwen 词对应连续多个 MFA 片段（例如 contraction）；聚合其首尾时间，禁止把单个 MFA 区间等分以捏造多个词时长。
6. 恢复录音时间：`MFA 裁剪内时间 + 裁剪的录音绝对起点`。对整句所有字词限制边缘：左边不早于未修改前词的结束，右边不晚于未修改后词的起点，且不越出切片。整句每个字词必须严格为正时长、单调且不重叠。
7. 验证通过才一次性替换整个句子/句组，所有替换词标记 `source=mfa`；否则整句/句组保持 Qwen 输出，记录原词、完整句子、裁剪范围和拒绝原因。最后全量复查并生成 `unresolved.json`。

零时长是 MFA 触发条件；其余结构错误同样统计并记录，但不偷偷扩大本次回退策略。ASR 的标点及 chunk 边缘语义断句仍取决于上游标注，不能由本流程证明其正确性。

## 输入与输出

公共入口支持 JSONL 的 `{id, audio, text, duration, chunk_start, chunk_end, language}`，也支持原 `core.py` 的 recording/chunks 结构和 JSON 数组。`audio` 必须是已经裁好的片段音频；`chunk_start`、`chunk_end` 为其在原录音中的位置，不能把整段录音误当成切片。相对路径按 manifest 所在目录解析。`language` 缺失或为空时使用 `--language`，默认 English。

两方案输出分别为 `results/qwen-fa/` 和 `results/qwen3-fa+mfa/`：

- `predictions.jsonl`：最终全部字词、录音时间戳、来源、逐词异常及句子范围。
- `qwen_predictions.jsonl`：未做 MFA 替换的 Qwen 结果。
- `failures.jsonl`：推理异常；批次出错后逐条重试，避免一条失败连累整批。
- `error_stats.json`：替换前后零时长词率、去重异常词率、句子错误率、片段错误率及各类计数。
- `runtime.json`：总墙钟、输入和加载时间、Qwen 时间、可选回退和输出时间、RTF 及排除加载后的处理 RTF。
- `metadata.json`：命令、模型、manifest 哈希、GPU 映射、Python 环境及时间坐标说明。
- `unresolved.json`：未修复项。混合版含拒绝句子细节和所有残留坏片段。
- `example.json`：比较脚本抽取的同一个完整句子的实际输出。

混合版另存 `mfa/tasks.jsonl`、`repairs.jsonl`、`runtime.json`、`mfa.log` 和原始 `mfa/aligned/*.json`，可追溯每次接受或拒绝。

错误率为时间戳**结构异常率**，不等同于强制对齐准确率。字词分母为全部输出词元；句子分母为全部非空句子；片段分母包括静音片段。推理失败、非空文本却无输出另外计数并计入片段错误。没有人工词级真值，无法报告 MAE 或对齐边界准确率。

## 本次环境与复现命令

本机默认 Python 3.9 没有 Torch，复用已有 Qwen 环境：

```bash
QWEN_PYTHON=/mnt/data/L202500293/obed/envs/qwenasr/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

# 仅首次需要下载；官方固定模型版本与 SHA-256 见 models/mfa-english/provenance.json
python src/setup_mfa_models.py

# 本地 MFA 环境移动后，脚本 shebang 指向旧目录。
# 代码使用显式 models/mfa/bin/python，子进程自行设置其 PATH 和 LD_LIBRARY_PATH。
# 准备切片时 ffmpeg 同样来自该环境：
PATH="$PWD/models/mfa/bin:$PATH" \
LD_LIBRARY_PATH="$PWD/models/mfa/lib:${LD_LIBRARY_PATH:-}" \
python src/prepare_candor.py \
  --asr ../annotation_pipeline/results/v4/candor_v4_0_1/asr.jsonl \
  --audio-dir /mnt/data/L202500721/stevezhuang/candor \
  --output data/candor_full

# 两个终端分别启动，输出目录必须不存在，以免覆盖既有实验。
CUDA_VISIBLE_DEVICES=0 "$QWEN_PYTHON" -u src/qwen-fa/process.py \
  --input data/candor_full/manifest.jsonl --output results/qwen-fa \
  --device cuda:0 --batch-size 4 > results/qwen-fa.log 2>&1

CUDA_VISIBLE_DEVICES=1 "$QWEN_PYTHON" -u src/qwen3-fa+mfa/process.py \
  --input data/candor_full/manifest.jsonl --output results/qwen3-fa+mfa \
  --device cuda:0 --batch-size 4 > results/qwen3-fa+mfa.log 2>&1

"$QWEN_PYTHON" -m unittest discover -s tests -v
python src/compare_results.py
```

重复实验时为 `--output` 指定新目录，数据准备可直接复用现有 manifest。比较其他目录可传 `--baseline`、`--hybrid` 和 `--report`。

本地 `models/mfa` 是已有可执行环境，英文权重/词典原先缺失，因此新增固定下载。模型来自 [官方 English US ARPA 仓库](https://huggingface.co/MontrealCorpusTools/english_us_arpa/tree/9c8d5d8e1dd5d9cf188a5571203de0e09ca6bede)，参考 [MFA 官方入门说明](https://montreal-forced-aligner.readthedocs.io/en/v3.4.3/first_steps/index.html)。两方案的 FA 计时都不包含首次模型下载和共用切片准备；后者单独写入 `data/candor_full/preparation.json`。

## 新增 Nyra 直接回退与三级回退

新增两个独立入口，共享 `src/qwen3-fa/nyra_repair.py`：

- `src/qwen3-fa+nyra/process.py`：Qwen 逐词检查 → 零时长所在完整句子/连续句组交给 Nyra → 验收后整句替换。
- `src/qwen3-fa+mfa+nyra/process.py`：Qwen → 完整执行 MFA 整句回退 → 只把 MFA 验收未通过的原句组交给 Nyra。已经接受的 MFA 句子不重复跑 Nyra。Nyra 仍未通过则保留该句的 Qwen 原结果并记录。

MFA 模块的 `prepare_sentence_tasks` 被抽为共享函数。新模式的触发词、句子索引、裁剪音频、0.35 秒余量及边界验收与原模式相同。三级模式直接复用本次 MFA 拒绝的任务与裁剪音频，避免改变上下文导致比较失真。`map_mfa_words` 支持来源参数，两后端使用同一文本覆盖检查和 `splice_sentence`。

Nyra 采用本地已有 0.2.2 推理代码和官方 standard 英文模型，固定 revision `9c8e724072261c30641a91ca365ce4f7b4ae7f83`，FP32。独立的 `um` / `uh` token 转为 `[UM]` / `[UH]`，验收输出时恢复输入文本。`skipped`、文本覆盖缺失、词序变化、零/负时长、边界裁剪造成无时长，均拒绝整个候选句组，不拆词等分时间。

本地原 `models/nyra` 是依赖运行库，本轮重新下载缺失的权重，放在 `models/nyra-forced-aligner/checkpoints/`，包含完整 `wavlm/`。固定来源、文件大小与 SHA-256 写入 `download-provenance.json`。推理时启用离线模式，避免首次下载混入 RTF。官方接口及转录约定参考 [Nyra 官方代码](https://github.com/nyrahealth/nyra-forced-aligner)；权重与许可随 [官方模型](https://huggingface.co/nyralabs/nyra_forced_aligner_en/tree/9c8e724072261c30641a91ca365ce4f7b4ae7f83) 保存在本地。

Nyra 需要 espeak-ng 为词典外词生成发音。本轮通过 `src/setup_nyra_espeak.py` 将 Ubuntu 的 libespeak-ng1、espeak-ng-data、libpcaudio0、libsonic0 解包到 `models/nyra-espeak/`，包哈希见其 `provenance.json`；不修改系统安装。音频库依赖使用现有 `models/mfa/lib/libpulse-simple.so.0`，与 MFA 对齐调用无关。代码按绝对路径加载依赖并指定本地 espeak 数据目录。

```bash
python src/setup_nyra_models.py
python src/setup_nyra_espeak.py

QWEN_PYTHON=/mnt/data/L202500293/obed/envs/qwenasr/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p results/candor_four_modes

# 本轮先重新实测原两种模式，各自在一个终端启动。
CUDA_VISIBLE_DEVICES=0 "$QWEN_PYTHON" -u src/qwen-fa/process.py \
  --input data/candor_full/manifest.jsonl --output results/candor_four_modes/qwen-fa \
  --device cuda:0 --batch-size 4 > results/candor_four_modes/qwen-fa.log 2>&1

CUDA_VISIBLE_DEVICES=1 "$QWEN_PYTHON" -u src/qwen3-fa+mfa/process.py \
  --input data/candor_full/manifest.jsonl --output results/candor_four_modes/qwen3-fa+mfa \
  --device cuda:0 --batch-size 4 > results/candor_four_modes/qwen3-fa+mfa.log 2>&1

# 原两种模式结束后，分别在 GPU 0/1 启动两个新模式。
CUDA_VISIBLE_DEVICES=0 "$QWEN_PYTHON" -u src/qwen3-fa+nyra/process.py \
  --input data/candor_full/manifest.jsonl --output results/candor_four_modes/qwen3-fa+nyra \
  --device cuda:0 --batch-size 4 > results/candor_four_modes/qwen3-fa+nyra.log 2>&1

CUDA_VISIBLE_DEVICES=1 "$QWEN_PYTHON" -u src/qwen3-fa+mfa+nyra/process.py \
  --input data/candor_full/manifest.jsonl --output results/candor_four_modes/qwen3-fa+mfa+nyra \
  --device cuda:0 --batch-size 4 > results/candor_four_modes/qwen3-fa+mfa+nyra.log 2>&1

"$QWEN_PYTHON" -m unittest discover -s tests -v
python src/compare_four_modes.py
```

入口可用 `--nyra-checkpoint`、`--nyra-runtime`、`--nyra-espeak` 指定其他本地路径；MFA 参数与原模式一致。目标目录必须不存在，重复实验用新目录并将根路径传给 `compare_four_modes.py --root`。

四方案实验报告为 `docs/candor_four_modes.md`，汇总为 `results/candor_four_modes/comparison.json`。每个模式独立保存最终输出、原始 Qwen、错误率、RTF 和未解决 JSON；Nyra 的原始裁剪内结果及接受/拒绝记录保存在 `nyra/`，并保留所有未通过验收的尝试。三级模式额外保存 `mfa_predictions.jsonl`、`mfa_error_stats.json`、`mfa/unresolved.json`。

计时规则：端到端 RTF 包含 Qwen 和 Nyra 模型加载、MFA 完整生命周期、全部成功与失败尝试；`nyra_model_load_seconds` 包含惰性 WavLM 加载和同步；`processing_rtf` 排除 Qwen 输入/加载和 Nyra 显式模型加载，但包含首条推理、Nyra JIT 编译、MFA CLI 启动及文件处理。本轮四方案都从头运行，旧双方案结果不覆盖。


# Updates

## 格式统一

按照hlt-fd数据格式重新match
```json
{
  "conversation_id": "personas-5260_speaking_style-16_repeat-0",
  "user_audio_path": "/hbt_share/hbt/zxy_workspace/projects/MOSS-TTS/audio_pause_compare/workdirs/lychee_v2pause_moss/synthesis_pause0p8_v2/force_align_v1/audio/user_full/shard_00062/personas-5260_speaking_style-16_repeat-0.wav",
  "audio_duration_sec": 16.08,
  "user_audio_force_align": {
    "timestamps": [[0.0, 0.32], [0.72, 0.8], [0.8, 1.04], [1.04, 1.12], [1.12, 1.28], [1.28, 1.52], [1.52, 1.6], [1.6, 1.84], [2.64, 2.8], [2.88, 2.96], [2.96, 3.12], [3.12, 3.28], [3.28, 3.6], [3.76, 3.84], [3.84, 4.16], [4.16, 4.4], [7.92, 8.24], [8.24, 8.32], [8.32, 8.64], [8.64, 8.8]],
    "words": ["嗯", "我", "其", "实", "是", "吃", "鱼", "的", "特", "别", "是", "寿", "司", "我", "超", "爱", "真", "的", "假", "的"]
  },
  "turns": [
    {
      "user": {
        "start": 0.0,
        "end": 4.48,
        "text": "嗯，我其实是吃鱼的，特别是寿司，我超爱！"
      },
      "assistant": {
        "start": 4.88,
        "end": 9.76,
        "text": "哇，寿司！那真是太棒了，我也很喜欢！你最喜欢哪种寿司呢？",
        "clip_path": "/hbt_share/hbt/zxy_workspace/projects/MOSS-TTS/audio_pause_compare/workdirs/lychee_v2pause_moss/synthesis_pause0p8_v2/force_align_v1/final_format_bc_aug15_pause_clean_normalized_merged_audio/audio/assistant_merged_interrupted/shard_00141/personas-5260_speaking_style-16_repeat-0/turn_0000_assistant_merged.wav",
        "force_align": {
          "timestamps": [[0.0, 0.24], [0.24, 0.4], [0.4, 0.56], [0.88, 1.2], [1.2, 1.28], [1.28, 1.36], [1.36, 1.6], [1.6, 1.76], [1.76, 1.92], [2.24, 2.4], [2.4, 2.56], [2.56, 2.72], [2.72, 2.8], [2.8, 3.04], [3.52, 3.68], [3.68, 3.92], [3.92, 4.0], [4.0, 4.16], [4.16, 4.24], [4.24, 4.4], [4.4, 4.56], [4.56, 4.64], [4.64, 4.8]],
          "words": ["哇", "寿", "司", "那", "真", "是", "太", "棒", "了", "我", "也", "很", "喜", "欢", "你", "最", "喜", "欢", "哪", "种", "寿", "司", "呢"]
        },
        "speech_token": [5683, 740, 3647, 4400, 4499, 4516, 5057, 5759, 6140, 5834, 4887, 4131, 4191, 5073, 4579, 4569, 2218, 4509, 4482, 2214, 2188, 2188, 9, 156, 402, 321, 2562, 4496, 4469, 2293, 656, 6033, 1939, 4085, 6488, 4227, 2490, 4406, 5012, 6544, 6013, 512, 5615, 4424, 4420, 1800, 4400, 4607, 4488, 2188, 2188, 2188, 2188, 2188, 2243, 740, 1460, 1469, 1469, 5843, 5630, 1433, 2144, 5286, 4431, 4408, 6042, 2178, 5697, 3565, 1631, 5116, 2945, 5366, 4552, 4594, 2379, 2188, 2189, 2188, 2188, 2188, 2188, 2188, 2188, 2188, 2188, 1, 102, 237, 911, 911, 3836, 1311, 2754, 3950, 6292, 3836, 5293, 4807, 3809, 2972, 4874, 3129, 51, 5283, 5807, 1393, 4166, 1505, 3770, 5759, 6149, 5843, 5589, 6318, 6144, 1374, 2672, 2321, 4513, 28]
      },
      "is_ai_been_interrupted": true,
      "user_backchannel": [],
      "ai_backchannel": [
        {
          "clip_path": "/hbt_share/hbt/zxy_workspace/projects/MOSS-TTS/audio_pause_compare/workdirs/lychee_v2pause_moss/synthesis_pause0p8_v2/force_align_v1/audio/backchannels/bc_aug15_normalized/personas-5260_speaking_style-16_repeat-0/turn_0000_assistant_bc_00.wav",
          "end_time": 2.84,
          "force_align": null,
          "start_time": 2.04,
          "stoken": null,
          "text": "嗯。",
          "speech_token": [5638, 487, 243, 2187, 2278, 825, 1617, 1590, 1617, 1617, 1590, 3009, 3540, 1362, 2325, 2244, 2323, 246, 2430, 2431]
        }
      ],
      "side_talk": []
    },
    {
      "user": {
        "start": 7.92,
        "end": 9.04,
        "text": "真的假的？"
      },
      "assistant": {
        "start": 9.44,
        "end": 16.08,
        "text": "真的呀，我特别喜欢吃三文鱼寿司，还有一些加州卷，你呢？",
        "clip_path": "/hbt_share/hbt/zxy_workspace/projects/MOSS-TTS/audio_pause_compare/workdirs/lychee_v2pause_moss/synthesis_pause0p8_v2/force_align_v1/audio/assistant_clips/shard_00062/personas-5260_speaking_style-16_repeat-0/turn_0001_asst_Main.wav",
        "force_align": {
          "timestamps": [[0.0, 0.24], [0.24, 0.32], [0.32, 0.56], [0.96, 1.12], [1.12, 1.28], [1.28, 1.36], [1.36, 1.52], [1.52, 1.68], [1.68, 2.0], [2.16, 2.48], [2.48, 2.56], [2.56, 2.72], [2.72, 2.88], [2.88, 3.2], [3.52, 3.76], [3.76, 3.92], [3.92, 4.0], [4.0, 4.32], [4.48, 4.72], [4.72, 4.96], [4.96, 5.04], [6.0, 6.16], [6.16, 6.56]],
          "words": ["真", "的", "呀", "我", "特", "别", "喜", "欢", "吃", "三", "文", "鱼", "寿", "司", "还", "有", "一", "些", "加", "州", "卷", "你", "呢"]
        },
        "speech_token": [4705, 656, 5030, 6226, 5802, 1212, 564, 1752, 6015, 3809, 5283, 4561, 4498, 4514, 4431, 3, 0, 1, 2244, 4512, 4593, 4590, 2322, 2215, 749, 1460, 5834, 4640, 3903, 303, 5889, 6127, 512, 6406, 6301, 3969, 4781, 5915, 2972, 5351, 2667, 1695, 971, 5030, 5759, 1925, 4112, 6298, 6299, 6037, 6019, 5300, 4593, 2322, 4509, 4944, 4860, 4137, 4831, 2400, 6477, 1778, 1745, 4587, 1928, 1559, 743, 1547, 4084, 5030, 6362, 5861, 5834, 5349, 5589, 1740, 6378, 5739, 4479, 4488, 4485, 3, 1, 0, 0, 0, 1, 1, 2214, 4432, 4514, 4472, 4558, 6005, 5995, 5836, 3647, 1462, 1658, 1901, 1648, 4806, 3565, 1919, 6293, 6273, 6258, 5277, 4590, 2325, 30, 330, 651, 5751, 1865, 5553, 2807, 4913, 4994, 1394, 6488, 6374, 3458, 1743, 650, 3731, 4385, 4472, 5309, 3086, 4, 1, 0, 243, 243, 244, 244, 244, 244, 244, 244, 244, 244, 244, 244, 244, 0, 0, 1, 3, 120, 963, 1667, 1640, 6031, 321, 5721, 4940, 4931, 4940, 4670, 4526, 4432, 274, 1, 1]
      },
      "is_ai_been_interrupted": false,
      "user_backchannel": [
        {
          "clip_path": "/hbt_share/hbt/zxy_workspace/projects/MOSS-TTS/audio_pause_compare/workdirs/lychee_v2pause_moss/synthesis_pause0p8_v2/force_align_v1/audio/backchannels/bc_aug15_normalized/personas-5260_speaking_style-16_repeat-0/turn_0001_user_bc_00.wav",
          "end_time": 15.48,
          "start_time": 14.68,
          "text": "嗯。",
          "speech_token": [5046, 491, 271, 28, 1, 2187, 4011, 2022, 1779, 798, 795, 783, 54, 28, 28, 28, 31, 31, 272, 272]
        }
      ],
      "ai_backchannel": [],
      "side_talk": []
    }
  ],
  "speech_token": [4461, 2277, 2100, 2019, 1776, 1533, 1533, 36, 0, 0, 1, 27, 54, 54, 27, 28, 2, 11, 6077, 5888, 5857, 420, 4941, 4592, 1892, 4348, 5543, 5516, 4110, 4354, 5057, 4327, 3850, 656, 5084, 4841, 2086, 1435, 1793, 1466, 1625, 1176, 3489, 4859, 2591, 4525, 2241, 1, 1, 1, 1, 1, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 4, 3, 492, 492, 2409, 4516, 3459, 5655, 989, 1223, 6059, 6040, 5570, 5030, 2167, 5543, 5030, 6394, 5888, 3649, 4860, 4860, 4131, 1599, 4272, 3543, 30, 0, 1, 1, 2, 11, 1730, 5870, 5928, 656, 5057, 4510, 4496, 4937, 5654, 5653, 5010, 2666, 4583, 4583, 5301, 2403, 109, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 412, 5030, 5057, 4598, 4110, 6306, 402, 561, 1227, 6306, 906, 2754, 5509, 6059, 4826, 2363, 4550, 3093, 519, 4614, 4615, 4497, 2298, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  "data_source": "lychee_zh_v2pause_moss.converted_data.speech_tokens"
}
```

## 2026-11-09：格式落地与 chunk 粒度

实际检查 `data/candor_full/manifest.jsonl` 及其原始 ASR 后确认：Candor 当前按 **120 秒定长 chunk** 切分，不是 turn；原始 ASR 只有 `chunk_id/start_s/end_s/text/language`，没有说话人、角色或 turn 边界。OpenYap ROVER 也是已有 VAD/ROVER 区间，不能视为可靠的对话 turn。句末标点仅用于整句 FA 回退，不代表换人。

上面的 HLT-FD 会话格式仍作为有真实 turn/角色数据时的目标格式：`user_audio_force_align` 对应 user 完整音轨，时间戳以该音轨为原点；`assistant.force_align` 对应 `clip_path`，以 clip 为原点。生成完整 `turns` 前，需要可靠的说话人、角色和边界，不能通过定长 chunk 或标点推断 user/assistant。

当前数据先采用明确的 **`hlt-fd-chunk-v1`** 变体，统一核心 FA 格式为 `force_align: {"timestamps": [[start, end], ...], "words": [word, ...]}`，两数组逐项对应。保留中英文原始输出词元、词序及数值精度；零时长、越界等异常原样保留并标记 `alignment_status: "needs_review"`，序列化不修改对齐结果。

每个输出保留 `conversation_id`、唯一 `chunk_id`、`text`、`audio_path`、`audio_duration_sec`、`audio_start_sec/audio_end_sec`、原录音路径及 `recording_start_sec/recording_end_sec`。FA 时间戳相对于 `audio_path[audio_start_sec:audio_end_sec]` 选定区间的起点：Candor 已裁出的 WAV 从 0 秒起；ROVER 若引用整段录音，则明确给出实际采样对应的裁剪区间。`turns: []` 与 `turns_status: "not_available"` 联合表示缺少 turn 标注，不表示没有语音；不生成没有依据的 user/assistant、speech token、打断或 backchannel 信息。

保存入口统一为 `src/qwen3-fa/prediction_format.py`，纯 Qwen、三个回退入口及生产流式 worker 共用：

- `predictions.jsonl`：新格式主预测。
- `alignment_predictions.jsonl`：保留原录音坐标、词级来源、句子索引及异常的完整审计结果。
- `annotation_schema.json`：格式名称、时间原点和 turn 缺失语义。
- `qwen_predictions.jsonl`、`mfa_predictions.jsonl`、回退原始结果及未解决记录继续服务修复审计。

比较程序优先读 `alignment_predictions.jsonl`，旧实验目录仍可读取原来的 `predictions.jsonl`。旧实验结果不批量改写；生产归档支持新旧两种布局，并核对标准输出与审计输出的词元和时间偏移。

灰度入口：`python src/smoke_test_annotations.py`。按 manifest 顺序选择前 3 条 **chunk**，不是前 3 个完整录音；选取连续两次计算负载为 0% 且空闲显存至少 32 GiB 的 GPU，先运行 Qwen＋MFA，再运行 Qwen＋MFA＋Nyra。输入、选卡快照、逐模式结果、日志及格式/整句替换核验保存在 `results/smoke-test/`，记录见 [smoke_test_annotation_format.md](smoke_test_annotation_format.md)。

已完成的灰度可用 `python src/smoke_test_annotations.py --summarize-only` 重新核验和生成两方案报告，不触发推理。

单条 OpenYap 音轨灰度使用 `src/smoke_test_rover_audio.py --audio <WAV>`，从现有四卡 ROVER manifest 提取该音轨的原始 ASR 区间，只运行 Qwen＋MFA＋Nyra 一次。输出逐区间新格式及整条音轨 `recording_prediction.json`；后者的时间戳直接对应完整 WAV。实际结果见 [open_yap_single_audio_smoke.md](open_yap_single_audio_smoke.md)。`--finalize-only` 只做格式核验和结果汇总；归档异常时可使用 `--recover-scratch <临时目录>` 恢复已完成且通过验证的推理结果，不重复对齐。

## 2026-11-09 update：新 ASR 灰度与 JSON 文件粒度

Candor 灰度改为将原选定的前 3 条 120 秒 WAV 交给 `Pipeline_github_v5`，执行 ingest、DNSMOS、VAD、增强、Qwen/FunASR、disagreement、按需 FireRed、ROVER；随后依次运行 Qwen＋MFA 和 Qwen＋MFA＋Nyra。不再复用旧 Candor ASR。前次 Candor 灰度结果已删除，speaker_b 结果保留；清理及文件校验记录在 `results/smoke-test/candor_cleanup.json`。本次新结果统一保存于 `results/smoke-test/candor_v5/`。

本次 v5 的 chunk 是 VAD/ASR 区间，仍然不是 turn；120 秒 WAV 是输入文件粒度。增强 WAV 为 48 kHz，而 VAD 原始采样坐标为 16 kHz：FA 读取增强 WAV 时使用 `audio_start_sample/audio_end_sample` 与该 WAV 的真实采样率。原始 VAD 坐标单独保留在审计结果中，不混用两套采样坐标。

保存格式支持两种独立选择，同一份对齐结果还可以同时导出两种：

- `chunk`：`chunk_json/*.json`，每个 VAD/ASR chunk 一个文件。`force_align.timestamps` 相对该 chunk 起点；`audio_start_sec/audio_end_sec` 指定它在实际 WAV 中的位置。
- `audio`：`audio_json/*.json`，每个实际 WAV 一个文件。`annotation_unit: "audio_file"`，FA 时间戳相对完整 WAV 起点，所有 chunk 按文件内起点排列。
- `both`：同时生成上述两个目录，只运行一次 FA。

audio JSON 的 `annotated_segments` 每项必须带 `chunk_id/start/end/text`；`start/end` 相对该 WAV 起点，`text` 保留该 chunk 的完整 ASR 文本。额外的 `word_start_index/word_end_index` 指定此区间对应的 FA 数组切片，右边界不包含。逐 chunk JSON 直接保留顶层 `text`。两个粒度的 FA 核心字段都使用 `force_align: {timestamps, words}`，词元、词序及异常区间原样保留；缺少可靠角色信息时继续使用 `turns: []`、`turns_status: "not_available"`。

共同入口的可选 `--export-unit chunk|audio|both` 在推理结束后导出独立 JSON；不指定时保留原有 JSONL 行为。生产 streaming worker 结果也可以使用以下独立命令导出，无需加载模型或重新推理：

```bash
python src/export_annotations.py --input <FA结果目录或predictions.jsonl> --output <导出目录> --unit chunk
python src/export_annotations.py --input <FA结果目录或predictions.jsonl> --output <导出目录> --unit audio
python src/export_annotations.py --input <FA结果目录或predictions.jsonl> --output <导出目录> --unit both
```

`audio/both` 导出需要能读取实际音频的 Python 环境（含 soundfile）；`export_index.json` 保存文件名、chunk ID、音频路径及文件数量。相同目录已存在对应导出子目录时拒绝覆盖，避免混入上次结果。音频按实际文件路径分组，不会把来自同一原始 MP3 的 3 个 WAV 错合并为一个文件。

本次保存的续跑脚本为 `src/finish_candor_v5.py`：核验 v5 全部阶段成功、持久化实际增强音频及新 ASR、依次运行两种 FA，并对同一份预测导出 `both`。`--finalize-only` 只重新核验已有输出。完整记录见 [smoke_test_annotation_format.md](smoke_test_annotation_format.md)。

目前的 MFA 配置为 Montreal Forced Aligner **3.4.3**（`models/mfa/bin/python`）配合英文声学模型 `models/mfa-english/english_us_arpa.zip` 和词典 `models/mfa-english/dictionary/english_us_arpa.dict`。`mfa` 是程序环境，`mfa-english` 是英文资源目录，两者不是两套对齐后端；双 FA 与三 FA 方案使用同一配置。streaming worker 同样支持 `--export-unit chunk|audio|both`。

## 其他类型的fa问题

目前qwen-fa+mfa+nyra架构已经算解决了0时长词问题，
然后汇总这三个fa limitation里提到的问题：
1. Nyra fa只支持英语，所以中文语音数据目前只采用qwen-fa+mfa，需要屏蔽代码里的nyra fa，可能需要再找一个更强力的中文fa来进一步消除0时长中文
2. nyra在一些复杂场景，如语音不连贯，有笑声，紧邻backchannel和强烈interruption下，以及噪声的标注偏移量远低于mfa和qwen-fa，所以怎么在老路由基础上做此类行为的判别，从而进行路由。具体类型查看https://arxiv.org/pdf/2610.22583 nyra论文的第四页table3，4，5和对应的论述。

### 2026-11-09：补充看法与建议

我的判断是：下一步应同时补齐语言适配和边界质量评估，再扩展路由。现有三级回退已经显著减少零时长，但还不能说完全解决；[四方案实验](candor_four_modes.md)中零时长词从 1201 个降至 47 个，仍有残留。更关键的是，词的时长为正、顺序合法，只能证明结构可用，无法证明起止位置准确。后续需要把“结构修复”和“边界优化”分别统计。

#### 1. 论文支持哪些判断

核对 [Nyra 论文第 4 页 Table 3–5 及第 4、5 节](https://arxiv.org/pdf/2610.22583#page=4)后，可以确认以下证据边界：

- Table 3：Nyra 在会话和不流畅语音上更好，但在干净朗读上，MFA US ARPA 的 MAE 更低。因此没有充分理由把所有句子统一切到 Nyra。
- Table 4：单独统计的是紧邻 filler、laughter、word cutoff 的普通词，不是 backchannel 或多人 interruption。词语截断也不能直接解释为被另一位说话人打断；后两类应作为待验证的路由假设。
- Table 5：验证的是加性噪声，并区分噪声类型和 SNR，不能直接外推到重叠人声。论文最终结果使用 noise-robust 版本，而本项目已有实验使用 standard 权重，需要先确认模型对应关系。
- 论文 MFA 对照使用 beam 400/retry 4000，本项目使用 100/400，且 CLI 版本不同；直接比较之前应增加配置对照。

以下路由与验收设计是针对本项目的建议，尚未实现或验证，不应当作论文结论。

#### 2. 中文先补语言门控和 MFA 资源

“中文采用 Qwen＋MFA”需要加一个前提：MFA 必须加载对应语言的声学模型和词典。当前两个回退入口默认都是 `mfa-english` 下的 English US ARPA 资源；只屏蔽 Nyra，仍不能构成可靠的中文回退链路。当前 `nyra_repair.py` 也没有按语言过滤任务。

建议在任务进入后端前统一做能力检查：英语允许现有 MFA/Nyra；普通话只有配置并验证对应 MFA 资源后才启用 MFA；其他语言、语言未知或中英混说单独记录。不能仅凭句中出现几个拉丁字符就把整句送英文 Nyra，也不能把缺省 English 当作经过确认的语言。屏蔽应发生在依赖检查和模型加载之前，使纯中文任务无需准备 Nyra 权重，并记录 `unsupported_language` 等明确原因。

中文第三个后端可以另做候选调研，但选型目标应是人工边界误差下降、文本覆盖可靠和成本可接受。建议先比较“Qwen＋正确配置的中文 MFA”与候选后端，再决定是否增加一层；保留中文逐字粒度，明确数字、多音字、儿化和中英混说的文本映射，不通过等分词区间生成字级时间戳。没有可用后端时保留原输出并标记待复核。

#### 3. 在零时长回退旁增加复杂场景入口

现有 `prepare_sentence_tasks` 只选择含零时长词的句子；三级模式又只把 MFA 拒绝的句组交给 Nyra。因此，“Qwen 无零时长但边界偏移”及“MFA 结构验收通过但边界仍偏移”都会漏过。建议保留原链路，增加独立的风险触发原因，并对同一句去重：

| 场景 | 可用的判别信号 | 建议处理 |
|---|---|---|
| 零/负时长、越界等结构异常 | 现有逐词检查 | 保留零时长链路；其他硬异常作为显式新增策略单独验证 |
| 填充词、重复、未说完的词 | 保留口语现象的转录、ASR 分歧、局部声学证据 | 英语完整句子可产生 Nyra 候选，即使没有零时长 |
| 笑声等非词汇事件附近 | 带时间区间和置信度的事件检测 | 将与事件相邻的完整句子加入候选，检查事件是否被普通词吸收 |
| 噪声较强 | 现有 DNSMOS 作初筛，结合噪声检测或 SNR 估计 | 经验证后使用匹配的 Nyra 权重；DNSMOS 不直接等价于 SNR |
| backchannel、打断或重叠说话 | 独立声道、说话人活动、重叠区间与上下文 | 先确认目标说话人及转录，再判断能否对齐；不能仅凭短句或“嗯”定性 |
| 可疑边界但无硬错误 | 词区间与语音活动明显冲突、异常长短词、后端边界分歧 | 作为复核或多候选触发信号，不直接认定某个后端正确 |

检测结果允许多标签和未知状态，建议保存 `risk_tags`、事件区间、置信度、检测器版本和 `route_reason`。阈值在开发集上确定；标点、停顿和片段时长只能提供线索，不能生成缺少依据的角色、backchannel 或打断标签。

多人重叠尤其需要单独处理：换一个 FA 不能自动解决“文本属于谁”的问题。有独立音轨时优先按目标音轨对齐；只有混音且目标语音不可辨时，应进入复核或另行评估分离流程。不同说话人的词在时间上重叠可能是真实现象，不能用单个目标词序列的“不重叠”约束消除它。

#### 4. 事件、裁剪和候选验收要一起调整

当前 `nyra_text` 只显式转换独立的 `um/uh`，没有自动检测或插入笑声标签。如果上游转录漏掉笑声、重复或半截词，单纯换后端仍可能让邻词承担这些声学片段。建议保留原 ASR 文本，另外生成可追溯的对齐输入；只有可靠的事件证据才加入后端支持的标签。事件输出单独保存到 `events`，再将普通词映射回原始 `words`，不能将事件当漏词拒绝，也不能为通过覆盖检查而静默删掉普通词。

当前音频裁剪和替换边界依赖 Qwen 时间戳。若 Qwen 已经整体偏移，固定左右 0.35 秒可能没有覆盖真实语音；错误的未修改邻词也可能挡住正确候选。建议在风险句上参考原 ASR/VAD 区间和邻近可靠锚点，必要时扩展为连续完整句组；仍无法确定范围时保留待复核状态，不能用强制裁边来掩盖冲突。

对零时长任务，继续执行整句原子替换和文本、时序硬验收。对原来结构正常的风险句，硬验收只是最低门槛：先以 shadow 模式保存 Nyra 候选，暂不覆盖；待人工评估证明该类场景有收益后，再启用对应规则。后端间分歧能提示不确定性，但多数一致也不等于真值；不建议直接平均多个后端的时间戳。

#### 5. 建议实施顺序与验收指标

1. **先做语言门控和资源核验。** 覆盖英语、中文、混说及未知语言，确认不支持的后端不会被加载，跳过原因可追溯。
2. **建立小规模人工边界评测集。** 按干净语音、不流畅、笑声相邻、噪声、重叠/打断和语言分层，包含没有零时长的样本；同一会话不跨开发集与测试集。先确认转录是否完整，并标记不可辨认或边界有歧义的词。
3. **做固定输入的后端与配置对照。** 比较当前链路、英语风险句直送 Nyra、MFA 宽 beam，以及确认可用的 Nyra 权重变体；单独记录转录/事件标签变化，避免将收益混在一起。
4. **最后验证自动路由。** 分开评估检测器是否找对风险句，以及候选是否改善边界，再决定哪些规则上线；保留现有链路作为基线。

报告至少包含 onset/offset MAE、P95/P99、超过 200 ms 的严重误差比例、未对齐/漏词比例、零时长率、候选接受率和 RTF，并给出各场景样本量。若复现论文指标，还应分别报告其起止误差均值与词尾阈值准确率，避免混用定义。除总体收益外，需要检查正常句被替换后是否退化，以及检测器漏掉了多少严重偏移；没有人工真值的全量结果继续只报告结构异常和风险统计。
