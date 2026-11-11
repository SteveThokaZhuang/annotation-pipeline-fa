# FA 代码结构与有预算邻词调整实验报告

日期：2026-11-11。仓库：`/mnt/data/L202500721/stevezhuang/fa`。

本文基于当前源代码、已保存的实验 JSON、逐词调整记录和测试日志整理。源代码清单覆盖 `src/` 的 49 个 Python 文件，测试清单覆盖 `tests/` 的 10 个 Python 文件；不展开第三方模型、运行环境、缓存及大规模逐条结果。当前工作区尚未提交，复现实验应以结果目录中的代码快照及 SHA-256 为准。

## 1. 结论与当前状态

目前的主流程是 **Qwen FA → 对零时长词所在的完整 chunk 尝试 MFA → 失败时退到句级 MFA → 仍不合法则保留 Qwen**。接缝策略可选 `clamp`、`midpoint`、`bounded`。当前 CLI 默认仍为 `midpoint`；本次新增的 `bounded` 通过 `--mfa-seam-policy bounded` 显式启用，没有替换上一轮全量交付标注。

本次 `bounded` 的核心约束是：**MFA 时间戳不动；正常 Qwen 词的每个起止边界相对最初 Qwen 最多移动 20 ms，词时长最多缩短 10%**。预算不足就原子回退。20 ms 和 10% 是初始工程参数，不能解释成已证明的人耳不可感知阈值。

在 559 个 chunk 的压力抽样回放中，原始 Qwen 有 708 个零时长词；整 chunk＋中点方案剩 2 个，新整 chunk＋有预算调整方案剩 34 个。新方案的 Qwen 端点最大改动为 20 ms、最大时长压缩为 6.25%；对应中点方案分别约为 117.5 ms 和 28.125%。这说明约束有效，但代价是拒绝更多修复，尚不能据此认定边界准确率提高。

另对 8 个真实 chunk 实际重跑 CPU MFA：原有 16 个零时长词，整 chunk 优先入口全部修好；句级入口留下 8 个。两入口 MFA 退出码均为 0，审计重放通过，当前 89 项回归测试通过。本次回放和实际 MFA 重跑均未使用 GPU，Qwen 只复用已有结果。

## 2. 目录树及各文件用途

### 2.1 顶层目录

```text
fa/
├── src/           # 49 个 Python 源文件，详细清单见下
├── tests/         # 10 个测试文件；当前共 89 项测试
├── docs/          # 设计说明、历史实验和本报告
├── data/          # manifest、源文件索引及小规模音频切片
├── models/        # 模型权重、词典和独立运行环境
├── results/       # 预测、审计、汇总；部分目录为指向本机 /tmp 的链接
├── .gitignore     # 忽略 results、models、tests、data
└── .git/          # 版本库
```

`tests/` 当前被 `.gitignore` 忽略，不能仅凭 Git 提交假定测试已包含在内；本次相关测试另存于实验代码快照。大文件结果部分保存在本机 `/tmp`；共享摘要与本机明细的存储位置见第 8 节。

### 2.2 完整源代码树：每个文件的职责

```text
src/
├── qwen3-fa/                         # 共享 FA 主干与边界处理
│   ├── core.py                      # 输入展开、JSONL 读写、Qwen 模型封装、词时间戳校验
│   ├── process.py                   # 公共 CLI/运行流程、输入加载、结构错误统计、可选修复钩子
│   ├── sentences.py                 # 按标点把句子映射到 Qwen 原始词单元，校验拆句不改变分词
│   ├── prediction_format.py         # 内部逐词记录与 HLT-FD 数组格式转换，按 chunk/音频导出
│   ├── nyra_repair.py               # Nyra 整句修复；支持直接 Qwen→Nyra 与 MFA→Nyra 级联
│   ├── seams.py                     # 旧中点策略：同录音/声道的跨 chunk Qwen/MFA 接缝统一
│   └── bounded_seams.py             # 新策略：20 ms/10% 预算、动态邻词传播、跨 chunk 事务回退
│
├── qwen-fa/
│   └── process.py                   # 纯 Qwen 的兼容入口，委托共享主干运行
├── qwen3-fa+mfa/
│   ├── process.py                   # Qwen＋MFA CLI，配置修复范围、接缝策略及 CPU MFA 资源
│   └── repair.py                    # MFA 词映射、任务裁剪、CPU 调用、chunk→句级回退、审计
├── qwen3-fa+nyra/
│   └── process.py                   # Qwen＋Nyra 入口
├── qwen3-fa+mfa+nyra/
│   └── process.py                   # Qwen→MFA→Nyra 入口
├── mfa/
│   └── process.py                   # 纯 CPU MFA 基线，对完整输入 chunk 对齐，不依赖 Qwen 时间
│
├── complex_handling/                # 可选复杂场景检测/路由，当前为 shadow 候选模块
│   ├── __init__.py                  # 对外导出 ComplexHandler 等公共接口
│   ├── config.py                    # 版本化配置、资源路径和语言能力声明校验
│   ├── audio.py                     # 精确读取选定声道/采样区间，管理原音频与实际 FA 音频坐标
│   ├── evidence.py                  # 读取外部风险证据，统一到原录音坐标，校验区间和覆盖
│   ├── detectors.py                 # 文本、质量、重叠、对齐风险检测；可选 PANNs/pyannote 适配
│   ├── router.py                    # 按确定性规则选择候选、跳过或人工复核，不负责模型加载
│   ├── backends.py                  # Nyra 候选后端适配，尽量复用主流程已加载的模型
│   ├── handler.py                   # 统一入口：检测、门控、路由、shadow 候选生成和审计
│   ├── example.json                 # 配置示例，不是模型权重
│   └── README.md                    # 模块接口、证据格式、语言门控与使用限制
│
├── prepare_candor.py                # 复用已有 ASR 文本，准备 Candor 音频切片和 manifest
├── prepare_rover.py                 # 清点 ROVER 结果，去重并生成按原始采样区间划分的 manifest
├── finish_candor_v5.py              # v5 Candor 输入准备、采样率坐标转换、两条 FA 路线与格式导出
│
├── rover_worker.py                 # 常驻模型流式 worker，精确取音频、分 block 处理、归档与统计
├── run_rover_experiments.py         # 历史 GPU 实验套件调度：级联及双后端消融
├── rover_status.py                 # 查看已完成进度，区分处理中和已完成数据
├── watch_rover_report.py            # 监听实验完成状态并更新报告
├── report_rover_experiments.py      # 从已完成的 ROVER 实验结果生成可审计报告
├── finalize_rover_experiments.py    # 历史套件完成后串联归档、校验和报告生成
├── archive_rover_predictions.py     # 合并压缩预测/未解决项，校验完整性并尝试共享存储归档
├── verify_rover_outputs.py          # 校验历史导出、覆盖回执、预测哈希及代码快照
│
├── compare_results.py              # 两方案比较；按审计重放候选和邻词修改，检查词序及未解决项
├── compare_four_modes.py           # Candor 四种 FA 路线的零时长、结构问题和 RTF 比较
├── compare_mfa_mae.py               # 词文本匹配，计算相对纯 MFA 的起点/终点/平均边界 MAE
├── run_rover_mfa_mae.py             # 原始 Qwen 无零时长 chunk 的 CPU MFA 基线、分批与 MAE 汇总
├── verify_rover_mfa_mae.py          # 从逐词结果独立重算上述 MAE、覆盖率和误差分位数
├── run_rover_zero_chunk_mfa.py      # 对原始 Qwen 含零时长的 chunk 批量运行完整 CPU MFA
├── replay_mfa_midpoint.py           # 复用已保存的句级 MFA 原始结果，重放中点方案，不重新推理
├── finalize_rover_replacements.py   # 比较旧句级/中点/整 chunk 策略，合并全量中点结果和抽检包
├── verify_rover_replacements.py     # 全量替换结果校验：文本、词数、时间数组、MAE、接缝审计
│
├── smoke_bounded_replacement.py     # 本次：复用真实候选，比较四种修复策略并保存精确输入/修改
├── smoke_bounded_cpu_mfa.py         # 本次：选 8 个真实 chunk，实际重跑 CPU MFA 的两个入口
├── smoke_test_annotations.py        # 历史 Candor 小样本 MFA/级联及标注格式 smoke 测试
├── smoke_test_rover_audio.py        # 历史单条源 WAV 的 Qwen→MFA→Nyra 全流程 smoke 测试
├── export_annotations.py            # 从已有预测导出单 chunk 或整音频 JSON，无需重新推理
│
├── setup_mfa_models.py              # 获取固定 revision 的官方 MFA 英文声学模型和匹配词典
├── setup_nyra_models.py             # 获取固定 revision 的 Nyra/WavLM 权重
└── setup_nyra_espeak.py             # 在 models 内安装 Nyra 所需 espeak 运行库
```

树中 `example.json` 和 `README.md` 为辅助文件，不计入 49 个 Python 源文件。

需要特别区分：`run_rover_experiments.py`、`rover_worker.py` 及部分历史 smoke 脚本会加载 Qwen/Nyra，不能当作 CPU-only 复现入口。其中当前 `rover_worker.py` 将 Qwen 设备设为 `cuda:0`。本次运行的是两个 `smoke_bounded_*` 脚本，前者只回放结果，后者只新跑 CPU MFA。

`complex_handling` 是可选扩展；当前 shadow 候选保存 `applied: false`，不自动替换主输出。本次有预算接缝实验未启用其额外模型，不把复杂场景模块的能力当成本次实验验证结论。

### 2.3 测试树

```text
tests/
├── test_bounded_seams.py            # 新预算策略、多词传播、短词保护、累计预算、审计和跨 chunk 回退
├── test_sentence_repair.py          # 整句替换、MFA 映射、裁剪、clamp/中点边界、结构问题统计
├── test_cross_chunk_seams.py        # 中点跨 chunk 接缝、声道隔离、空行、拒绝与审计重放
├── test_rover_replacement_policy.py # 整 chunk 接受/拒绝、句级回退、生产封装与归档
├── test_rover_streaming.py          # 精确采样读取、分批统计、流式输入和模型复用
├── test_prediction_format.py        # 时间原点、词单元、chunk/整音频导出及审计兼容
├── test_nyra_repair.py              # Nyra 词覆盖、时间戳和级联路由约束
├── test_complex_handling.py         # 复杂场景规则、证据、语言门控和候选审计
├── test_complex_integration.py      # 复杂场景模块与公共入口/worker 归档接入
└── test_candor_v5.py                # v5 增强音频的真实采样率与区间坐标转换
```

89 项测试中，对模型依赖较重的路由和集成检查使用模拟后端；它们验证程序契约，不代表运行了 89 次真实神经模型推理。实际 CPU MFA 的证据单独保存在 live smoke 中。

### 2.4 模型、数据与已有文档

| 目录/文档 | 用途 |
|---|---|
| `models/qwen3-forced-aligner/` | Qwen 强制对齐权重。 |
| `models/mfa/` | MFA 独立 Python/工具运行环境。 |
| `models/mfa-english/` | 英文声学模型 `english_us_arpa.zip` 和匹配词典。 |
| `models/nyra/` | Nyra 运行环境。 |
| `models/nyra-forced-aligner/` | Nyra 模型及其依赖权重。 |
| `models/nyra-espeak/` | Nyra 的本地 espeak 依赖。 |
| `data/candor_full/` | 早期 Candor 全量切片输入。 |
| `data/candor_v5_smoke/` | v5 Candor smoke 输入。 |
| `data/open_yap_1k_rover/` | Open YAP manifest、来源索引；当前为本机目录链接。 |
| `docs/design.md` | 主设计、命令、策略演化和本次有预算方案说明。 |
| `docs/candor_comparison.md` | 早期 Candor Qwen/MFA 双方案实验。 |
| `docs/candor_four_modes.md` | Candor 四路线历史比较。 |
| `docs/open_yap_streaming_design.md` | Open YAP 流式处理设计。 |
| `docs/open_yap_1k_experiments.md` | Open YAP 历史大规模实验。 |
| `docs/open_yap_single_audio_smoke.md` | 单条 ROVER 音频历史 smoke。 |
| `docs/smoke_test_annotation_format.md` | 标注格式 smoke 记录。 |

历史报告描述当时的参数和默认行为；当前代码和本报告明确区分新旧策略，不能用历史命令的缺省参数自动复现当时结果。

## 3. 数据、时间坐标与主流程

### 3.1 chunk 是什么

Open YAP 的 chunk 是上游处理已经划定的单条音轨内片段，不是本次为了 MFA 临时重新切出来的语义句子。输入通常引用整段说话人 WAV，并通过 `channel`、`start_sample`、`end_sample`、`source_sample_rate` 指定真实区间。

例如实际输入的一条片段为原音轨的 **0.50–3.93 秒**，时长 **3.43 秒**，48 kHz 对应采样范围 **24000–188640**，文本为：

> it's kind of creepy it as your friend is in the room

MFA 的完整 chunk 实验读取这段原始采样区间及声道，重采样为 16 kHz。句级实验才会在 chunk 内，根据 Qwen 句子范围及 padding 生成更小裁剪。上游 chunk、文本句子、模型词单元是三个不同层级。

内部 `words[].start/end` 使用**原录音绝对秒数**。标准导出的 `force_align.timestamps` 使用**选定 chunk 的相对秒数**，转换时扣除一次 `chunk_start`。ROVER 的 `audio_path` 常是完整录音，因此还需配合 `audio_start_sec/audio_end_sec` 读取相应音频。

### 3.2 目前的 Qwen＋MFA 流程

```text
原音频区间＋既有转写
        ↓
Qwen FA（本次复用已有结果）
        ↓
检查 abs(end-start) <= 1e-7 的零时长词
        ├── 没有：保持原词，可能参与相邻接缝的受审计调整
        └── 存在：整 chunk CPU MFA
                    ├── 词覆盖、词单元和区间均合法 → 接受
                    └── 不合法 → 含坏词的完整句子/相邻句组 CPU MFA
                                  ├── 映射与接缝验收通过 → 接受
                                  └── 不通过 → 保留原 Qwen
        ↓
按同一录音、同一声道处理相邻 chunk 接缝
        ↓
输出预测、逐词来源、调整审计、拒绝原因、未解决项
```

`map_mfa_words` 允许 MFA 将一个 Qwen 词拆成多个连续子词后合并首尾时间，例如缩写；禁止把一个 MFA 区间均分为多个 Qwen 词。如果两模型文本覆盖相同但词单元划分不一致，仍可能拒绝。这与时间接缝问题应分开统计。

纯 MFA 基线没有“保留原 Qwen”的含义；混合方案里的“MFA 回退”是指 Qwen 出问题后尝试 MFA，若 MFA 候选也验收失败，再保留原来的 Qwen。

### 3.3 三种接缝策略

| 策略 | 对 MFA 的处理 | 对正常 Qwen 的处理 | 主要风险/限制 |
|---|---|---|---|
| `clamp` | 把候选限制在原邻词之间的空间内。 | 邻词不变。 | MFA 边缘词可能被裁成零时长，整次替换被拒绝。 |
| `midpoint` | 和接触 Qwen 边界一起移动到重叠中点；候选也会检查/限制 chunk 边界。 | 修改紧邻接触端点，没有 20 ms 或 10% 上限。 | 结构合法不代表边界改动小；也可能使短词塌缩。 |
| `bounded` | 原始 MFA 候选时间保持不动；自身越界、重叠或零时长则拒绝。 | 在相对原始 Qwen 的预算内缩短/平移，必要时逐词扩展。 | 预算不足保留更多零时长词；不能用多词压缩绕过接触端点的预算。 |

三者的差异不只体现在中点公式：`bounded` 还明确禁止通过裁剪 MFA 时间凑合法区间。因此不能把所有接受率差异都归因于“动态选了几个邻词”。

## 4. 新方案的算法和实现细节

### 4.1 约束定义

对原始 Qwen 正常词 \(i\)，记起止为 \(s_i,e_i\)，调整后为 \(s'_i,e'_i\)，原时长 \(d_i=e_i-s_i\)。本次约束为：

\[
|s'_i-s_i|\leq0.020,\qquad |e'_i-e_i|\leq0.020
\]

\[
e'_i-s'_i\geq0.9d_i
\]

时间单位为秒。每次修复使用同一份最初 Qwen 作为参照，已经移动过的词不会重新获得一份 20 ms 预算。另要求改后区间有限、正时长、顺序合法且在对应 chunk 内，词文本和词数保持不变。

这里的预算是**对已有标注的改动上限**，不是对人工真值的绝对误差上限。若原始 Qwen 已经偏了 80 ms，限制新增改动 20 ms 并不能证明改后误差小于 20 ms。

### 4.2 动态传播如何进行

左侧发生重叠时，先将紧邻 Qwen 词的结束向左移动到容纳 MFA 的位置；如果这样会超过原时长 10% 的压缩上限，再将它的开始向左平移，保留至少 90% 原时长。若这又碰到前一个词，继续向前传播。右侧对称处理。已有词间空隙优先吸收位移，遇到足够空隙便停止，不预设固定邻词数量。

例：两个原 Qwen 词依次为 `[0.000, 0.100]`、`[0.100, 0.200]`，MFA 修复段从 `0.185` 开始：

```text
第 2 词：结束 0.200 → 0.185，开始 0.100 → 0.095，保留 90 ms
第 1 词：结束 0.100 → 0.095，开始不变，保留 95 ms
MFA：   仍从 0.185 开始，时间戳不改
```

这是测试中验证的多词传播情形。若 MFA 从 `0.150` 开始，紧邻词结束必须移动 50 ms，已经超过其 20 ms 预算；即使再向外选很多词也不能通过。

原 Qwen 接触词若存在轻微 chunk 越界，可以在预算内修正；新端点同时满足 MFA 接触边界和 Qwen 自身 chunk 边界。例如 MFA 开始晚于两 chunk 的分界时，左侧 Qwen 结束应回到自己的 chunk 末端，而不是留在其音频区间之外。

### 4.3 接受、回退和审计

`bounded_seams.py` 的核心函数如下：

| 函数 | 职责 |
|---|---|
| `valid` | 检查正时长、有限值、区间顺序及边界。 |
| `adjust_run` | 沿一侧动态传播，逐词验证相对原始 Qwen 的端点及压缩预算。 |
| `splice_bounded` | 在副本上完成句内候选替换；两侧全部通过才返回结果和审计。 |
| `reconcile_bounded_rows` | 处理批内跨 chunk 接缝；失败恢复对应 MFA chunk 的原 Qwen，再从候选快照重新处理。 |

句内任一侧失败，都不提交此前另一侧的临时修改。跨 chunk 失败时恢复整个相关 MFA chunk，连同该 chunk 内此前的修复一起撤回；这比只保留局部修复更保守。重新处理使用跨 chunk 修改前的候选快照，以免保留失效替换带来的邻词修改。

调整记录包含词索引、原始 Qwen、调整前后词、起止位移、时长压缩和参数。跨 chunk 回退还保存 `rolled_back_attempt` 与拒绝原因。`compare_results.validate_repairs` 根据最初 Qwen 和审计重放，检查最终输出确实由已记录的合法修改产生。

当前跨 chunk 处理只保证**传入批次所包含的相邻数据**。单个 streaming block 的成功验收不等于跨所有 block 的全局验收；全量使用新策略时需要包含完整相邻上下文的合并步骤。历史 `finalize_rover_replacements.py` 仍是中点全量合并脚本，不能不改参数/实现便将其当作 bounded 的全量导出器。

## 5. 实验设置与结果

### 5.1 数据和实验范围

上游 ROVER 数据来源：

```text
/hbt_share02/202610/ob/ob_ppl_env_data/audio_pipeline_optimized/
Pipeline_github_v2/outputs/production/open_yap_1k/rover/rover/batch_results
```

历史全量输入有 919,459 个 chunk、10,301,121 个 Qwen 词单元；其中 303,054 个 chunk 含零时长词。本次没有重新运行全量 Qwen 或全量 bounded，而是进行以下两种 smoke：

1. **保存结果回放**：取最先的 256 个原始零时长 chunk、上一轮全部已知的 5 次句内中点调整和 28 个跨 chunk 接缝案例，再加入相邻上下文。去重后共 559 个 chunk、12,902 个词，其中 323 个 chunk 原始含零时长词。5 次句内调整不是 5 个互不相同的 chunk。
2. **实际 CPU MFA 重跑**：从记录输入中取 5 个普通 chunk 和 3 个已知句内接缝 chunk，共 8 个，分别走整 chunk 优先和句级入口。

样本刻意包含接缝困难案例，并非随机代表性抽样。所有已知旧接缝被纳入，是为了验证约束及拒绝行为，不能直接外推到整个 Open YAP 的拒绝率。

回放使用已保存的真实 MFA 原始候选，没有重新推理；后者实际读取原音频重跑 MFA。MFA 使用相同 English US ARPA 声学模型和匹配词典，`beam=100`、`retry_beam=400`；小样本实际运行 `num_jobs=2`，没有额外扩大 beam 作为失败重试。

### 5.2 回放结果：各方案均包含样本内跨 chunk 接缝处理

| 方案 | 零时长词 | 异常词（去重） | 含结构异常的 chunk | Qwen 端点最大改动 | Qwen 时长最大压缩 |
|---|---:|---:|---:|---:|---:|
| 原始 Qwen | 708 | 746 | 351 | — | — |
| 句级 MFA＋中点 | 2 | 32 | 29 | 175.031 ms | 54.697% |
| 句级 MFA＋bounded | 42 | 61 | 38 | 20.000 ms | 6.250% |
| 整 chunk MFA 优先＋句级中点回退 | 2 | 32 | 29 | 117.500 ms | 28.125% |
| 整 chunk MFA 优先＋句级 bounded 回退 | 34 | 53 | 35 | 20.000 ms | 6.250% |

“Qwen 端点最大改动”和“时长最大压缩”只针对最终仍保留 Qwen 来源的词，参照原始 Qwen，不统计被整词替换为 MFA 的词。它们不是人工边界误差。异常词按索引去重，本样本主要为零时长和越界。

新整 chunk 方案从 708 个零时长词修复至 34 个，修复 674 个，减少约 **95.20%**。中点方案能更激进地接受替换，但可能明显改动正常 Qwen 邻词。

### 5.3 新整 chunk 方案的接受与拒绝

323 个触发 chunk 中，整 chunk 候选在跨 chunk 验收前接受 322 个；另 1 个候选自身区间不合法，句级回退也未修好。跨 chunk 验收又回退 17 个 chunk，因此最终接受 305 个、拒绝 18 个。

| 项目 | 数量 | 解释 |
|---|---:|---|
| 已知跨 chunk 接缝 | 28 | 与中点对照采用同一组案例。 |
| bounded 成功接缝 | 11 | MFA 不动，Qwen 调整在预算内。 |
| bounded 拒绝接缝 | 17 | 最终 17 个全部因为相对原始 Qwen 的端点位移需超过 20 ms。 |
| 正常 Qwen 邻词调整操作 | 11 | 本样本均为跨 chunk 操作。 |
| 因跨 chunk 回退保留的零时长词 | 32 | 来自上述 17 个恢复为原 Qwen 的 chunk。 |
| 候选区间自身不合法保留的零时长词 | 2 | 与接缝预算不足分开统计。 |
| 合计残留零时长词 | 34 | 32＋2。 |

一次真实接受案例：Qwen 的 `original` 词为 `[1090.780, 1090.940]`，结束改到 `1090.935`，开始不动；位移 5 ms，时长从 160 ms 缩到 155 ms，压缩约 3.125%，MFA 不改。

一次真实拒绝案例：Qwen 接触词 `her` 结束于约 `5606.555`，MFA 下一个词从 `5606.510` 开始。仅容纳 MFA 就需要前移约 45 ms，超过 20 ms，因此回退。另有 `explained` 结束约 `1932.440`、MFA `explain` 开始 `1932.205` 的案例，所需移动约 235 ms。这类差异无法靠多选几个词、同时保持接触端点最多移动 20 ms 来解决。

本次真实样本中，**没有观察到成功的句内多词传播**：原有句内重叠候选因预算不足被拒绝；成功案例是跨 chunk 接触词调整。多词传播的功能目前由定向测试验证，尚缺真实数据上的正例证据。

### 5.4 实际 CPU MFA 重跑

| 入口 | 输入 chunk | 原零时长词 | MFA 任务组 | 接受/拒绝组 | 最终零时长词 | MFA 子进程墙钟 |
|---|---:|---:|---:|---:|---:|---:|
| 整 chunk 优先＋bounded | 8 | 16 | 8 | 8 / 0 | 0 | 67.720 s |
| 单独句级＋bounded | 8 | 16 | 9 | 5 / 4 | 8 | 66.082 s |

句级的 4 个拒绝组均因超过 20 ms 预算，未提交部分修改。这里的时间包含小批次 MFA 的启动及处理开销，不作为稳定吞吐量或 GPU/CPU 性能比较。整 chunk 入口在此 8 个案例全部接受，**没有实际触发其句级 fallback**；fallback 链路另有回归测试和保存候选的回放验证。

该结果支持“完整 chunk 有时能减少句级裁剪所产生的接缝问题”，但仅 8 个样本，不能推断所有音频都应采用整 chunk MFA。

### 5.5 测试与验收覆盖

89 项回归测试通过，关键检查包括：

- 左右对称传播，已有空隙吸收位移，短词保留至少 90% 时长。
- 多次修复仍参照最初 Qwen，不能累加获得新预算。
- 一侧已临时修改、另一侧失败时，整次修改撤回。
- MFA 时间不被取中点或裁边；非法 MFA、词覆盖及分词不一致被拒绝。
- 跨 chunk 超预算时恢复原 Qwen，不遗留失效候选造成的邻词调整。
- 词序和词数不变、没有新增零时长词、所有修改可审计重放。

程序验收通过只证明约束和记录一致。最终输出仍可能保留原始 Qwen 的结构异常；例如新整 chunk smoke 最终仍有 35 个异常 chunk。不能将“已接受区间合法”误读为“全部输出已无异常”。

## 6. 与此前全量结果及 MAE 的关系

### 6.1 全量结构结果是历史背景，不是本次 bounded 全量测试

| 历史策略 | 全量零时长词 |
|---|---:|
| 原始 Qwen | 535,259 |
| 旧句级 MFA＋clamp | 747 |
| 句级 MFA＋中点 | 742 |
| 仅整 chunk MFA，失败直接保留 Qwen | 16,468 |
| 整 chunk 优先＋句级中点回退 | 696 |

完整 chunk MFA 在 303,054 个零时长 chunk 中接受 292,414 个，约 96.49%；拒绝 10,640 个，其中 10,508 个是 Qwen/MFA 词单元不一致。这个现象解释了为什么“仅整 chunk MFA”不如“整 chunk 优先＋句级回退”。时间接缝策略解决不了分词映射失败。

历史最终 696 个残留零时长词中，653 个对应缺少 MFA 输出，22 个为转写覆盖不一致，19 个为区间不合法，2 个为显式未知标签。它们不是全部由中点接缝造成；在这批缺失候选的案例上，仅改变接缝规则没有可供放入的 MFA 结果。

上述历史全量结果没有被本次 smoke 覆盖。不能把本次 559 个 chunk 的 34 个残留直接与全量 696 个作优劣比较。

### 6.2 MAE 的定义和已有结果

对文本匹配成功的第 \(i\) 个词：

\[
\delta_i=\frac{|s_i-\hat{s}_i|+|e_i-\hat{e}_i|}{2},\qquad
\operatorname{MAE}=\frac{1000}{N}\sum_{i=1}^{N}\delta_i
\]

时间戳为秒，MAE 输出为毫秒。匹配先做 NFKC、大小写及字母数字归一化，再用词序列 Levenshtein 对齐，只评分完全匹配的词；显式 `<unk>/spn` 不纳入。起点 MAE 和终点 MAE另行统计。`<unk>/spn` 计数不等于所有词典 OOV 数量。

| 已有评估范围 | 方案 | 相对纯 MFA 的平均词边界 MAE |
|---|---|---:|
| 排除原始零时长 chunk 的 616,394 个非空 chunk；匹配 3,829,800 词 | Qwen / 历史 Qwen＋MFA | 两者均为 54.059 ms |
| 同上 | MFA 自身 | 0，属于自比较 |
| 原始含零时长的 303,054 个 chunk；公共匹配 6,433,946 词 | Qwen | 64.136 ms |
| 同上 | 旧句级 MFA | 10.136 ms |
| 同上 | 句级中点 MFA | 10.136 ms |
| 同上 | 整 chunk 优先＋句级中点 | 0.083 ms |

第二组 MAE 是**逐 chunk 替换后、最终跨 chunk 中点统一前**的历史指标；本次 smoke 表格则已经处理样本内跨 chunk 接缝，计算阶段不同。两组评估范围也不同，不能把 54.059 和 64.136 视为同一数据上两种策略的比较。

无零时长子集里历史混合输出逐词等于 Qwen，因此二者 MAE 相同，是触发规则的预期行为。整 chunk 替换后的 MAE 接近 0，则很大程度来自直接复制作为参考的 MFA，不能据此证明比 Qwen 更接近真实语音边界。

**本次 bounded smoke 没有新增人工边界真值或独立边界 MAE。** 目前能回答“改了多少原 Qwen、修复多少零时长、为何拒绝”，尚不能回答“人听起来是否更准”。MFA 在这里应称为伪参考，而非经过人工确认的金标准。

## 7. 分析与后续建议

### 7.1 已得到支持的判断

1. **20 ms / 10% 的约束确实生效。** 逐词记录与最终输出检查均通过，实际最大压缩 6.25%，没有靠多次修改重置预算。
2. **完整 chunk 优先在本样本更有利。** bounded 下整 chunk 优先比句级少 8 个残留零时长词；实际 CPU 小样本也有相同方向。但样本不足以做普遍结论。
3. **中点能减少结构性失败，但可能明显改动邻词。** 在同一 smoke 样本里，中点对照的端点位移和压缩最大值远高于新预算。是否改善声学准确率，需要人工参考。
4. **新方案是约束更强的替换，不是消除所有接缝问题的办法。** 17 个跨 chunk 拒绝均超过端点预算，不能把空间无限分摊给更远的词。

### 7.2 尚未验证的部分

未进行人工听审；多词传播缺少真实成功案例；未完成全量 bounded 运行；20 ms 和 10% 尚未通过数据集校准；缺少 MFA 候选、覆盖失败及分词失败仍需独立解决。上游相邻 chunk 的文本重复或切分错误也不能仅靠时间戳挤压修正。

### 7.3 建议的下一阶段

先保留 bounded 为可选策略，不立即切换全量默认。优先人工检查同一组样本里的三类案例：11 个预算内成功接缝、17 个预算不足回退接缝，以及句内多词传播候选。结合波形/频谱及音频确认是 Qwen 边界偏差、MFA 偏差还是上游切分问题，再决定应放宽哪个约束。

若继续做参数实验，可固定同一候选、同一词单元和同一抽样集，比较 10/20/30 ms 端点上限；先固定 10% 压缩上限，避免同时改多个参数难以归因。报告应同时给出修复率、正常词改动分布、短词压缩、拒绝原因以及独立人工参考的边界 MAE。然后补充真实多词传播正例，再做全量 bounded 的全局跨 block 合并校验。这些是建议，本文生成过程未启动额外实验。

## 8. 记录位置、复现与存储

### 8.1 主要记录

| 路径 | 内容 |
|---|---|
| [共享实验摘要](../results/open_yap_1k_summary/bounded_smoke_20261111.json) | 本次参数、四方案统计、CPU 证明、实际重跑、89 项测试、版本迭代与局限。 |
| [完整记录](../results/open_yap_bounded_smoke/experiment_record.json) | 与共享摘要对应的完整实验说明及本机明细索引。 |
| `results/open_yap_bounded_smoke/inputs/` | 精确 Qwen 输入、句级任务、原始 MFA 候选和完整 chunk MFA 参考。 |
| `results/open_yap_bounded_smoke/selection.json` | 每条样本的来源 block 和选中原因。 |
| `results/open_yap_bounded_smoke/{sentence_midpoint,sentence_bounded,chunk_then_midpoint,chunk_then_bounded}/` | 各方案预测、邻词调整、整 chunk 尝试、接缝及拒绝审计。 |
| `results/open_yap_bounded_smoke/live_cpu_{chunk,sentence}/` | 两次真实 CPU MFA 的音频裁剪、命令、原始对齐、预测和校验。 |
| `results/open_yap_bounded_smoke/live_final_source_revalidation.json` | 使用最终源代码再次重放验收真实 CPU MFA 输出。 |
| `results/open_yap_bounded_smoke/{code_snapshot,provenance.json}` | 最终代码和输入文件 SHA-256。 |
| `results/open_yap_bounded_smoke/live_start_code_snapshot/` | 实际 CPU MFA 启动时的代码快照，与后续小修改区分。 |
| `results/open_yap_bounded_smoke/{tests.log,replay.log,live_cpu.log,cpu_validation.json}` | 测试、回放、实际 MFA 日志和进程 CPU 环境记录。 |
| [历史无零时长 MAE 汇总](../results/open_yap_1k_summary/mfa_pseudo_reference_mae.json) | 616,394 个非空无零时长 chunk 的历史 CPU MFA 评估。 |
| [历史全量替换汇总](../results/open_yap_1k_summary/midpoint_chunk_comparison.json) | 全量中点/整 chunk 实验、696 个残留及相关校验。 |

`results/open_yap_bounded_smoke` 实际指向 `/tmp/fa_bounded_smoke_20261111_v4`，其中 live 子目录链接到初次运行目录 `/tmp/fa_bounded_smoke_20261111/`。初次回放、增加同口径对照、修正自身 chunk 边缘约束、最终版本都保留在摘要的 `revisions` 字段中。共享摘要已落到仓库所在共享路径，大型原始候选和日志仍是本机文件，并非已全部复制到共享持久存储。

### 8.2 CPU-only 复现命令

以下命令用于重新生成一份新目录，避免覆盖现有记录；需要本机保存的输入快照仍可访问。

```bash
cd /mnt/data/L202500721/stevezhuang/fa

# 只回放已记录输入，不运行 Qwen 或 MFA 模型。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /mnt/data/L202500293/obed/envs/qwenasr/bin/python \
  src/smoke_bounded_replacement.py \
  --recorded-inputs results/open_yap_bounded_smoke \
  --output /tmp/fa_bounded_report_repro

# 读取上述记录中的 8 个真实 chunk，实际重跑 CPU MFA。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /mnt/data/L202500293/obed/envs/qwenasr/bin/python \
  src/smoke_bounded_cpu_mfa.py --replay /tmp/fa_bounded_report_repro

# 契约和回归测试，不加载真实 Qwen/Nyra 模型权重。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /mnt/data/L202500293/obed/envs/qwenasr/bin/python \
  -m unittest discover -s tests -v
```

复现输出目录需使用尚未存在的路径。本文仅整理已有证据、生成报告，没有重跑上述实验，也没有启动 GPU 任务。

## 9. 2026-11-11 OTO141 四卡实跑：Qwen3-FA＋MFA bounded

### 9.1 输入与格式 adapter

本次实际输入为：

```text
/hbt_share02/202610/ob/ob_ppl_env_data/audio_pipeline_optimized/Pipeline_github_v3_Merge/outputs/oto104_magiczh_full/production/oto_141/rover
```

该目录是 `recordings/*.json` 格式：每个 recording 文件顶层包含 `recording_channel`、整段 `text`、`chunks` 和 `words`；本批 `words` 为空，逐 chunk 文本和音频区间位于 `chunks[].text`、`chunks[].chunk`，候选 ASR 位于 `chunks[].hypotheses`。它不是当前 `rover_worker.py` 直接读取的逐行 manifest 格式，也不是旧版 `batch_results/*/*.json` 的顶层 `results` 格式。

独立转换器位于 [`adapters/oto141_20261111/recordings_to_manifest.py`](../adapters/oto141_20261111/recordings_to_manifest.py)，没有修改 `src/` 下现有代码。转换规则如下：

| 层级 | OTO141 recording JSON | worker manifest | 输出 `prediction_format` |
|---|---|---|---|
| 文件组织 | 一个 recording 一个 JSON，`chunks` 为数组 | 每行一个 chunk，按音频分组分到 `gpu_0..3.jsonl` | 每个 block 一个 `predictions.jsonl`/`alignment_predictions.jsonl` |
| 文本 | `chunks[].text`；ROVER 已融合文本 | `text`，原样保留 | `words[].text` 及导出时间戳 |
| 音频 | `chunk.audio_path` 指向增强 WAV；区间用 `audio_start_sample/audio_end_sample` | `audio`、`source_sample_rate=48000`、`start_sample/end_sample` | 内部词时间使用源录音绝对秒；标准导出按 chunk 相对秒 |
| 标识 | `chunk.id`、`recording_id` | `id`、`recording_id`、`channel` | 保留 `id` 和来源审计字段 |
| 候选系统 | `hypotheses` 数组 | 仅作为 provenance 保存，不参与 Qwen 推理 | Qwen/MFA 来源记录在 `source`/修复审计 |

本批共 1,076 个 recording、122,763 个 chunk、535,284.7338 秒音频；增强 WAV 均存在，转换时没有 invalid row 或 missing audio。生成的临时 manifest 在 `/tmp/fa_oto141_manifest/`，完整推理结果和日志在 `/tmp/fa_oto141_bounded_20261111/`，MFA scratch 在 `/tmp/fa_oto141_bounded_scratch/`。

### 9.2 运行命令与环境

四卡启动器是独立临时脚本 `/tmp/run_oto141_bounded_fixed.py`，每个 worker 使用独立的 `CUDA_VISIBLE_DEVICES=0,1,2,3`，worker 内设备为 `cuda:0`，参数为 `--method qwen+mfa --mfa-seam-policy bounded --mfa-repair-scope chunk --mfa-jobs 32 --batch-size 128 --block-size 32768`。等价的单卡 worker 命令形式为：

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/data/L202500293/obed/envs/qwenasr/bin/python -u src/rover_worker.py \
  --input /tmp/fa_oto141_manifest/gpu_0.jsonl \
  --output /tmp/fa_oto141_bounded_20261111/gpu_0 \
  --scratch /tmp/fa_oto141_bounded_scratch/gpu_0 \
  --method qwen+mfa --mfa-seam-policy bounded \
  --mfa-repair-scope chunk --mfa-jobs 32 \
  --batch-size 128 --block-size 32768
```

可复现实验使用的环境记录在 [`adapters/oto141_20261111/environment.txt`](../adapters/oto141_20261111/environment.txt) 和 [`adapters/oto141_20261111/qwenasr_requirements.txt`](../adapters/oto141_20261111/qwenasr_requirements.txt)：Python 为 `/mnt/data/L202500293/obed/envs/qwenasr/bin/python`，Torch 2.6.0+cu124，CUDA 12.4，四张 NVIDIA H100 80GB。MFA 使用仓库内 `models/mfa` 环境和 `models/mfa-english` 模型/词典；没有在 `models/` 新建环境，因为现有 qwenasr 环境已能正常加载 Torch 和四卡，MFA 环境也已可运行。

### 9.3 结果和 RTF

| 指标 | 结果 |
|---|---:|
| chunk | 122,763 |
| 音频时长 | 535,284.7338 s（148.69 h） |
| 四卡 wall time | 1,320.1717 s |
| 四卡 wall RTF | **0.00246630** |
| 四卡 worker 时间合计 | 5,133.8313 s |
| worker work RTF | **0.00959084** |
| MFA 请求句组 | 41,378 |
| MFA 接受句组 | 41,375 |
| MFA 拒绝句组 | 3，均为 transcript coverage mismatch |
| MFA 子进程退出码 | 四卡均为 0 |
| Qwen 原始零时长词 | 74,643 |
| 最终零时长词 | 4 |
| 零时长词减少 | 99.9946%（修复 74,639 个） |
| 原始异常 chunk | 41,378 个触发修复 |
| 最终异常 chunk | 59 个；其中 3 个保留零时长词、其余为越界项 |

四卡 wall RTF 是从四个 worker 启动到归档汇总完成、除以全部唯一 chunk 音频时长；worker work RTF 是四个 worker 的独立耗时相加后除以同一分母，不能当作实际墙钟吞吐量。每卡日志中的 GPU 在 Qwen 阶段有计算负载，进入 MFA 后 GPU 利用率降为 0% 属正常现象：MFA 使用 CPU，Qwen 权重仍留在显存中。实跑期间主机还存在其他 MFA 作业，因此该 RTF 是本次并发环境下的观测值，不是独占主机上限。

各卡的 wall/工作 RTF 分别为 GPU0 `0.00961660`、GPU1 `0.00961672`、GPU2 `0.00950739`、GPU3 `0.00962265`；四卡负载按音频时长均衡，单卡处理 29,686–31,539 个 chunk。完整机器可读汇总为 `/tmp/fa_oto141_bounded_20261111/summary.json`，逐卡 `error_stats.json`、`validation.json` 和日志均保留在该目录。

完整复现顺序为先生成 manifest，再并行启动四个 worker；adapter 不依赖 `src/` 的导入：

```bash
/mnt/data/L202500293/obed/envs/qwenasr/bin/python \
  adapters/oto141_20261111/recordings_to_manifest.py \
  /hbt_share02/202610/ob/ob_ppl_env_data/audio_pipeline_optimized/Pipeline_github_v3_Merge/outputs/oto104_magiczh_full/production/oto_141/rover \
  /tmp/fa_oto141_manifest_new

mkdir -p /tmp/fa_oto141_bounded_new /tmp/fa_oto141_bounded_scratch_new
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g \
  /mnt/data/L202500293/obed/envs/qwenasr/bin/python -u src/rover_worker.py \
    --input /tmp/fa_oto141_manifest_new/gpu_${g}.jsonl \
    --output /tmp/fa_oto141_bounded_new/gpu_${g} \
    --scratch /tmp/fa_oto141_bounded_scratch_new/gpu_${g} \
    --method qwen+mfa --mfa-seam-policy bounded \
    --mfa-repair-scope chunk --mfa-jobs 32 \
    --batch-size 128 --block-size 32768 > /tmp/fa_oto141_bounded_new/gpu_${g}.log 2>&1 &
done
```

四个 worker 必须等待全部退出码为 0 后再读取各自 `blocks/b*/error_stats.json`、`validation.json` 和 `runtime.json`，并用各卡 `audio_seconds` 加权计算 wall RTF；不能把任一单卡的 `rtf` 直接当成四卡 wall RTF。
