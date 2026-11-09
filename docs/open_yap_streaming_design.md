# OpenYap ROVER 四卡流式实验设计

用户指定输入为 `/hbt_share02/202610/ob/ob_ppl_env_data/audio_pipeline_optimized/Pipeline_github_v2/outputs/production/open_yap_1k/rover/rover/batch_results`，并明确同意流式分片。实验顺序为 Qwen＋MFA＋Nyra → Qwen＋MFA → Qwen＋Nyra，每组使用全部四张 H100；一组全部完成后再启动下一组，每组独立运行与计时。

## 数据适配与四卡分片

`src/prepare_rover.py` 逐文件处理批 JSON 的 `results`，以 `chunk.id` 去重，保存每个来源文件及分片的 SHA-256。始终使用原 ROVER 融合文本；缺失 ASR hypothesis 允许为 `null`，不会因此丢弃 ROVER 样本。相同重复项不重复计时；文本或采样区间冲突则选择来源文件 mtime 最新项，并记录两个版本。无效输入、重复冲突、缺失音频分别记录。

SQLite 索引在本地 `/tmp` 构建，按每个音频通道文件的标注总时长做最长任务优先的贪心平衡。同一通道固定分到一张卡，分片内按音频路径、起始采样点排序。三组实验复用完全相同的四份 manifest。

原 WAV 是整段录音。`src/rover_worker.py` 复用文件句柄，按 `start_sample:end_sample` 精确 seek/read，选择 `channel`，只将该区间重采样到 16 kHz；不把完整 WAV 误当成已切片音频，也不预先落地几十万个单独 WAV。

每卡使用 8 个 I/O 线程，每个线程有独立的 SoundFile 句柄，避免共享 seek 指针。线程池保持输入结果顺序，逐样本单元检查确认与串行切片一致。串行 I/O 的启动检查只用于定位瓶颈，不计入正式实验结果。

运行环境为共享主机。正式测量期间观察到其他数据转换作业占用 CPU 和存储 I/O；四张 GPU 由本实验使用，但不将这些 RTF 描述为独占主机的性能上限。

## 流式处理与模型常驻

每个 GPU worker 每块最多处理 32,768 个片段，Qwen batch size 为 128。Qwen 每卡每组加载一次；Nyra 首次需要时加载，同一组后续块复用。当前块音频保存在内存中，供 Qwen 和回退裁剪共用；块归档后释放。MFA 每块批量执行，配置每卡 32 CPU jobs、每 job BLAS/OMP 线程为 1。四卡最多使用 128 个 MFA jobs；主机可用 CPU 数为 192。

块内先按源文件顺序读取音频，再按时长排序组成 Qwen batch，减少大量短片段被 padding 到 25 秒造成的浪费；输出恢复至原 manifest 顺序。三组消融采用同样的分桶规则。未分桶的短启动尝试没有完成全量，保留为调试数据，不纳入正式 RTF。

Qwen 使用本地 0.6B、public 后端、BF16、SDPA；Nyra 为固定 revision 英文 standard＋WavLM、FP32；MFA 使用同一 English US ARPA 模型及词典，beam 100/retry 400。权重来源和哈希沿用本地 provenance，不把模型下载计入推理时间。

ROVER 融合文本通常去掉了标点。缺乏句末标点时，以完整 ROVER/VAD 片段作为可用上下文整体回退，不编造自然句边界。保留标点时沿用现有句子拆分及 Qwen 词元覆盖检查。

1. 三级方案：Qwen 逐词检测，零时长所在完整可用句子/片段交 MFA；仅把 MFA 验收未通过的原任务交 Nyra。
2. Qwen＋MFA 消融：同样触发范围，只使用 MFA。
3. Qwen＋Nyra 消融：同样触发范围，直接使用 Nyra。

候选必须词序与覆盖完整、每个字词正时长、不越出片段、与未修改邻词不重叠。裁剪左右各加 0.35 秒余量。失败保留上一阶段原结果，写入独立记录，不使用最小时长或等分方法伪造时间。裁剪相对时间恢复到原录音时只添加一次偏移。

公共实现的扩展位于 `src/qwen3-fa+mfa/repair.py` 的可选 `audio_loader` 和 `src/qwen3-fa/nyra_repair.py` 的实例模型缓存。原独立入口的默认文件读取及单次流程保留。回归检查覆盖准确采样切片、通道选择、全局加权分母、内存音频回退及模型跨块只加载一次。

## 存储布局与空间事件

本轮准备阶段，共享项目盘实际返回 `ENOSPC`；未修改原始数据。改为把大体积 manifest、索引及正式实验输出放在本机磁盘，项目内保留符号链接和小体积文档/代码。代码原文件仍保存在项目 `src/`，不是只存在于临时脚本。

- `data/open_yap_1k_rover` 指向本地完整 manifest/索引目录；其 `preparation.json` 保存实际路径、来源哈希、数量及四卡分配。
- `results/open_yap_1k_rover_4gpu` 指向本地正式结果根目录；本轮实际位置记录在 `storage.json`。
- 本地存储是当前机器的运行结果位置，不将它描述为共享盘的持久归档。恢复共享存储空间后，可整体迁移实际目录并更新链接。

完整索引在空间错误前已完成，恢复时复用该索引并重新计算来源哈希及导出分片；266 条 ASR hypothesis 为 null 的数据适配问题已修正并补回，不静默丢弃。恢复事件见 `results/open_yap_inventory_recovery.json`。`preparation.json` 的恢复导出耗时不冒充初次完整清点时间。

## 归档、验证与进度

`src/run_rover_experiments.py` 每组启动四个独立 GPU 进程。结果按 `模式/gpu_N/blocks/b00000/` 分块保存：原始 Qwen、可选 MFA 中间结果、最终预测、错误统计、回退接受/拒绝、未解决 JSON、阶段时间和全量替换校验。最终 `results.json` 列出所有预测分片，不重复拼接一份巨型文件。

逐任务 MFA 原始输出合并为 `raw_alignments.jsonl.gz`。临时 WAV 和 Kaldi 工作目录在本地 scratch 中，归档后删除；记录保留源音频、采样率、通道和裁剪绝对时间，可以从原始音频重建。分块输出有助于诊断；当前 runner 不把中断实验残留当成完整新一轮。

每卡更新 `progress.json`，协调器更新 `status.json`；每组完成写 `summary.json`，整套实验写 `suite_summary.json`。每卡 Qwen 原始字词时间戳有独立哈希，用于核对消融间 Qwen 输出是否一致。若 worker 异常退出，停止后续消融，先处理失败。

## RTF 和零时长口径

主 RTF = **从启动四个 worker 到全部结束、归档及汇总完成的墙钟秒数 / 所有唯一标注片段秒数之和**。计时包括全部模型加载、原 WAV 读取、重采样、Qwen、MFA/Nyra、失败尝试、校验与归档 I/O。共享的输入清点和导出单独记录，不混入每组 FA 计时。

同时报告 worker 耗时之和 / 音频秒数，称为归一化工作量，不能把它当成四卡吞吐 RTF。Qwen、MFA、Nyra 的分阶段耗时为各 worker 时间之和；各卡阶段会重叠，不能直接累加为真实总墙钟。

音频分母为唯一标注区间长度之和，不含 VAD 未保留的静音，不把双通道音频时长当作会话墙钟时长。按 Qwen、MFA 后、最终阶段分别统计总字词、零时长数/率、含零时长的句子与片段、最长连续零时长词、全部结构异常和推理失败。所有百分比按全量计数重算，不平均各卡或各块的比例。没有人工时间真值，不报告对齐边界准确率。

## 复现入口

```bash
python src/prepare_rover.py --input 原始batch_results目录 --output 本地manifest目录
python src/run_rover_experiments.py \
  --input data/open_yap_1k_rover \
  --output /tmp/fa_open_yap_results \
  --scratch /tmp/fa_open_yap_work \
  --batch-size 128 --block-size 32768 --mfa-jobs 32
```

输出目录必须不存在，重复实验指定新目录。`experiment_design.json` 保存真实命令和参数。最终数值、实际样例及空间限制会另写到 `docs/open_yap_1k_experiments.md`。

`src/watch_rover_report.py` 在每组完整完成后更新记录。`src/finalize_rover_experiments.py` 等待三组全部完成，再调用 `src/archive_rover_predictions.py` 将最终预测压缩成完整 gzip JSONL，并尝试复制到项目 `results/open_yap_1k_summary/predictions/`。若共享复制再次遭遇空间错误，删除本次不完整副本，保留本地完整压缩文件并写明错误。压缩导出在基准后单独计时，最终报告同时列出包含这一步的 RTF。

导出同时逐条对照四卡 manifest，核验 ID、原文、来源路径、采样边界、通道及顺序，独立重算总词数与零时长词数并比对汇总，检查每块路由验证均已通过。归档索引保存这些核验结果与压缩文件 SHA-256。

`python src/verify_rover_outputs.py` 复核三组完整覆盖回执、全部块的路由校验、Qwen 一致性、代码快照和共享归档 SHA-256，并读取压缩未解决记录核对数量；结果保存为 `results/open_yap_1k_summary/final_audit.json`。
