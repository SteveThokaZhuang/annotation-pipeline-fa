# OpenYap 单音频三 FA 格式灰度

输入：`/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing/open_yap_1k/conv_57b6528be18e/speaker_b.wav`。物理 GPU 3，使用连续两次采样计算负载为 0% 且空闲显存足够的卡。

全文件 2383.943000 秒、48000 Hz、1 通道；匹配 440 个原有 ROVER 区间，标注区间总长 1474.530000 秒。沿用融合 ASR 原文，按 Qwen → MFA → Nyra 路由执行一次，没有重跑 ASR。

本次目的是验证标注 pipeline 的输出格式。源文件是 speaker_b 音轨，但没有真实 turn/角色元数据；保留 `turns_status: "not_available"`，不把 VAD 区间写成 user/assistant turn。

完成 440 个区间、3413 个字词；Qwen/MFA/最终零时长数：283/0/0。

Nyra 实际请求 0 个句组；仅在 MFA 未通过时触发。

- `recording_prediction.json`：整条音轨的并列 words/timestamps，时间戳对应完整 WAV；未标注的静音区间不补造词。
- `qwen3-fa+mfa+nyra/predictions.jsonl`：逐区间标准预测，时间戳对应各音频选定区间的起点。
- `alignment_predictions.jsonl`：原录音坐标及完整修复审计；块内保留 MFA/Nyra 接受、拒绝与未解决记录。
- `summary.json`：格式、词元/时间偏移、覆盖及整句回退校验结果，均通过。

归档时共享盘曾返回 ENOSPC，使用完整本机结果恢复，没有重复推理；见 `archive_recovery.json`。Candor 的可重建 MFA cache/work 移至本机并保留链接，主预测及审计文件保存于共享项目目录，见 `results/smoke-test/storage.json`。

结果目录：`/mnt/data/L202500721/stevezhuang/fa/results/smoke-test/open_yap_1k/conv_57b6528be18e/speaker_b`。

第一条实际区间结果：

```json
{
  "conversation_id": "6e1d5d4054c774cde1b1b708",
  "chunk_id": "6e1d5d4054c774cde1b1b708_c0_134400_203040",
  "annotation_unit": "chunk",
  "audio_path": "/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing/open_yap_1k/conv_57b6528be18e/speaker_b.wav",
  "audio_start_sec": 2.8,
  "audio_end_sec": 4.2299999999999995,
  "audio_duration_sec": 1.43,
  "text": "hi",
  "force_align": {
    "timestamps": [
      [
        0.31999999999999984,
        0.8799999999999999
      ]
    ],
    "words": [
      "hi"
    ]
  },
  "turns": [],
  "turns_status": "not_available",
  "source_audio_path": "/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing/open_yap_1k/conv_57b6528be18e/speaker_b.wav",
  "recording_start_sec": 2.8,
  "recording_end_sec": 4.2299999999999995,
  "alignment_status": "aligned",
  "language": "English",
  "channel": 0,
  "source_sample_rate": 48000,
  "start_sample": 134400,
  "end_sample": 203040
}
```
