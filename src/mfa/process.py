"""CPU-only MFA baseline: align full input chunks without Qwen timestamps."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/qwen3-fa'))
from core import write_json, write_jsonl
from prediction_format import write_predictions
from process import audit, load_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--mfa-python', type=Path, default=ROOT / 'models/mfa/bin/python')
    p.add_argument('--dictionary', type=Path, default=ROOT / 'models/mfa-english/dictionary/english_us_arpa.dict')
    p.add_argument('--acoustic-model', type=Path, default=ROOT / 'models/mfa-english/english_us_arpa.zip')
    p.add_argument('--mfa-jobs', type=int, default=8)
    p.add_argument('--timeout', type=int, default=7200)
    args = p.parse_args()
    if args.mfa_jobs < 1:
        p.error('--mfa-jobs must be positive')
    for path in (args.mfa_python, args.dictionary, args.acoustic_model):
        if not path.exists():
            p.error('Missing dependency: ' + str(path))
    started = time.perf_counter()
    rows = load_rows(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    corpus = args.output / 'corpus'
    corpus.mkdir()
    tasks = []
    # Reuse the exact physical chunks; no model timestamps define these bounds.
    for index, row in enumerate(rows):
        name = f'c{index:06d}'
        (corpus / (name + '.wav')).symlink_to(Path(row['audio']).resolve())
        (corpus / (name + '.lab')).write_text(row['text'] + '\n')
        tasks.append(dict(task_id=name, id=row['id'], audio=row['audio'], text=row['text']))
    write_jsonl(args.output / 'tasks.jsonl', tasks)
    env = dict(os.environ)
    prefix = args.mfa_python.resolve().parent.parent
    env.update(PATH=str(prefix / 'bin') + os.pathsep + env.get('PATH', ''),
               LD_LIBRARY_PATH=str(prefix / 'lib') + os.pathsep + env.get('LD_LIBRARY_PATH', ''),
               MFA_ROOT_DIR=str((args.output / 'cache').resolve()), CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [str(args.mfa_python), '-c',
               'from montreal_forced_aligner.command_line.mfa import mfa_cli; mfa_cli()',
               'align', str(corpus.resolve()), str(args.dictionary.resolve()),
               str(args.acoustic_model.resolve()), str((args.output / 'aligned').resolve()),
               '--output_format', 'json', '--num_jobs', str(args.mfa_jobs),
               '--temporary_directory', str((args.output / 'work').resolve()),
               '--single_speaker', '--use_threading', '--no_use_postgres',
               '--beam', '100', '--retry_beam', '400', '--clean', '--overwrite']
    write_json(args.output / 'metadata.json', dict(method='mfa', device='cpu',
        input=str(args.input.resolve()), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        command=command, jobs=args.mfa_jobs, cuda_visible_devices='',
        acoustic_model_sha256=hashlib.sha256(args.acoustic_model.read_bytes()).hexdigest(),
        dictionary_sha256=hashlib.sha256(args.dictionary.read_bytes()).hexdigest(),
        alignment_unit='full input chunk', tokenization='native MFA words; silence excluded; OOV retained',
        timestamp_unit='seconds', audit_timestamp_origin='source recording'))
    print(f'MFA CPU: aligning {len(rows)} full chunks with {args.mfa_jobs} jobs', flush=True)
    command_started = time.perf_counter()
    with (args.output / 'mfa.log').open('w') as log:
        try:
            status = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout, check=False).returncode
        except subprocess.TimeoutExpired:
            status = 'timeout'
    command_seconds = time.perf_counter() - command_started
    predictions, failures = [], []
    for row, task in zip(rows, tasks):
        value = dict(row, words=[])
        if not row['text'].strip():
            value['alignment_note'] = 'Empty input transcript; no words to align'
            predictions.append(value)
            continue
        try:
            matches = list((args.output / 'aligned').rglob(task['task_id'] + '.json'))
            if len(matches) != 1:
                raise ValueError(f'No unique MFA output (exit={status})')
            data = json.loads(matches[0].read_text())
            tiers = [v for k, v in data['tiers'].items() if k.casefold().endswith('words')]
            if len(tiers) != 1:
                raise ValueError('No unique MFA word tier')
            for a, b, label in tiers[0]['entries']:
                if not label.strip() or label in {'<eps>', 'sil', 'sp', '<sil>'}:
                    continue
                if not all(math.isfinite(float(t)) for t in (a, b)):
                    raise ValueError('Nonfinite MFA timestamp')
                value['words'].append(dict(text=label, start=float(a) + row['chunk_start'],
                                           end=float(b) + row['chunk_start'], source='mfa'))
            if row['text'].strip() and not value['words']:
                raise ValueError('MFA produced no words for nonempty transcript')
        except Exception as exc:
            value['error'] = str(exc)
            failures.append(dict(id=row['id'], error=str(exc)))
        predictions.append(value)
    stats = audit(predictions)
    stats['oov_words'] = sum(w['text'] in {'<unk>', 'spn'} for r in predictions for w in r['words'])
    stats.pop('sentences')
    stats.pop('bad_sentences')
    stats.pop('sentence_error_rate')
    for row in predictions:
        # Unknown words need review even when their intervals are well formed.
        row['issues'].extend(dict(index=i, text=w['text'], start=w['start'], end=w['end'], reason='oov')
                             for i, w in enumerate(row['words']) if w['text'] in {'<unk>', 'spn'})
    stats['empty_transcript_chunks'] = sum(not r['text'].strip() for r in predictions)
    stats['needs_review_chunks'] = sum(bool(r.get('error') or r['issues']) for r in predictions)
    write_predictions(args.output, predictions)
    write_jsonl(args.output / 'failures.jsonl', failures)
    write_json(args.output / 'unresolved.json', dict(residual_chunks=[
        dict(id=r['id'], error=r.get('error'), issues=r['issues'],
             oov_words=[w for w in r['words'] if w['text'] in {'<unk>', 'spn'}])
        for r in predictions if r.get('error') or r['issues'] or
        any(w['text'] in {'<unk>', 'spn'} for w in r['words'])]))
    write_json(args.output / 'error_stats.json', stats)
    seconds = time.perf_counter() - started
    duration = sum(r['duration'] for r in rows)
    runtime = dict(method='mfa', device='cpu', samples=len(rows), audio_seconds=duration,
                   total_seconds=seconds, mfa_command_seconds=command_seconds,
                   rtf=seconds / duration, mfa_exit_code=status)
    write_json(args.output / 'runtime.json', runtime)
    print(json.dumps(dict(runtime=runtime, stats=stats)), flush=True)
    if status != 0 or failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
