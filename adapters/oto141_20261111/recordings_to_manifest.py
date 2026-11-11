import glob,json,hashlib,sqlite3,collections,sys,time
from pathlib import Path
src=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=False)
files=sorted((src/'recordings').glob('*.json'))
rows=[]; invalid=[]
for p in files:
    x=json.loads(p.read_text())
    for item in x.get('chunks',[]):
        try:
            c=item['chunk']; audio=c.get('audio_path') or c['path']
            sr=48000 if c.get('audio_path') else int(c['sample_rate'])
            start=int(c.get('audio_start_sample',c.get('start_sample'))); end=int(c.get('audio_end_sample',c.get('end_sample')))
            if not Path(audio).is_file() or end<=start: raise ValueError('bad audio/bounds')
            hs=[h for h in item.get('hypotheses',[]) if isinstance(h,dict) and h.get('language')]
            lang=(hs[0].get('language') if hs else 'English') or 'English'
            rows.append(dict(id=c['id'],audio=audio,source_audio=audio,audio_scope='recording',text=item.get('text',''),language=lang,
                recording_id=c['recording_id'],channel=int(c.get('channel',0)),source_sample_rate=sr,start_sample=start,end_sample=end,
                chunk_start=start/sr,chunk_end=end/sr,duration=(end-start)/sr,source_json=str(p),rover_signature=(item.get('_provenance') or {}).get('signature'),
                input_language=lang,hypotheses=[{k:h[k] for k in ('text','language','source_result') if k in h} for h in item.get('hypotheses',[]) if isinstance(h,dict)]))
        except Exception as e: invalid.append({'source':str(p),'error':repr(e)})
if invalid: (out/'invalid_inputs.jsonl').write_text('\n'.join(json.dumps(x) for x in invalid)+'\n')
rows.sort(key=lambda r:(r['audio'],r['start_sample'],r['id']))
# duration balance by audio groups
by=collections.defaultdict(list)
for r in rows: by[r['audio']].append(r)
gpus=['0','1','2','3']; loads={g:0.0 for g in gpus}; groups=[]
for a,rs in sorted(by.items(),key=lambda kv:-sum(x['duration'] for x in kv[1])):
 g=min(gpus,key=loads.get); groups.append((g,a,rs)); loads[g]+=sum(x['duration'] for x in rs)
streams={g:(out/f'gpu_{g}.jsonl').open('w') for g in gpus}; counts={g:0 for g in gpus}
for g,a,rs in groups:
 for r in rs: streams[g].write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n'); counts[g]+=1
for f in streams.values(): f.close()
shards=[]
for g in gpus:
 p=out/f'gpu_{g}.jsonl'; h=hashlib.sha256(p.read_bytes()).hexdigest(); shards.append(dict(gpu=g,chunks=counts[g],audio_seconds=sum(json.loads(l)['duration'] for l in p.open()),manifest_sha256=h))
prep=dict(input=str(src),files=len(files),raw_rows=len(rows),unique_chunks=len(rows),invalid_rows=len(invalid),missing_audio=[],recording_channels=len(by),audio_seconds=sum(x['duration'] for x in rows),shards=shards,source_batches=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],denominator='sum unique enhanced audio chunk durations')
(out/'preparation.json').write_text(json.dumps(prep,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(prep,ensure_ascii=False,indent=2))
