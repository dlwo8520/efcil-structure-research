"""Record an intentional source/config update while retaining original lineage."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,csv,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--path',required=True)
p.add_argument('--experiment',required=True)
p.add_argument('--reason',required=True)
a=p.parse_args()
target=(ROOT/a.path).resolve()
try:relative=str(target.relative_to(ROOT))
except ValueError:raise SystemExit('source must be inside this repository')
if target.suffix not in {'.py','.sh','.json','.yaml','.yml'}:raise SystemExit('only source/config updates are supported')
if target.suffix in {'.json','.yaml','.yml'} and not any(v in target.parts for v in {'configs','exps_jsons'}):raise SystemExit('frozen result metadata cannot be resealed by this tool')
records=json.loads((ROOT/'experiments/registry.json').read_text())
records += [json.loads(p.read_text()) for p in (ROOT/'experiments').glob('*/experiment.json')]
if a.experiment not in {r['id'] for r in records}:raise SystemExit('unknown experiment ID')
manifest=ROOT/'provenance/file_manifest.csv'
with manifest.open(newline='') as f:rows=list(csv.DictReader(f))
row=next((r for r in rows if r['export_path']==relative),None)
if row is None:raise SystemExit('new files need no original-export reseal; record them in experiment metadata')
raw=target.read_bytes();current=hashlib.sha256(raw).hexdigest()
if current==row['export_sha256']:raise SystemExit('no source change to record')
event={'date_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'experiment':a.experiment,'path':relative,'previous_export_sha256':row['export_sha256'],'new_sha256':current,'original_sha256':row['original_sha256'],'reason':a.reason}
row['export_sha256']=current;row['bytes']=len(raw);row['transformation']+='; intentional source update '+a.experiment
with manifest.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (ROOT/'provenance/code_updates.jsonl').open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
print(relative,'updated; original lineage retained. Run relevant tests and update CHANGELOG before commit.')
