"""Create an unrun experiment record; never fabricate results or past events."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,json,re
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--name',required=True);p.add_argument('--parent',required=True);a=p.parse_args()
if not re.fullmatch(r'EFCIL-[A-Za-z0-9-]+',a.id):raise SystemExit('ID must start EFCIL- and contain letters, digits or hyphens')
registry=json.loads((ROOT/'experiments/registry.json').read_text())
parents={r['id'] for r in registry}
parents.update(x.parent.name for x in (ROOT/'experiments').glob('*/experiment.json'))
if a.parent not in parents:raise SystemExit('unknown parent ID')
dest=ROOT/'experiments'/a.id
if dest.exists():raise SystemExit('ID exists; refusing overwrite')
date=datetime.now(ZoneInfo('Asia/Seoul')).date().isoformat()
for d in ('code','results'): (dest/d).mkdir(parents=True,exist_ok=True)
record={'id':a.id,'name':a.name,'created_date_kst':date,'parent':a.parent,'status':'planned','hypothesis':None,'problem':None,'method_difference':None,'code':str(dest.relative_to(ROOT)/'code'),'config':str(dest.relative_to(ROOT)/'config.json'),'result_path':str(dest.relative_to(ROOT)/'results'),'local_artifacts':'new_runs/'+a.id,'command':None,'dataset':None,'task_partition':None,'seed':None,'environment':None,'source_commit':None,'evaluation_timing':None,'metrics':{},'analysis_document':str(dest.relative_to(ROOT)/'analysis.md')}
(dest/'experiment.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
(dest/'config.json').write_text('{}\n')
(dest/'analysis.md').write_text('# '+a.id+': '+a.name+'\n\nStatus: planned; no result.\n\n## Hypothesis / problem\n\n## Baseline and difference\n\n## Setup and exact command\n\n## Measured results and units\n\n## Interpretation and limitations\n\n## Next decision\n')
log=ROOT/'research_log'/f'{date}.md'
if not log.exists():log.write_text('# '+date+'\n\nNew metadata skeleton created for '+a.id+'. No experiment executed by this tool.\n\n## Problem\n\n## Code / method change\n\n## Executed experiments\n\n## Observed results\n\n## Issues\n\n## Next plan\n')
print(dest.relative_to(ROOT))
