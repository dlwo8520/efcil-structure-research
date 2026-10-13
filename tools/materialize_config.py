"""Write a separate local config with explicit workspace paths."""
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
if a.destination.exists():raise SystemExit('refusing to overwrite destination')
s=a.source.read_text().replace('${RESEARCH_WORKSPACE}',str(a.workspace.resolve()))
if '"extends"' in s and a.destination.parent.resolve()!=a.source.parent.resolve():
    raise SystemExit('config has relative extends; place local copy beside source or materialize parent chain explicitly')
a.destination.parent.mkdir(parents=True,exist_ok=True);a.destination.write_text(s)
print(a.destination)
