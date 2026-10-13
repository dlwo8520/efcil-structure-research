"""Restore exact upstream git checkout; keep exported source as a local backup."""
from pathlib import Path
import csv,hashlib,shutil,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[1]
vendor=ROOT/'ia10_bob_vs_strong_full/vendor/BeyondObjects'
commit='8bc17e0537991f6bb5c0399339bddb3d1418d220'
backup=ROOT/'local-artifacts/BeyondObjects-export'
if (vendor/'.git').exists():
    if subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()!=commit:raise SystemExit('unexpected vendor revision')
    if subprocess.check_output(['git','-C',str(vendor),'status','--porcelain'],text=True).strip():raise SystemExit('vendor checkout is modified')
    print('exact clean upstream checkout already present');raise SystemExit(0)
if backup.exists():raise SystemExit('backup already exists; inspect manually')
for r in csv.DictReader((ROOT/'provenance/file_manifest.csv').open()):
    if r['export_path'].startswith('ia10_bob_vs_strong_full/vendor/BeyondObjects/'):
        if hashlib.sha256((ROOT/r['export_path']).read_bytes()).hexdigest()!=r['export_sha256']:raise SystemExit('exported vendor has local changes; refusing move')
with tempfile.TemporaryDirectory() as tmp:
    clone=Path(tmp)/'BeyondObjects'
    subprocess.run(['git','clone','https://github.com/princetonvisualai/BeyondObjects.git',str(clone)],check=True)
    subprocess.run(['git','-C',str(clone),'checkout','--detach',commit],check=True)
    backup.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(vendor,backup)
    shutil.move(clone,vendor)
print('restored exact upstream checkout; source export retained in local-artifacts')
