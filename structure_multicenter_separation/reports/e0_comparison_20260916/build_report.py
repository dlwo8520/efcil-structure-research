#!/usr/bin/env python3
"""Rebuild the audited seed-1993 report from saved records; no training or GPU use.

Requires Python >=3.9, numpy, matplotlib. This is deliberately a snapshot report:
other seeds are inventoried but never silently pooled into seed-1993 conclusions.
"""
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[3])))
import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, Patch
from matplotlib import font_manager
import numpy as np

METHODS = [f'E{i}' for i in range(8)]
SEED = 1993
EPS = 1e-10
NAMES = ['Single', 'Multi', 'Single + Iso', 'Single + Diag', 'Single + Full',
         'Multi + Iso', 'Multi + Diag', 'Multi + Full']
COLORS = ['#222222', '#377eb8', '#e68a2e', '#984ea3', '#999999', '#009e73', '#d55e00', '#cc79a7']
CMAP = LinearSegmentedColormap.from_list('delta', ['#c64242', '#eeeeee', '#248b53'])
ABS_CMAP = plt.get_cmap('Blues')
UNIT = 'ImageNet-A · seed 1993 · n=1 · SD=N/A'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
ROOT, OUT = args.project_root.resolve(), args.output_dir.resolve()
DATA = ROOT / 'outputs_recording_v2/imageneta'
for sub in ['figures', 'tables']:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False, 'svg.fonttype': 'path',
                     'figure.dpi': 120, 'savefig.dpi': 170})
sources, checks, inventory, config_diffs = {}, [], [], []


def track(path):
    path = Path(path)
    if path.exists() and str(path) not in sources:
        sources[str(path)] = {'bytes': path.stat().st_size,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return path


def js(path, default=None):
    return json.loads(track(path).read_text()) if path.exists() else default


def rows(path):
    return list(csv.DictReader(track(path).open(encoding='utf-8'))) if path.exists() else []


def jsonl(path):
    return [json.loads(x) for x in track(path).read_text().splitlines() if x.strip()] if path.exists() else []


def num(x):
    return float(x) if x not in ('', None) else float('nan')


def check(name, ok, detail=''):
    checks.append({'check': name, 'passed': bool(ok), 'detail': str(detail)})
    if not ok:
        raise ValueError(f'{name}: {detail}')


def write_csv(name, records, fields=None):
    records = list(records)
    fields = fields or list(dict.fromkeys(k for r in records for k in r))
    with (OUT / 'tables' / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: ('N/A' if isinstance(v, float) and not math.isfinite(v) else v)
                        for k, v in r.items()})


def md_table(headers, records):
    def escape(x):
        return str(x).replace('|', '\\|').replace('\n', '<br>')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                       '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                      ['| ' + ' | '.join(escape(x) for x in r) + ' |' for r in records])


def fmt(v, signed=False):
    if not np.isfinite(v):
        return 'N/A'
    if abs(v) <= EPS:
        return '0.00'
    if abs(v) < .005:
        return f'{v:+.2e}' if signed else f'{v:.2e}'
    return f'{v:+.2f}' if signed else f'{v:.2f}'


def hms(seconds):
    seconds = round(seconds)
    return f'{seconds // 3600}:{seconds % 3600 // 60:02}:{seconds % 60:02}'


def save(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / 'figures' / f'{name}.{ext}', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def color(v, reverse=False):
    if abs(v) <= EPS:
        return '#aaaaaa'
    return '#248b53' if (v < 0 if reverse else v > 0) else '#c64242'


R = {}
for m in METHODS:
    d = DATA / m / f'seed_{SEED}'
    r = {k: js(d / f'{k}.json') for k in ['resolved_config', 'run_metadata', 'summary',
          'protocol_manifest', 'checkpoint_manifest', 'source_state', 'class_order']}
    check(m + ' complete', r['summary']['status'] == 'complete' and r['run_metadata']['status'] == 'complete')
    T = r['protocol_manifest']['num_tasks']
    check(m + ' tasks', r['summary']['completed_tasks'] == T)
    check(m + ' gaps', not r['run_metadata']['recording_gaps'])
    for k in ['task_accuracy', 'task_summary', 'class_accuracy', 'epoch_metrics',
              'cluster_summary', 'cluster_details', 'cluster_assignments', 'apr_diagnostics', 'task_metrics']:
        r[k] = rows(d / f'{k}.csv')
    for k in ['separation_cache', 'separation_task_end_diagnostics']:
        r[k] = jsonl(d / f'{k}.jsonl')
    track(d / 'training.log')
    r['regions'] = [js(d / f'region_statistics_task_{t}.json') for t in range(T)]
    checkpoint = d / 'checkpoint_latest.pt'
    check(m + ' latest checkpoint', checkpoint.is_file() and r['checkpoint_manifest']['checkpoint_state_id'] == f'task_end:task_{T}')
    r['checkpoint_bytes'] = checkpoint.stat().st_size
    a = np.full((T, T), np.nan)
    cells = [x for x in r['task_accuracy'] if x['eval_stage'] == 'task_final']
    check(m + ' triangle completeness', len(cells) == T * (T + 1) // 2 and
          len({(x['trained_task'], x['evaluated_task']) for x in cells}) == len(cells))
    for x in cells:
        t, i = int(x['trained_task']) - 1, int(x['evaluated_task']) - 1
        check(m + f' a({t+1},{i+1}) correct counts', abs(num(x['accuracy']) - 100 * int(x['num_correct']) / int(x['num_samples'])) < EPS)
        a[t, i] = num(x['accuracy'])
    r['a'] = a
    check(m + ' valid triangle', all(np.isfinite(a[t, i]) for t in range(T) for i in range(t+1)))
    stored_matrix = js(d / 'task_accuracy_matrix.json')
    stored_csv = rows(d / 'task_accuracy_matrix.csv')
    check(m + ' stored matrix shapes', len(stored_matrix)==T and len(stored_csv)==T)
    for t in range(T):
        check(m + f' T{t+1} stored matrix values',
              len(stored_matrix[t])==t+1 and all(abs(a[t,i]-stored_matrix[t][i])<EPS and
              abs(a[t,i]-num(stored_csv[t][f'eval_task_{i+1}']))<EPS for i in range(t+1)))
    task_rows = sorted([x for x in r['task_summary'] if x['eval_stage'] == 'task_final'], key=lambda x: int(x['task_id']))
    check(m + ' task summary complete', len(task_rows) == T and [int(x['task_id']) for x in task_rows] == list(range(1,T+1)))
    r['A'], r['old'], r['new'] = [], [], []
    for t, x in enumerate(task_rows, 1):
        rr = [q for q in cells if int(q['trained_task']) == t]
        correct, count = sum(int(q['num_correct']) for q in rr), sum(int(q['num_samples']) for q in rr)
        check(m + f' T{t} weighted A', correct == int(x['num_correct']) and count == int(x['num_samples']) and abs(100*correct/count-num(x['accuracy'])) < EPS)
        cc = [q for q in r['class_accuracy'] if int(q['trained_task']) == t and q['eval_stage'] == 'task_final']
        check(m + f' T{t} class counts', sum(int(q['num_correct']) for q in cc) == correct and sum(int(q['num_samples']) for q in cc) == count)
        for i in range(1,t+1):
            ids = set(r['protocol_manifest']['tasks'][i-1]['mapped_class_ids'])
            q = next(q for q in rr if int(q['evaluated_task']) == i)
            sub = [c for c in cc if int(c['mapped_class_id']) in ids]
            check(m + f' T{t}/T{i} class sum', sum(int(c['num_correct']) for c in sub) == int(q['num_correct']) and sum(int(c['num_samples']) for c in sub) == int(q['num_samples']))
        new = next(q for q in rr if int(q['evaluated_task']) == t)
        old_correct, old_count = correct-int(new['num_correct']), count-int(new['num_samples'])
        old_acc = 100*old_correct/old_count if old_count else np.nan
        check(m + f' T{t} new', abs(num(x['new_accuracy']) - num(new['accuracy'])) < EPS)
        check(m + f' T{t} old', (np.isnan(old_acc) and not x['old_accuracy']) or abs(old_acc-num(x['old_accuracy'])) < EPS)
        r['A'].append(100*correct/count); r['old'].append(old_acc); r['new'].append(num(new['accuracy']))
    r['F_by_task'] = [float(np.max(a[i:T-1, i])-a[T-1,i]) for i in range(T-1)]
    r['Inc'], r['Last'], r['F'] = float(np.mean(r['A'])), r['A'][-1], float(np.mean(r['F_by_task']))
    for key, oldkey in [('Inc', 'Inc-Acc'), ('Last', 'Last-Acc'), ('F', 'Forgetting')]:
        check(m + ' recalc '+key, abs(r[key]-r['summary'][oldkey]) < EPS)
    r['wall'] = (datetime.fromisoformat(r['run_metadata']['ended_at_utc']) - datetime.fromisoformat(r['run_metadata']['started_at_utc'])).total_seconds()
    r['final_classes'] = {int(x['original_class_id']): x for x in r['class_accuracy'] if int(x['trained_task']) == T and x['eval_stage'] == 'task_final'}
    R[m] = r

T = R['E0']['protocol_manifest']['num_tasks']
check('snapshot protocol', T == 5 and R['E0']['protocol_manifest']['dataset'] == 'imageneta', 'This narrative is audited for ImageNet-A 5-task.')
allowed_method = {'multi_center', 'old_new_separation', 'separation_covariance'}
identity = {'experiment_id', 'prefix', 'config', 'run_dir'}
for m, r in R.items():
    check(m + ' protocol equality', r['protocol_manifest'] == R['E0']['protocol_manifest'])
    for k in sorted(set(r['resolved_config']) | set(R['E0']['resolved_config'])):
        v, b = r['resolved_config'].get(k), R['E0']['resolved_config'].get(k)
        if v != b:
            config_diffs.append({'method': m, 'key': k, 'E0': json.dumps(b), 'value': json.dumps(v),
                                 'category': 'method' if k in allowed_method else 'identity' if k in identity else 'other_control'})
    expected_mc, expected_sep = m in ['E1','E5','E6','E7'], m not in ['E0','E1']
    check(m + ' method intent', r['resolved_config']['multi_center'] == expected_mc and r['resolved_config']['old_new_separation'] == expected_sep)
    workload = [(x['task_id'],x['stage'],x['epoch'],x['processed_samples'],x['optimizer_steps'],x['learning_rates']) for x in r['epoch_metrics']]
    ref = [(x['task_id'],x['stage'],x['epoch'],x['processed_samples'],x['optimizer_steps'],x['learning_rates']) for x in R['E0']['epoch_metrics']]
    check(m + ' workload equality', workload == ref)
    for x in r['epoch_metrics']:
        weighted=[]
        for loss in ['classification','scc','feature_distillation','logit_distillation','separation']:
            if x[loss+'_active']=='True':
                check(m + f" {x['stage']}/T{x['task_id']}/E{x['epoch']} {loss} weight",
                      abs(num(x[loss+'_raw'])*num(x[loss+'_coefficient'])-num(x[loss+'_weighted']))<1e-6)
                weighted.append(num(x[loss+'_weighted']))
        check(m + f" {x['stage']}/T{x['task_id']}/E{x['epoch']} total loss sum",
              abs(sum(weighted)-num(x['total_loss']))<1e-5)

for base in [ROOT/'outputs', ROOT/'outputs_recording_v2']:
    for d in sorted(base.glob('imageneta/E*/seed_*')):
        s, meta = js(d/'summary.json', {}), js(d/'run_metadata.json', {})
        included = d == DATA/d.parent.name/f'seed_{SEED}' and d.parent.name in R
        inventory.append({'path': str(d), 'method': d.parent.name, 'seed': d.name.split('_')[-1],
                          'status': s.get('status', meta.get('status', 'incomplete/legacy')),
                          'completed_tasks': s.get('completed_tasks', len(js(d/'task_metrics.json', []))),
                          'included': included, 'reason': 'common seed 1993' if included else 'legacy incomplete or outside this snapshot'})
for m in METHODS:
    for seed in [1996,1997]:
        if not (DATA/m/f'seed_{seed}').exists():
            inventory.append({'path': str(DATA/m/f'seed_{seed}'), 'method': m, 'seed': seed,
                              'status': 'missing', 'completed_tasks': 0, 'included': False, 'reason': 'planned remaining seed; no saved run'})
write_csv('run_inventory.csv', inventory)
write_csv('config_differences.csv', config_diffs)

metrics, task_data, cell_data, forgetting, class_data = [], [], [], [], []
for m,r in R.items():
    b = R['E0']; s = r['summary']; c = r['resolved_config']
    metrics.append({'method':m,'seed':SEED,'n':1,'sample_sd':'N/A','multi_center':c['multi_center'],
        'separation_covariance': c['separation_covariance'] if c['old_new_separation'] else 'none',
        'Inc_Acc':r['Inc'],'delta_Inc_pp':r['Inc']-b['Inc'],'Last_Acc':r['Last'],'delta_Last_pp':r['Last']-b['Last'],
        'Forgetting_pp':r['F'],'delta_Forgetting_pp':r['F']-b['F'], 'wall_seconds':r['wall'],
        'incremental_train_seconds':s['total_training_seconds'],
        'wall_ratio_vs_E0':r['wall']/b['wall'], 'peak_cuda_allocated_GiB':s['max_peak_cuda_allocated_bytes']/2**30,
        'peak_cuda_reserved_GiB':s['max_peak_cuda_reserved_bytes']/2**30,
        'final_statistics_MiB':s['final_total_statistics_bytes']/2**20,
        'peak_separation_cache_MiB':s['max_separation_cache_bytes']/2**20,
        'checkpoint_file_MiB':r['checkpoint_bytes']/2**20})
    for t in range(T):
        task_data.append({'method':m,'seed':SEED,'n':1,'task':t+1,'A':r['A'][t], 'delta_A_pp':r['A'][t]-b['A'][t],
                         'old':r['old'][t],'new':r['new'][t], 'delta_old_pp':r['old'][t]-b['old'][t],
                         'delta_new_pp':r['new'][t]-b['new'][t], 'sample_sd':'N/A'})
        for i in range(t+1):
            cell_data.append({'method':m,'seed':SEED,'n':1,'trained_task':t+1,'evaluated_task':i+1,
                              'accuracy':r['a'][t,i], 'paired_delta_pp':r['a'][t,i]-b['a'][t,i],
                              'accuracy_sample_sd':'N/A','paired_delta_sample_sd':'N/A'})
    for i,v in enumerate(r['F_by_task']):
        forgetting.append({'method':m,'seed':SEED,'evaluated_task':i+1,'forgetting_pp':v,'delta_pp':v-b['F_by_task'][i]})
    for cid,x in r['final_classes'].items():
        base = b['final_classes'][cid]
        check(m + f' class {cid} identity', x['class_name']==base['class_name'] and x['mapped_class_id']==base['mapped_class_id'] and x['num_samples']==base['num_samples'])
        class_data.append({'method':m,'seed':SEED,'original_class_id':cid,'mapped_class_id':x['mapped_class_id'],
            'class_name':x['class_name'],'test_samples':int(x['num_samples']),'accuracy':num(x['accuracy']),
            'baseline_accuracy':num(base['accuracy']),'delta_pp':num(x['accuracy'])-num(base['accuracy']),
            'correct_difference':int(x['num_correct'])-int(base['num_correct'])})
    for name,matrix in [('absolute',r['a']),('delta_E0',r['a']-b['a'])]:
        write_csv(f'{m}_{name}_matrix.csv', [{'trained_task':t+1, **{f'T{i+1}':matrix[t,i] if i<=t else '' for i in range(T)}} for t in range(T)])
for name,data in [('metrics_per_seed',metrics),('task_metrics_recomputed',task_data),('cells_per_seed_and_sd',cell_data),
                  ('forgetting_per_task',forgetting),('class_final_deltas',class_data)]:
    write_csv(name+'.csv',data)
for key in ['task_accuracy','task_summary','class_accuracy','epoch_metrics','apr_diagnostics','cluster_details','cluster_assignments']:
    write_csv('raw_'+key+'.csv',[x for r in R.values() for x in r[key]])

cluster_data, cluster_stats = [], []
for m in ['E1','E5','E6','E7']:
    r = R[m]; seen=set(); detail_index=defaultdict(list); assignment_index=defaultdict(list)
    for x in r['cluster_details']: detail_index[x['clustering_run_id']].append(x)
    for x in r['cluster_assignments']: assignment_index[x['clustering_run_id']].append(x)
    for x in r['cluster_summary']:
        uid=x['clustering_run_id']; check(m+' unique clustering '+uid,uid not in seen); seen.add(uid)
        counts, weights = json.loads(x['cluster_counts']),json.loads(x['cluster_ratios'])
        n,k=int(x['class_sample_count']),int(x['actual_k'])
        ass=assignment_index[uid]; det=detail_index[uid]
        check(uid+' counts/weights',sum(counts)==n and len(counts)==k and abs(sum(weights)-1)<EPS and all(abs(v-c/n)<EPS for v,c in zip(weights,counts)))
        check(uid+' assignment consistency',len(ass)==n and len({a['sample_id'] for a in ass})==n and
              [sum(int(a['cluster_id'])==j for a in ass) for j in range(k)]==counts)
        check(uid+' details',len(det)==k and sorted(int(a['cluster_sample_count']) for a in det)==sorted(counts))
        check(uid+' training sample counts',n==r['protocol_manifest']['train_samples_per_mapped_class'][x['mapped_class_id']])
        final=next(q for q in class_data if q['method']==m and q['original_class_id']==int(x['original_class_id']))
        cluster_data.append({**x,'largest_cluster_share':max(weights),'smallest_cluster_samples':min(counts),
                             'final_accuracy':final['accuracy'],'delta_final_pp':final['delta_pp'],
                             'test_samples':final['test_samples'],'counts_valid':True,'weights_valid':True})
    check(m+' 200 cluster classes',len(seen)==r['protocol_manifest']['num_classes'])
    sub=[q for q in cluster_data if q['experiment_id']==m]; ks=Counter(int(q['actual_k']) for q in sub)
    reasons=Counter(reason for q in sub for reason in json.loads(q['fallback_reasons']))
    shares=[q['largest_cluster_share'] for q in sub]
    cluster_stats.append({'method':m,'seed':SEED,'n':1,'K1_classes':ks[1],'K2_classes':ks[2],'K3_classes':ks[3],
        'reduced_K_classes':sum(int(q['actual_k'])<int(q['requested_k']) for q in sub),
        'min_cluster_samples':min(q['smallest_cluster_samples'] for q in sub),
        'max_share':max(shares),'mean_max_share':float(np.mean(shares)),
        'empty_final_clusters':sum(0 in json.loads(q['cluster_counts']) for q in sub),
        'singleton_final_clusters':sum(json.loads(q['cluster_counts']).count(1) for q in sub),
        'reasons':json.dumps(reasons),
        'descriptive_share_delta_correlation':float(np.corrcoef(shares,[q['delta_final_pp'] for q in sub])[0,1])})
write_csv('kmeans_all_classes.csv',cluster_data)
write_csv('kmeans_summary.csv',cluster_stats)

diag, intrusion, feature_space_audit, workload = [], [], [], []
for m,r in R.items():
    ee=r['epoch_metrics']; active=[x for x in ee if x['separation_active']=='True']
    def values(k, subset=ee): return [num(x[k]) for x in subset if x.get(k) not in ('',None)]
    def bounds(k,subset=ee):
        vv=values(k,subset); return (min(vv),max(vv)) if vv else (np.nan,np.nan)
    fmin,fmax=bounds('student_feature_norm'); smin,smax=bounds('separation_raw',active); gmin,gmax=bounds('separation_parameter_gradient_norm_first_batch',active)
    numeric_keys=['total_loss','classification_raw','scc_raw','separation_raw','student_feature_norm',
                  'separation_parameter_gradient_norm_first_batch','nearest_old_rho_mean','intrusion_ratio']
    nonfinite=sum(not np.isfinite(v) for k in numeric_keys for v in values(k))
    diag.append({'method':m,'seed':SEED,'n':1,'active_separation_epochs':len(active),
       'zero_separation_epochs':sum(abs(num(x['separation_raw']))<=EPS for x in active),
       'separation_min':smin,'separation_max':smax,'first_batch_sep_grad_min':gmin,'first_batch_sep_grad_max':gmax,
       'feature_norm_min':fmin,'feature_norm_max':fmax,'recorded_nonfinite_values':nonfinite,
       'nan_flags':sum(x['nan_or_inf_detected']=='True' for x in ee),
       'numeric_fallbacks':sum(int(x['numerical_fallback_count']) for x in ee),
       'cache_refresh_seconds':sum(x['refresh_seconds'] for x in r['separation_cache']),
       'cache_factorization_fallbacks':sum(x['factorization_fallbacks'] for x in r['separation_cache'])})
    for task in range(1,T+1):
        for stage in ['feature_training','classifier_retraining']:
            subset=[x for x in ee if int(x['task_id'])==task and x['stage']==stage]
            workload.append({'method':m,'seed':SEED,'task':task,'stage':stage,'epochs':len(subset),
                             'optimizer_steps':sum(int(x['optimizer_steps']) for x in subset),
                             'processed_samples':sum(int(x['processed_samples']) for x in subset)})
        sub=[x for x in active if int(x['task_id'])==task]
        ends=[x for x in r['separation_task_end_diagnostics'] if x['task']+1==task]
        if sub:
            intrusion.append({'method':m,'seed':SEED,'task':task,'first_epoch_intrusion':num(sub[0]['intrusion_ratio']),
                'last_epoch_intrusion':num(sub[-1]['intrusion_ratio']),
                'epoch_intrusion_change':num(sub[-1]['intrusion_ratio'])-num(sub[0]['intrusion_ratio']),
                'task_end_intrusion':ends[0]['intrusion_ratio'] if ends else np.nan,
                'task_end_rho_mean':ends[0]['nearest_old_rho_mean'] if ends else np.nan,
                'delta_A_pp':r['A'][task-1]-R['E0']['A'][task-1]})
    for x in r['separation_cache']:
        expected=f"task_{x['task']}:epoch_{x['epoch']}_cache:student_adapter_{2 if x['task']%2 else 1}:raw_768d"
        check(m+' cache target',x['target_feature_space']==expected)
        check(m+' cache boundary',x['boundary_policy']==r['resolved_config']['sep_boundary_policy'] and x['timing']==r['resolved_config']['sep_timing'] and x['covariance_view']==r['resolved_config']['separation_covariance'])
        feature_space_audit.append({'method':m,'kind':'epoch_cache','task_zero_based':x['task'],'epoch_zero_based':x['epoch'],
            'source':json.dumps(x['source_feature_space']),'target':x['target_feature_space'],'boundary':x['boundary_policy'],
            'note':'target naming and boundary match config; source tags are not proof of tensor correctness'})
    for x in r['apr_diagnostics']:
        feature_space_audit.append({'method':m,'kind':'task_final_apr','task_one_based':x['task_id'],
            'source':x['source_feature_space'],'target':x['target_feature_space'],
            'note':'source tag is coarse and uses another task numbering convention; see report'})
    for idx,region in enumerate(r['regions']):
        if region is not None:
            check(m+f' region file task {idx}',region['task']==idx)
        else:
            check(m+f' region file task {idx} inapplicable',not r['resolved_config']['multi_center'] and not r['resolved_config']['old_new_separation'])
write_csv('numerical_diagnostics.csv',diag)
write_csv('intrusion_by_task.csv',intrusion)
write_csv('feature_space_audit.csv',feature_space_audit)
write_csv('workload.csv',workload)
pairs=[('E1','E0'),('E2','E0'),('E3','E0'),('E4','E0'),('E5','E1'),('E6','E1'),('E7','E1'),
       ('E5','E2'),('E6','E3'),('E7','E4'),('E3','E2'),('E4','E2'),('E4','E3'),('E6','E5'),('E7','E5'),('E7','E6')]
contrasts=[]
for m,b in pairs:
    contrasts.append({'comparison':m+'−'+b,'seed':SEED,'n':1,'delta_Inc_pp':R[m]['Inc']-R[b]['Inc'],
         'delta_Last_pp':R[m]['Last']-R[b]['Last'],'delta_F_pp':R[m]['F']-R[b]['F'],
         'factor_device_differs':R[m]['resolved_config']['sep_factor_device']!=R[b]['resolved_config']['sep_factor_device'],
         'source_code_identity':'unverified; working-tree hashes differ'})
write_csv('controlled_contrasts.csv',contrasts)

# Plotting. All comparison heatmaps share a symmetric range from unrounded cells.
vmax=max(abs(x['paired_delta_pp']) for x in cell_data) or 1.
vmax=math.ceil(vmax*10)/10
norm=Normalize(-vmax,vmax); anorm=Normalize(0,100)
def heat(ax,m,kind,small=False):
    a=R[m]['a']; delta=a-R['E0']['a']
    for t in range(T):
        for i in range(T):
            if i>t: continue
            absolute=kind=='absolute' or (kind=='A' and t==i)
            v=a[t,i] if absolute else delta[t,i]
            missing=not np.isfinite(v)
            fill='#eeeeee' if missing else ABS_CMAP(anorm(v)) if absolute else '#eeeeee' if abs(v)<=EPS else CMAP(norm(v))
            ax.add_patch(Rectangle((i-.5,t-.5),1,1,facecolor=fill,edgecolor='white',linewidth=2,
                                   hatch='///' if missing else None))
            ax.text(i,t,fmt(v,not absolute),ha='center',va='center',fontsize=10 if small else 13,
                    color='white' if absolute and v>65 else '#111111')
    ax.set(xlim=(-.5,T-.5),ylim=(T-.5,-.5),xticks=range(T),yticks=range(T),
           xticklabels=[f'T{i+1}' for i in range(T)],yticklabels=[f'T{i+1}' for i in range(T)],
           xlabel='평가 대상 task',ylabel='학습 완료 task')
    ax.set_aspect('equal'); ax.set_title(f'{m} · {NAMES[METHODS.index(m)]}'+(' 절대 정확도' if kind=='absolute' else ' vs E0'))
    for spine in ax.spines.values(): spine.set_visible(False)

for kind in ['A','B']:
    subtitle='대각선: 절대 정확도(%) / 아래 삼각형: E0 대비 차이(pp)' if kind=='A' else '대각선 포함 전체 유효 영역: E0 대비 차이(pp)'
    for m in METHODS[1:]:
        fig,ax=plt.subplots(figsize=(7,6.3),layout='constrained'); heat(ax,m,kind)
        fig.suptitle(UNIT+'\n'+subtitle,fontsize=10)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=CMAP),ax=ax,label='Δ accuracy (pp)',shrink=.8)
        if kind=='A': ax.legend(handles=[Patch(facecolor=ABS_CMAP(.8),label='대각선: 절대 정확도(%)')],loc='upper center',bbox_to_anchor=(.5,-.16))
        save(fig,f'heatmap_{kind}_{m}')
    fig,axes=plt.subplots(2,4,figsize=(19,10.2),layout='constrained')
    for ax,m in zip(axes.flat,METHODS[1:]): heat(ax,m,kind,True)
    axes.flat[-1].axis('off'); axes.flat[-1].text(.05,.65,subtitle.replace(' / ','\n')+'\n\n초록: 관측 개선 / 빨강: 관측 저하\n회색: 0 / N/A+빗금: 기록 누락\n흰 공백: 미래 task\n통계적 유의성을 나타내지 않음',va='center',wrap=True)
    fig.suptitle(UNIT+' · E0 대비 종합 비교 '+kind)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=CMAP),ax=axes,label='Δ accuracy (pp)',shrink=.75)
    save(fig,'heatmap_'+kind+'_overview')
fig,ax=plt.subplots(figsize=(7,6),layout='constrained');heat(ax,'E0','absolute')
fig.suptitle(UNIT+' · E0 절대 정확도 행렬')
fig.colorbar(plt.cm.ScalarMappable(norm=anorm,cmap=ABS_CMAP),ax=ax,label='Accuracy (%)');save(fig,'E0_absolute')

fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for ax,key,label in zip(axes,['Inc','Last'],['Δ Inc-Acc','Δ Last-Acc']):
    vv=[R[m][key]-R['E0'][key] for m in METHODS[1:]]
    ax.bar(METHODS[1:],vv,color=[color(v) for v in vv]);ax.axhline(0,color='#222',lw=1)
    for i,v in enumerate(vv): ax.annotate(fmt(v,True),(i,v),xytext=(0,5 if v>=0 else -13),textcoords='offset points',ha='center')
    ax.set(title=label+' vs E0',ylabel='pp');ax.margins(y=.25)
fig.suptitle(UNIT+' · 정확도 변화 (양수 개선)');save(fig,'performance_deltas')

def lines(ax,key,delta=False,final=False):
    for j,m in enumerate(METHODS):
        yy=R[m]['a'][-1] if final else np.array(R[m][key])
        if delta: yy=yy-(R['E0']['a'][-1] if final else np.array(R['E0'][key]))
        ax.plot(range(1,T+1),yy,label=m,color=COLORS[j],lw=2.8 if j==0 else 1.4,
                marker=['o','s','^','D','x','v','P','+'][j],ls='--' if m in ['E4','E7'] else '-')
    if delta: ax.axhline(0,color='#777',lw=.8)
    ax.set(xticks=range(1,T+1),xlabel='평가 대상 task' if final else '학습 완료 task',ylabel='Δ accuracy (pp)' if delta else 'Accuracy (%)')
    ax.grid(alpha=.2)

for delta in [False,True]:
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained');lines(ax,'A',delta);ax.legend(ncol=4)
    ax.set_title(UNIT+'\nseen 전체 정확도 A(t)'+(' − E0' if delta else ''));save(fig,'seen_delta' if delta else 'seen_accuracy')
fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
for row,delta in enumerate([False,True]):
    for col,key in enumerate(['old','new']):
        lines(axes[row,col],key,delta);axes[row,col].set_title(key+' accuracy'+(' − E0' if delta else ''))
axes[0,0].legend(ncol=4,fontsize=8);fig.suptitle(UNIT+' · old T1=N/A');save(fig,'old_new_accuracy')
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,delta in zip(axes,[False,True]): lines(ax,'',delta,True)
axes[0].legend(ncol=4,fontsize=8);fig.suptitle(UNIT+f' · 최종 T{T} 완료 후 평가 task별 정확도');save(fig,'final_task_accuracy')
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
axes[0].bar(METHODS,[R[m]['F'] for m in METHODS],color=COLORS);axes[0].set(title='Forgetting (낮을수록 좋음)',ylabel='pp')
vv=[R[m]['F']-R['E0']['F'] for m in METHODS[1:]];axes[1].bar(METHODS[1:],vv,color=[color(v,True) for v in vv]);axes[1].axhline(0,color='#222')
for i,v in enumerate(vv):axes[1].annotate(fmt(v,True),(i,v),xytext=(0,5 if v>=0 else -13),textcoords='offset points',ha='center')
axes[1].set(title='F_method − F_E0 (음수 개선)',ylabel='pp');axes[1].margins(y=.25);fig.suptitle(UNIT);save(fig,'forgetting')

fig,ax=plt.subplots(figsize=(8,4.7),layout='constrained');bottom=np.zeros(4)
for k,c in zip([1,2,3],['#bdd7e7','#6baed6','#2171b5']):
    vv=[x[f'K{k}_classes'] for x in cluster_stats];ax.bar([x['method'] for x in cluster_stats],vv,bottom=bottom,color=c,label=f'K={k}')
    for i,v in enumerate(vv):ax.text(i,bottom[i]+v/2,str(v),ha='center',va='center',color='white' if k==3 else 'black')
    bottom+=vv
ax.set(ylabel='Class 수',title=UNIT+' · 실제 K별 class 수');ax.legend();save(fig,'kmeans_K_distribution')
for m in ['E1','E5','E6','E7']:
    for task in range(1,T+1):
        sub=sorted([q for q in cluster_data if q['experiment_id']==m and int(q['task_id'])==task],key=lambda q:int(q['original_class_id']))
        fig,ax=plt.subplots(figsize=(13,4.6),layout='constrained');bottom=np.zeros(len(sub))
        for k,c in zip(range(3),['#377eb8','#ffb74d','#66bb6a']):
            vv=[(json.loads(q['cluster_ratios'])+[0]*3)[k]*100 for q in sub]
            ax.bar(range(len(sub)),vv,bottom=bottom,color=c,label=f'local cluster {k}');bottom+=vv
        ax.set(xticks=range(len(sub)),xticklabels=[q['original_class_id'] for q in sub],ylim=(0,100),ylabel='Train sample 비율 (%)',xlabel='원본 ImageFolder class ID',title=UNIT+f' · {m} · 생성 T{task} (40 classes)')
        ax.tick_params(axis='x',rotation=90,labelsize=8)
        ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.23),frameon=False)
        save(fig,f'clusters_{m}_T{task}')

loss_keys=['total_loss','classification_raw','classification_weighted','scc_raw','scc_weighted','separation_raw','separation_weighted']
for task in range(1,T+1):
    fig,axes=plt.subplots(2,4,figsize=(17,8.5),layout='constrained')
    for ax,key in zip(axes.flat,loss_keys):
        present=False
        for j,m in enumerate(METHODS):
            rr=[x for x in R[m]['epoch_metrics'] if int(x['task_id'])==task and x['stage']=='feature_training' and x.get(key)]
            if rr:
                present=True;ax.plot([int(x['epoch']) for x in rr],[num(x[key]) for x in rr],color=COLORS[j],label=m,lw=2 if j==0 else 1.2,ls='--' if m in ['E4','E7'] else '-')
        ax.set(title=key,xlabel='Epoch',ylabel='Loss');ax.grid(alpha=.2)
        if not present:ax.text(.5,.5,'N/A: 비활성',transform=ax.transAxes,ha='center')
    axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='center',ncol=2)
    fig.suptitle(UNIT+f' · T{task} feature_training · raw/weighted 분리');save(fig,f'loss_feature_T{task}')
fig,axes=plt.subplots(1,T-1,figsize=(16,4),layout='constrained')
for ax,task in zip(np.atleast_1d(axes),range(2,T+1)):
    for j,m in enumerate(METHODS):
        rr=[x for x in R[m]['epoch_metrics'] if int(x['task_id'])==task and x['stage']=='classifier_retraining']
        ax.plot([int(x['epoch']) for x in rr],[num(x['total_loss']) for x in rr],label=m,color=COLORS[j],ls='--' if m in ['E4','E7'] else '-')
    ax.set(title=f'T{task}',xlabel='Epoch',ylabel='CE loss (raw=weighted=total)');ax.grid(alpha=.2)
axes[0].legend(ncol=2,fontsize=8);fig.suptitle(UNIT+' · classifier_retraining');save(fig,'loss_classifier')

fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,key in zip(axes.flat,['intrusion_ratio','student_feature_norm','separation_parameter_gradient_norm_first_batch','nearest_old_rho_mean']):
    for j,m in enumerate(METHODS):
        rr=[x for x in R[m]['epoch_metrics'] if x['stage']=='feature_training']
        xx=[i+1 for i,x in enumerate(rr) if x.get(key)]; yy=[num(x[key]) for x in rr if x.get(key)]
        if yy:ax.plot(xx,yy,label=m,color=COLORS[j],ls='--' if m in ['E4','E7'] else '-')
    for boundary in [30,45,60,75]:ax.axvline(boundary+.5,color='#ccc',lw=.7)
    ax.set(title=key,xlabel='누적 feature epoch (T 경계 수직선)');ax.grid(alpha=.2)
    if key=='nearest_old_rho_mean':ax.set_yscale('log')
axes[1,0].legend(ncol=3,fontsize=8);fig.suptitle(UNIT+' · 학습 중 진단 (gradient: epoch 첫 batch)');save(fig,'numerical_diagnostics')
fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
for ax,key,label in zip(axes,['wall_seconds','peak_cuda_allocated_GiB','final_statistics_MiB'],['실행 시간 (분)','Peak CUDA allocated (GiB)','최종 statistics (MiB)']):
    vv=[x[key]/60 if key=='wall_seconds' else x[key] for x in metrics]
    ax.bar(METHODS,vv,color=COLORS);ax.set(title=label);ax.tick_params(axis='x',rotation=30)
fig.suptitle(UNIT+' · 시간 및 메모리');save(fig,'resource_costs')

# Report text and image captions are computed from this audited snapshot.
text=[]
def add(s):text.append(s)
def picture(name,caption):add(f'![{name}](figures/{name}.png)\n\n{caption}  \\n[SVG 원본](figures/{name}.svg)'.replace('\\n','\n'))
def extrema(m):
    a=R[m]['a']-R['E0']['a'];valid=[(a[t,i],t+1,i+1) for t in range(T) for i in range(t+1)]
    lo,hi=min(valid),max(valid)
    return f'전체 유효 셀에서 최대 변화는 T{hi[1]}→T{hi[2]}의 {fmt(hi[0],True)} pp, 최소 변화는 T{lo[1]}→T{lo[2]}의 {fmt(lo[0],True)} pp다.'

add('# ImageNet-A 5-task: E0 대비 E1–E7 결과 보고서\n\n작성 기준: 저장된 pilot seed 1993, 모든 비교 n=1. 정확도는 %, 차이는 pp. 학습·재평가는 수행하지 않고 기록을 재집계했다.')
add('## 1. 결과 요약\n\n**E0보다 Inc-Acc와 Last-Acc가 모두 높은 방법은 없다.** E3는 Last-Acc가 +0.06583 pp(최종 정답 1개 증가)지만 Inc-Acc는 −0.06574 pp다. E3의 Forgetting은 7.53367 pp로 E0보다 0.31077 pp 낮으며 관측값 중 가장 낮다. E5는 Last-Acc가 E0와 같고 Inc-Acc는 −0.08788 pp이며, E6는 T4에서 +0.56090 pp였다가 최종 −0.65833 pp로 바뀐다.\n\nE4의 정확도 행렬은 E0와, E7의 행렬은 E1과 동일하다. 두 full-covariance 실험은 separation 활성 60개 epoch 모두 loss·intrusion·기록된 separation gradient가 0이다. 이는 full separation의 이득을 확인한 결과가 아니다. **seed가 하나이므로 표본 표준편차와 유의성은 확인할 수 없다.**')
add('## 2. 실험 목록 및 비교 가능성\n\n모든 포함 run은 ImageNet-A **200 classes / 5 tasks / task당 40 classes**, seed 1993이며 완료 상태다. 목표 ImageNet-A 5-task와 일치한다. 공통 완료 seed가 1993 하나여서 전체 비교와 각 E0 paired 비교의 seed 집합은 모두 동일하다.')
add(md_table(['방법','실제 Multi-Center','실제 Separation','분해 장치','상태 / task','seed / n'],[
    [m,'On' if R[m]['resolved_config']['multi_center'] else 'Off',
     R[m]['resolved_config']['separation_covariance'] if R[m]['resolved_config']['old_new_separation'] else 'Off',
     R[m]['resolved_config']['sep_factor_device'],'complete / 5','1993 / 1'] for m in METHODS]))
protocol=R['E0']['protocol_manifest']
add(f"비교 프로토콜 파일은 8개 run 모두 동일하다. Train {protocol['splits']['train']['file_count']:,}장 / test {protocol['splits']['test']['file_count']:,}장, task별 test 표본 수는 "+', '.join(str(int(x['num_samples'])) for x in R['E0']['task_accuracy'] if x['trained_task']==str(T))+'장이다. 클래스 순서·원본 ID remapping·split 내용 SHA-256과 class별 표본 수까지 일치하며, 기록된 split 간 경로·이름·내용 중복은 모두 0이다. 원본 ID는 이 데이터의 ImageFolder ID이며 ImageNet-1K 전체 인덱스로 해석하지 않는다.')
add('백본은 `pretrained_vit_b16_224_in21k_adapter`, batch=48, SGD lr=0.01, init 30 epochs + incremental 15×4 epochs다. SCC 계수 5, APR On, T2–T5 classifier retraining 각 5 epochs, replay는 **class당 256개/epoch**이며 multi-center에서도 cluster당 256개가 아니다. 모든 run의 epoch별 sample 수·optimizer step·LR가 일치한다(각 run 총 110 epoch 기록, 5,155 optimizer steps, stage 합산 828,580 processed samples). feature 학습과 retraining 표본을 합친 수로 고유 이미지 수가 아니다.')
add('평가는 classifier retraining을 포함한 `incremental_train()` 종료 뒤의 `task_final`이다. 현재 평가 코드는 모든 seen-class logits에서 top-k를 구한 뒤 정답 label로 집계 대상을 나누므로 **single-head CIL**이다. task별 후보 제한이나 best-test epoch 선택은 사용하지 않는다. checkpoint는 `task_end:task_5` latest 상태이며, 이전 epoch/task에 기록된 같은 경로는 나중에 덮어쓰여 현재 과거 모델을 가리키지 않는다. 이번 분석은 저장된 task-final counts를 사용했고 checkpoint 역직렬화나 재평가는 하지 않았다.')
add('**통제 한계:** 방법 변경은 Multi-Center, Separation 활성화, covariance 방식이다. 그 외 E4·E7만 `sep_factor_device=cuda`, 나머지는 `cpu`여서 covariance 비교의 실행 장치 조건이 다르다(Off 실험에서는 해당 설정 비활성). Git commit은 동일하지만 모든 run의 `working_tree_sha256`은 다르고 dirty 상태다. 해시에 증가하는 output 파일도 포함되어 있어 해시 차이가 곧 코드 변경을 뜻하지는 않지만, 당시 소스 파일별 스냅샷이 없어 코드 동일성도 확정할 수 없다. 따라서 아래 비교는 **기록상 protocol·학습량이 맞는 paired 관찰**이며 완전히 통제된 방법론 인과 비교라고 단정하지 않는다.')
add('제외: 기존 `outputs/imageneta/E0/seed_1993`는 최종 summary 없이 1 task만 기록된 legacy 중단 run이다. `outputs_recording_v2`의 seed 1996·1997은 E0–E7 모두 저장 run이 없다(16개 미실행/누락). 이를 실패로 단정하지 않았다. [전체 run 및 제외 목록](tables/run_inventory.csv), [config 차이](tables/config_differences.csv), [실제 학습량](tables/workload.csv)에 보존했다.')
add('## 3. 전체 성능 표\n\n공통 seed={1993}, n=1. SD=N/A이며 0으로 대체하지 않았다. ΔInc·ΔLast는 양수가 개선, ΔForgetting=F_method−F_E0는 **음수가 개선**이다. 시간은 시작–종료 wall time, 통계 메모리는 final persistent statistics이며 checkpoint 디스크 크기·GPU peak와 다른 값이다.')
add(md_table(['방법','MC','Sep cov','n','Inc-Acc %','ΔInc pp','Last-Acc %','ΔLast pp','Forgetting pp','ΔF pp','시간 h:mm:ss','통계 MiB'],[
    [q['method'],'On' if q['multi_center'] else 'Off',q['separation_covariance'],1,fmt(q['Inc_Acc']),fmt(q['delta_Inc_pp'],True),fmt(q['Last_Acc']),fmt(q['delta_Last_pp'],True),fmt(q['Forgetting_pp']),fmt(q['delta_Forgetting_pp'],True),hms(q['wall_seconds']),f"{q['final_statistics_MiB']:.1f}"] for q in metrics]))
add('수치 검산: task별 correct/sample 합산 A(t), class→task→seen counts, old/new 정확도, Inc-Acc·Last-Acc·Forgetting 모두 저장 summary와 1e−10 허용오차 내 일치했다. 따라서 기존 지표와 요청한 정의 사이에 이번 데이터의 수치 차이는 없다. Inc-Acc 기준 E0=E4가 공동 최고이고 Last-Acc 기준 E3가 가장 높으나, n=1의 관측 순위다. [정밀 seed 원자료 및 비용](tables/metrics_per_seed.csv).')
add('## 4. Inc-Acc / Last-Acc 변화 및 task별 정확도\n\n아래 모든 그림은 seed 1993, n=1, seed 간 SD는 N/A다. 초록/빨강은 관측 방향이며 통계적 유의성이 아니다.')
picture('performance_deltas','0선 위의 초록 막대는 E0 대비 정확도 개선이다. E3만 Last-Acc가 소폭 양수이고, Inc-Acc는 어느 추가 방법도 E0를 넘지 못한다. E4는 두 지표가 같으며 E5는 마지막만 같다.')
picture('seen_accuracy','A(t)는 각 시점의 모든 seen test sample을 표본 수로 가중한 정확도다. 모든 방법의 T1은 83.89%로 같고 이후 seen classes가 늘면서 곡선이 내려간다. 평가 집합 자체가 달라지므로 이 하락만으로 forgetting을 측정할 수 없다.')
picture('seen_delta','E0를 0으로 맞추면 T2–T3의 음수와 T4에서 E2·E3·E5·E6의 양수가 드러난다. E6는 T4의 +0.56 pp가 T5의 −0.66 pp로 역전되어 중간 task만 보면 결론이 달라진다. E3도 Last-Acc의 작은 이득과 달리 전체 시점 평균은 음수다.')
add(md_table(['방법']+[f'A(T{t}) %' for t in range(1,T+1)],[[m]+[fmt(x) for x in R[m]['A']] for m in METHODS]))
add('## 5. 삼각형 heatmap\n\n행=학습 완료 task, 열=평가 대상 task. 미래 task는 흰 공백, 유효 셀의 기록 누락은 N/A+빗금이며 이번 8개 run은 유효 셀 누락이 없다. A의 파란 대각선은 절대 정확도이고 B는 대각선까지 E0 차이다. 모든 Δ 색상 범위는 **'+f'−{vmax:.1f}~+{vmax:.1f} pp'+'**로 공통이며 0은 연회색이다. 색 판단은 반올림 전 값, 0 판정 허용오차 1e−10 pp다.')
picture('E0_absolute','모든 유효 셀은 E0의 절대 정확도(%)이고 파란색은 0–100% 고정 범위다. 첫 task의 정확도는 처음 83.89%에서 마지막 '+fmt(R['E0']['a'][-1,0])+'%로 바뀐다. 이 기준 행렬과 동일 seed·동일 셀을 빼서 아래 차이를 계산했다.')
picture('heatmap_A_overview','파란 대각선끼리는 새 task의 절대 정확도를 비교하고, 아래 삼각형의 부호로 이전 task 보존 변화를 읽는다. E4의 아래 삼각형은 모두 0이며 E2·E3·E5·E6은 task에 따라 개선과 저하가 섞인다. 대각선의 개선 여부는 다음 B에서 확인해야 한다.')
picture('heatmap_B_overview','대각선을 포함해 같은 셀의 E0 차이를 표시한다. E2·E3·E5·E6은 T2 새 task 대각선이 음수인 반면 당시 T1 보존은 양수여서 trade-off가 드러난다. E4의 전 영역이 0이고 E7은 E1과 동일한 패턴이다.')
for m in METHODS[1:]:
    add(f'### {m}: {NAMES[METHODS.index(m)]}')
    picture('heatmap_A_'+m,'대각선은 이 방법의 새 task 절대 정확도이고 아래는 E0 대비 pp 차이다. '+extrema(m)+' 하나의 seed에서 측정한 셀 변화로 일반화할 수 없다.')
    diag_delta=np.diag(R[m]['a']-R['E0']['a'])
    picture('heatmap_B_'+m,'대각선도 차이로 읽는다. 새 task 변화(T1→T5)는 '+', '.join(fmt(v,True) for v in diag_delta)+' pp다. '+('모든 셀이 같아 관측된 정확도 이득이 없다.' if np.nanmax(abs(R[m]['a']-R['E0']['a']))<=EPS else '양수 셀과 음수 셀을 함께 보아야 하며 평균 변화가 모든 task의 개선을 뜻하지 않는다.'))
add('## 6. Old/new 성능과 forgetting')
picture('old_new_accuracy','T1 old는 정의되지 않아 N/A로 비웠다. E3의 T5 old 정확도는 E0보다 '+fmt(R['E3']['old'][-1]-R['E0']['old'][-1],True)+' pp, new는 '+fmt(R['E3']['new'][-1]-R['E0']['new'][-1],True)+' pp로 이전 class 보존과 새 class 학습 간 trade-off가 관측된다. E6는 T5 old와 new가 모두 E0보다 낮다.')
picture('final_task_accuracy','왼쪽은 최종 모델의 각 task 정확도, 오른쪽은 같은 셀의 E0 차이다. 최종 task의 대각선(새 T5)에서 E3·E5·E6는 E0보다 낮고, E3의 seen 전체 소폭 이득은 old 부분에서 나온다. task마다 test 표본 수가 달라 오른쪽 다섯 점의 단순 평균은 ΔLast-Acc가 아니다.')
picture('forgetting','오른쪽 raw ΔF는 음수일 때 초록이며 정확도 그림과 개선 부호가 반대다. E3의 −0.31 pp가 가장 큰 감소이고 E5는 −0.05 pp로 소폭 감소한다. 이 값은 각 과거 task의 final 이전 최고 정확도와 final 정확도의 차이를 평균한 것으로, 음수라도 0으로 자르지 않는다.')
add(md_table(['방법']+[f'T{i+1} forgetting pp' for i in range(T-1)],[[m]+[fmt(v) for v in R[m]['F_by_task']] for m in METHODS]))
add('## 7. K-means 분할 현황 및 class별 상세 결과\n\nE1·E5·E6·E7만 K-means 적용, E0·E2·E3·E4는 **K-means 미적용**이다. single prototype을 K=1 K-means로 해석하지 않는다. 요청 K=3, K-means++, n_init=10, 최대 100회, 최소 cluster 크기=2다. 표본·unique feature 제한이나 empty/undersized 시 K를 줄여 재실행하며, cluster를 병합하거나 표본을 버리는 방식이 아니다. 별도 의미적 K 선택 절차는 없다.')
add(md_table(['방법','K=1 class','K=2 class','K=3 class','K 축소 class','최소 cluster N','최대 cluster 비중','평균 최대 비중'],[[q['method'],q['K1_classes'],q['K2_classes'],q['K3_classes'],q['reduced_K_classes'],q['min_cluster_samples'],f"{100*q['max_share']:.1f}%",f"{100*q['mean_max_share']:.1f}%"] for q in cluster_stats]))
picture('kmeans_K_distribution','각 막대의 높이는 200 classes이고 숫자는 실제 K별 class 수다. 요청 K=3을 모두 유지한 방법은 없으며 E1/E7은 28개, E5는 16개, E6는 21개 class에서 K가 줄었다. 이는 제약에 따른 유효 cluster 수이지 의미적 하위 유형의 개수 증거가 아니다.')
add('800개 class-clustering 기록과 23,924개 assignment를 대조했다. cluster별 sample 합=class sample 수, weight 합=1, assignment별 count 및 class train 표본 수가 모두 일치했다. 최종 유효 cluster의 empty·singleton은 모두 0이며 최소 cluster 크기는 2다. 중간 초기화에서 부적합한 cluster는 재시도될 수 있으므로 최종 0건과 시도 0건을 혼동하지 않는다. 동일 class의 APR 갱신은 새 clustering으로 세지 않았고 `clustering_run_id`가 중복되지 않는지 확인했다.')
add('클래스 이름은 저장된 WordNet synset 문자열(예: `n04442312`)을 그대로 사용한다. 의미 이름을 추측해 바꾸지 않았다. [전체 800개 class 표](tables/kmeans_all_classes.csv)는 요청한 ID·생성 task·이름·N·요청/실제 K·counts·비율·fallback 사유·feature 공간·checkpoint key 및 최종 Δ를 모두 포함한다. [Cluster 상세 원자료](tables/raw_cluster_details.csv), [assignment](tables/raw_cluster_assignments.csv), [요약 및 사유](tables/kmeans_summary.csv)를 함께 제공한다.')
for m in ['E1','E5','E6','E7']:
    add(f'### {m}: 생성 task별 전체 class 비율')
    for task in range(1,T+1):
        sub=[q for q in cluster_data if q['experiment_id']==m and int(q['task_id'])==task]
        reduced=sum(int(q['actual_k'])<3 for q in sub); maxshare=max(q['largest_cluster_share'] for q in sub)
        picture(f'clusters_{m}_T{task}',f'원본 class ID 순서로 생성 T{task}의 40개 class를 표시하며 각 막대 합은 100%다. 이 task에서 K 축소는 {reduced}개 class, 가장 큰 단일 cluster 비중은 {maxshare*100:.1f}%다. cluster 번호와 색은 해당 clustering 내부 식별자여서 다른 class·실험의 같은 색이 같은 의미를 뜻하지 않는다.')
add('### 최종 class 정확도와의 연결\n\n아래는 각 방법의 가장 큰 변화 예시이며 전체 class를 [class_final_deltas.csv](tables/class_final_deltas.csv)에 보존했다. class 연결은 원본 ID로 수행하고 학습 ID·이름·test N 일치까지 검산했다. 표본 수가 적은 class의 한 정답 변화는 큰 pp가 되므로 pp만으로 우선순위를 정하면 안 된다.')
examples=[]
for m in METHODS[1:]:
    sub=[q for q in class_data if q['method']==m];changed=sum(abs(q['delta_pp'])>EPS for q in sub)
    add(f"{m}: 개선 {sum(q['delta_pp']>EPS for q in sub)} / 저하 {sum(q['delta_pp']<-EPS for q in sub)} / 동일 {200-changed} classes; 최종 정답 변화 {sum(q['correct_difference'] for q in sub):+d}개.")
    for q in [min(sub,key=lambda q:q['delta_pp']),max(sub,key=lambda q:q['delta_pp'])]:
        cl=next((z for z in cluster_data if z['experiment_id']==m and int(z['original_class_id'])==q['original_class_id']),None)
        examples.append([m,q['original_class_id'],q['class_name'],q['test_samples'],fmt(q['delta_pp'],True),q['correct_difference'],cl['actual_k'] if cl else '미적용',cl['cluster_counts'] if cl else 'N/A'])
add(md_table(['방법','원본 ID','class','test N','Δ pp','정답 Δ','K','cluster N'],examples))
add('군집 최대 비중과 최종 class Δ의 기술적 Pearson 상관도 `kmeans_summary.csv`에 저장했다. K=1은 비중이 자동으로 1이고 class별 test N이 다르며 class 난이도도 통제하지 않았으므로 이 상관을 원인·유의성으로 해석하지 않는다.')
add('## 8. Loss · intrusion · 수치 안정성 · 비용\n\nfeature stage의 목적함수는 classification + 5×SCC(활성 task) + 1×separation(활성 방법)이다. 아래 raw와 weighted를 분리했으며 distillation은 비활성으로 값이 없어 곡선을 만들지 않았다. 목적함수가 다른 total loss의 크기로 방법 우열을 판단하지 않는다.')
for task in range(1,T+1):
    picture(f'loss_feature_T{task}',f'T{task}의 feature 학습만 표시하며 retraining과 epoch 축을 섞지 않았다. '+('T1에서는 SCC·separation이 비활성이어서 해당 패널은 N/A이며 초기 학습 곡선이 겹친다.' if task==1 else 'SCC weighted는 raw의 5배이고 separation은 계수 1이라 raw와 weighted가 같다. E4·E7의 separation 곡선은 전 epoch 0이다.')+' 값은 batch 크기로 가중한 epoch 평균이다.')
picture('loss_classifier','T2–T5의 synthetic replay classifier CE만 표시한다(T1에는 retraining 기록 없음). 이 stage의 classification raw=weighted=total이고 SCC·separation은 비활성이다. 같은 class당 replay 수와 5 epochs를 사용해도 replay 분포가 달라 곡선 차이를 최종 성능 우열로 곧바로 바꿀 수 없다.')
picture('numerical_diagnostics','세로선은 task 경계이며 rho는 로그 축이다. E4·E7은 intrusion·loss·첫 batch gradient가 모두 0이고 full rho가 매우 커 활성 hinge 영역 밖에 있는 기록과 일치한다. E2·E3·E5·E6에서는 feature norm 최대치가 E0보다 커지지만, 이 사실만으로 폭주나 원인을 확정하지 않는다.')
add(md_table(['방법','sep 활성/0 epoch','sep loss min–max','첫 batch grad min–max','feature norm min–max','NaN flag/비유한','수치 fallback'],[
 [q['method'],f"{q['active_separation_epochs']}/{q['zero_separation_epochs']}",f"{fmt(q['separation_min'])}–{fmt(q['separation_max'])}",f"{fmt(q['first_batch_sep_grad_min'])}–{fmt(q['first_batch_sep_grad_max'])}",f"{fmt(q['feature_norm_min'])}–{fmt(q['feature_norm_max'])}",f"{q['nan_flags']}/{q['recorded_nonfinite_values']}",q['numeric_fallbacks']] for q in diag]))
add('기록된 NaN/Inf flag·검사한 수치의 비유한 값·factorization fallback은 모두 0이다. gradient는 **epoch 첫 batch의 separation 항에 한정**되어 전체 목적함수나 모든 batch의 gradient 폭주를 배제할 수 없다. full covariance에서 큰 rho·0 loss의 기하적 원인은 covariance rank, 정규화, birth tau와 변환 후 feature의 관계를 추가 점검해야 하며 현재 기록만으로 버그를 확정하지 않는다.')
add(md_table(['방법','T','첫 epoch intrusion','마지막 epoch intrusion','task-end intrusion','task-end rho 평균','ΔA pp'],[[q['method'],q['task'],fmt(q['first_epoch_intrusion']),fmt(q['last_epoch_intrusion']),fmt(q['task_end_intrusion']),fmt(q['task_end_rho_mean']),fmt(q['delta_A_pp'],True)] for q in intrusion]))
add('intrusion은 0–1 비율이다. task-end 값은 새 task의 train feature에 대한 분포 진단이며 test 정확도가 아니다. 학습 epoch 평균과 task-end 재계산은 시점·cache가 달라 같은 측정처럼 합치지 않았다. E6의 task-end intrusion은 E5보다 낮지만 Last-Acc도 더 낮아 intrusion 감소가 성능 개선으로 일관되게 연결되지 않는다. covariance마다 거리와 경계가 달라 방법 간 intrusion 절대값 자체도 공통 척도의 성능 점수가 아니다. E0·E1에는 대응 intrusion 진단이 없어 E0 대비 감소량은 확인 불가다.')
add('### APR / cache feature 공간과 경계 감사\n\n모든 separation run에서 cache 60개가 기록되어 T2–T5의 매 feature epoch 갱신과 맞고, target은 `task_k:epoch_e_cache:student_adapter_*:raw_768d`다. config와 cache의 `epoch_apr_cache`, `birth_quantile_fixed`, covariance 설정을 대조했다. 현재 구현은 corrected mean/covariance를 target 공간에 배치하고 birth tau를 유지한다. 그러나 source 태그가 텐서 변환의 정확성을 보증하는 것은 아니다. 특히 APR CSV의 source는 `task_1:task_final` 같은 축약 태그이고 target은 0-based task 및 adapter를 포함해 **번호 규약과 정보량이 다르다**. single-center cache source 태그에는 과거 birth 공간도 남을 수 있어 문자열만으로 완전 일치를 주장하지 않는다. [전체 feature-space 감사표](tables/feature_space_audit.csv)에 원문을 보존했다.')
picture('resource_costs','왼쪽은 데이터 로딩·통계·학습·평가·저장을 포함한 run wall time이다. E4와 E7은 대응 E0/E1과 정확도가 같아도 실행 시간이 늘었고 full cache의 GPU 비용도 증가한다. GPU 동시 점유 기록이 없어 시간 차이 전체를 알고리즘 연산량 차이로 돌릴 수 없다.')
add(md_table(['방법','wall / E0','학습 함수 분','peak allocated GiB','peak reserved GiB','statistics MiB','max cache MiB','cache 갱신 분'],[
 [q['method'],f"{q['wall_ratio_vs_E0']:.2f}×",f"{q['incremental_train_seconds']/60:.2f}",f"{q['peak_cuda_allocated_GiB']:.2f}",f"{q['peak_cuda_reserved_GiB']:.2f}",f"{q['final_statistics_MiB']:.1f}",f"{q['peak_separation_cache_MiB']:.2f}",f"{next(d for d in diag if d['method']==q['method'])['cache_refresh_seconds']/60:.2f}"] for q in metrics]))
start=datetime.fromisoformat(R['E0']['run_metadata']['started_at_utc']);end=datetime.fromisoformat(R['E7']['run_metadata']['ended_at_utc'])
add(f"전체 E0 시작→E7 종료는 {hms((end-start).total_seconds())}, run wall 합은 {hms(sum(r['wall'] for r in R.values()))}다. `total_training_seconds`는 `incremental_train` 호출 시간으로 통계·APR·내부 평가 등이 포함되어 순수 GPU kernel 시간은 아니다. statistics는 tensor storage 추정 바이트이며 RAM peak/디스크 크기와 다르고 cache peak와 final statistics는 시점이 달라 단순 합으로 peak를 만들지 않았다.")
add('## 9. 관찰된 변화, 한계, 다음 검증\n\n아래는 동일 seed 차이를 직접 계산한 요인별 비교다. 공통 데이터·학습량은 확인했지만 소스 동일성은 미확정이고 full covariance 비교에는 CPU/CUDA factorization 차이가 있다. 이를 완전 통제된 인과 효과로 명명하지 않았다.')
add(md_table(['비교','ΔInc pp','ΔLast pp','ΔF pp','factor 장치 차이'],[[q['comparison'],fmt(q['delta_Inc_pp'],True),fmt(q['delta_Last_pp'],True),fmt(q['delta_F_pp'],True),'있음' if q['factor_device_differs'] else '없음'] for q in contrasts]))
add('E1−E0에서는 Multi-Center만 추가해 두 정확도가 낮아졌다. single-center separation 중 E3만 Last-Acc가 증가하고 E2는 두 지표가 낮아지며 E4는 동일하다. multi-center 상태에서 E5−E1은 Inc·Last 모두 증가하지만 E0의 Inc-Acc까지 넘지는 못한다. E6−E1은 Inc가 약간 증가하나 Last가 감소하고 E7−E1은 동일하다. 같은 covariance의 MC 추가 효과는 isotropic(E5−E2)에서는 양수, diagonal(E6−E3)에서는 음수로 일관되지 않는다.')
add('다음 검증 우선순위는 (1) 나머지 seed를 확보해 동일 seed paired Δ의 표본 SD를 계산, (2) full rho·birth tau·저랭크 covariance 및 feature 변환을 별도로 감사, (3) snapshot 코드와 factor 장치를 고정해 재현성 확인, (4) E3의 old/new trade-off와 E6의 마지막 task 역전을 class별 counts로 추적하는 것이다. 이번 작업은 새 학습이나 방법 변경을 포함하지 않는다. 추가 seed의 결과가 나오기 전에는 E3의 1개 정답 증가를 일반적인 성능 향상으로 주장할 수 없다.')
add('## 10. 부록: 원자료, 정의, 출처 및 재생성\n\n`a(t,i)=100×correct(t,i)/N_i`는 task t의 모든 학습 종료 모델을 task i의 test samples에 평가하되 후보는 모든 seen classes다. `Δ_m,s(t,i)=a_m,s(t,i)−a_E0,s(t,i)`이다. `A(t)=100×Σ_i correct(t,i)/Σ_i N_i`, `Inc-Acc=mean_t A(t)`, `Last-Acc=A(T)`, `F_i=max_{u=i..T−1}a(u,i)−a(T,i)`, `Forgetting=mean_{i<T}F_i`를 사용했다. old는 i<t, new는 i=t의 correct/sample 합이고 T1 old는 N/A다. 정확도 단위 %, 차이·forgetting 단위 pp; memory MiB=2²⁰ bytes, GiB=2³⁰ bytes다. 음수 forgetting을 보정하지 않는다.')
add('이 보고서는 seed 1993 고정 snapshot이다. n=1이므로 모든 SD는 N/A이며 그림에 오차 막대를 그리지 않았다. 여러 seed 분석으로 확장할 때에는 공통 완료 seed를 먼저 선정하고, 평균 Δ와 표본 SD(ddof=1)를 **seed별 paired Δ에서 직접** 계산해야 한다. 현재 스크립트는 나중에 발견한 다른 seed를 이 보고서에 자동으로 섞지 않는다.')
add(md_table(['자료','파일'],[
 ['재계산 성능 및 시간·메모리','[metrics_per_seed.csv](tables/metrics_per_seed.csv)'],
 ['각 task A/old/new 및 E0 차이','[task_metrics_recomputed.csv](tables/task_metrics_recomputed.csv)'],
 ['정확도 셀·paired Δ·셀별 SD=N/A','[cells_per_seed_and_sd.csv](tables/cells_per_seed_and_sd.csv)'],
 ['task별 forgetting','[forgetting_per_task.csv](tables/forgetting_per_task.csv)'],
 ['원본 class별 최종 Δ','[class_final_deltas.csv](tables/class_final_deltas.csv)'],
 ['전체 K-means class 표','[kmeans_all_classes.csv](tables/kmeans_all_classes.csv)'],
 ['원본 epoch loss','[raw_epoch_metrics.csv](tables/raw_epoch_metrics.csv)'],
 ['수치 진단','[numerical_diagnostics.csv](tables/numerical_diagnostics.csv)'],
 ['요인별 paired 비교','[controlled_contrasts.csv](tables/controlled_contrasts.csv)'],
 ['검산 목록','[validation_checks.csv](tables/validation_checks.csv)'],
 ['출처·SHA-256·단위·누락','[metadata.json](metadata.json)']]))
add('절대/차이 행렬은 `tables/E{0..7}_absolute_matrix.csv`, `tables/E{0..7}_delta_E0_matrix.csv`이며 미래 셀은 빈 값이다. 원본 CSV는 `tables/raw_*.csv`로 함께 제공한다. metadata의 SHA-256은 보고서 생성 시 읽은 기록의 해시이고 당시 실행 코드의 동일성 증거로 대체하지 않는다. checkpoint는 manifest와 파일 존재/크기를 확인했으며 큰 binary 내용은 읽지 않았다.')
add('출처 구현: `trainer.py`의 task-final 집계·forgetting, `models/base.py`의 all-logit 평가 및 class당 replay, `models/CDS_adapter.py`의 stage loss·APR/cache, `utils/distribution_regions.py`의 K 축소 정책, `RECORDING_SCHEMA_KO.md`의 기록 규약을 현재 파일과 대조했다. 실행 당시 dirty 소스 전문이 저장된 것은 아니므로 현재 코드 해석에는 앞서 언급한 한계가 있다. 해시와 경로는 metadata에 보존한다.')
add('재생성(Python ≥3.9, numpy, matplotlib; pandas/GPU 불필요):\n\n```bash\nconda activate cds\npython '+str(OUT/'build_report.py')+' \\\n  --project-root '+str(ROOT)+' \\\n  --output-dir '+str(OUT)+('\n```\n\n현재 환경에서 사용한 명시적 interpreter는 `${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python`이다. 다른 곳으로 이동한 보고서에서도 그림과 표 링크는 상대경로로 유지된다. 재생성에는 `--project-root`로 원본 기록이 있는 프로젝트를 지정한다.').replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT)))

for p in ['trainer.py','models/base.py','models/CDS_adapter.py','utils/distribution_regions.py','utils/experiment_io.py','RECORDING_SCHEMA_KO.md','scripts/run_e0_e7.py']:
    track(ROOT/p)
track(Path(('${RESEARCH_WORKSPACE}/E0_비교_보고서_작성_프롬프트.md').replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT))))
track(Path(__file__))
report='\n\n'.join(text)+'\n'
for target in re.findall(r'\]\(([^)]+)\)',report):
    if not target.startswith(('http','#')) and target not in ['metadata.json','tables/validation_checks.csv']:
        check('report link '+target,(OUT/target).exists())
write_csv('validation_checks.csv',checks)
(OUT/'report.md').write_text(report,encoding='utf-8')
metadata={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'input_root':str(DATA),
    'report_scope':'fixed seed-1993 snapshot; future seeds are not silently aggregated',
    'used_seeds':[SEED],'n':1,'runs':[str(DATA/m/f'seed_{SEED}') for m in METHODS],
    'dataset':'imageneta','num_tasks':T,'class_order':protocol['class_order_original_ids'],
    'split_hashes':protocol['splits'],'evaluation_stage':'task_final','single_head':True,
    'checkpoint_selection':'final task; no best-test selection; latest file only',
    're_evaluation_performed':False,'training_performed':False,
    'units':{'accuracy':'percent','delta':'pp','forgetting':'pp','intrusion':'ratio 0..1','MiB':2**20,'GiB':2**30},
    'definitions':{'A':'100*sum(correct)/sum(samples)','Inc':'mean(A)','Last':'A(T)',
       'F':'mean_{i<T}(max_{u=i..T-1}a(u,i)-a(T,i)); no clipping',
       'SD':'sample SD ddof=1 if n>=2; N/A for n=1','paired_delta':'same seed and same cell minus E0'},
    'color_range_delta_pp':[-vmax,vmax],'color_range_absolute_percent':[0,100],
    'zero_tolerance_pp':EPS,'small_value_display':'scientific notation when nonzero abs(delta)<0.005',
    'missing_valid_cells':'N/A + hatch','future_cells':'white blank','config_differences':config_diffs,
    'inventory':inventory,'source_identity_limitation':'same commit, dirty working-tree hashes differ and include outputs; code identity unverified',
    'source_states':{m:r['source_state'] for m,r in R.items()},
    'resolved_configs':{m:r['resolved_config'] for m,r in R.items()},
    'protocol_manifest':protocol,
    'validation_tolerances':{'accuracy_pp':EPS,'weighted_loss':1e-6,'total_loss_sum':1e-5},
    'checkpoint_audit':{m:{'manifest':r['checkpoint_manifest'],'file_bytes':r['checkpoint_bytes'],'binary_loaded':False} for m,r in R.items()},
    'sources':sources,'validation_checks_passed':len(checks),'validation_failures':0,
    'software':{'numpy':np.__version__,'matplotlib':matplotlib.__version__},
    'figure_count':len(list((OUT/'figures').glob('*.png')))}
(OUT/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'report':str(OUT/'report.md'),'figures':metadata['figure_count'],'tables':len(list((OUT/'tables').glob('*.csv'))),'checks':len(checks),'delta_range':vmax},ensure_ascii=False))
