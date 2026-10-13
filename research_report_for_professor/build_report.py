"""Research evidence/figure builder. Reads existing results; never trains a model."""
from __future__ import annotations
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[1])))
import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
WS = ROOT.parent
AUDIT = WS / 'structure_prototype_oracle_audit'
PROTOCOLS = ('IA5', 'CAR5', 'IA10', 'CAR10')
ROOTS = ['cdsp_handoff_20260813', 'structure_multicenter_separation', 'structure_e3_followup',
    'structure_similarity_aware_followup', 'structure_similarity_control_experiments',
    'structure_selective_similarity_sweep', 'structure_rp_protoclad_task5', 'structure_stored_apr_rp',
    'structure_prototype_oracle_audit', 'fecam_boundary_followup', 'variation_consistency_pilot',
    'error_trajectory_diagnostic', 'cross_task_ca_reweighting', 'b1_ranking_preserving_correction',
    'type_a_generative_augmentation', 'generation_local_all_transition_audit', '논문',
    '실험완료', '시도는 좋았다', '0_보관소', '0923_code_bundle']
SOURCES = {}
RESULTS = []
EVIDENCE = []

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''): h.update(block)
    return h.hexdigest()

def register(path):
    p = Path(path)
    if not p.is_absolute(): p = WS / p
    if str(p) not in SOURCES:
        SOURCES[str(p)] = {'sha256': digest(p), 'bytes': p.stat().st_size}
    return p

def rows(path):
    with register(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def data(path): return json.loads(register(path).read_text())

def csv_write(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in items for k in r)))
        w.writeheader(); w.writerows(items)

def metric(experiment, protocol, name, value, unit, source, scope='', note=''):
    RESULTS.append(dict(experiment=experiment, protocol=protocol, model_seed=1993, scope=scope,
        metric=name, value=value, unit=unit, source_file=str(Path(source).relative_to(WS)), note=note))

def evidence(experiment, question, datasets, source, result, interpretation, verdict, diagnostic=True):
    EVIDENCE.append(dict(experiment=experiment, research_question=question, datasets=datasets,
        checkpoint_source='vanilla Structure seed 1993; details in source_map',
        metric='see key_results.csv / source', main_result=result, interpretation=interpretation,
        verdict=verdict, diagnostic_only=diagnostic, source_file=source))

def inventory():
    listing = []
    excluded = Counter()
    for name in ROOTS:
        root = WS / name
        if not root.exists(): continue
        for base, dirs, files in os.walk(root):
            skipped = [d for d in dirs if 'source_snapshot' in d or d in (
                '.cache', '.git', '__pycache__', 'node_modules', '.venv_generator', 'generated', 'contact_sheets')]
            for d in skipped: excluded[d] += 1
            dirs[:] = [d for d in dirs if d not in skipped]
            for filename in files:
                p = Path(base) / filename
                if p.suffix.lower() not in ('.md', '.pdf', '.json', '.csv'): continue
                relative = p.relative_to(WS)
                role = 'ARCHIVE_NOT_POOLED' if name in ('0_보관소', '시도는 좋았다', '실험완료', '0923_code_bundle') else (
                    'SMOKE_PREFLIGHT_NOT_POOLED' if any(x in str(relative).lower() for x in ('smoke', '_preflight', '_scratch')) else 'CANONICAL_CANDIDATE')
                listing.append(dict(project=name, path=str(relative), extension=p.suffix,
                    bytes=p.stat().st_size, role=role))
    csv_write(ROOT / 'appendix/file_inventory.csv', listing)
    counts = Counter(x['project'] for x in listing)
    reports = [x for x in listing if x['extension']=='.md' and 'report' in Path(x['path']).name.lower()]
    lines = ['# 전체 연구 파일 inventory', '', '작성일: 2026-10-12 KST. Canonical 연구 폴더와 보관본을 구분했다. Recursive source snapshot, cache, 생성 이미지, weights는 중복 집계하거나 본문 evidence로 사용하지 않는다.', '',
        f'검색한 결과/문서 파일 {len(listing):,}개; Markdown report 후보 {len(reports)}개. 세부 경로는 [file_inventory.csv](file_inventory.csv).', '', '| 프로젝트 | 문서/결과 파일 |', '|---|---:|']
    lines.extend(f'| {k} | {v} |' for k,v in counts.items())
    lines += ['', '## Report 후보', '']
    lines.extend('- ['+x['path']+'](../../'+x['path']+') — '+x['role'] for x in reports)
    lines += ['', '## 범위 / 비포함', '', 'IA5/CAR5/IA10/CAR10의 seed 1993 canonical source를 우선 사용한다. Snapshot 복제, 이전 incomplete report, smoke/preflight는 성능 evidence로 합산하지 않는다. PTM 초기 문서는 논문 reference와 자체 실행값을 분리한다. 원본 파일은 수정하지 않는다.']
    (ROOT / 'appendix/experiment_inventory.md').write_text('\n'.join(lines)+'\n')
    print('Inventory:',len(listing),'files;',len(reports),'report candidates',flush=True)
    return listing


FIGURES = []

def collect():
    """Only read already completed experiments. Assert sample-level identities."""
    import numpy as np
    out = dict(age={}, oracle={}, source={}, classifier={}, probe={}, trajectory={},
               ranking={}, reweight={}, support={}, transitions={}, generation={}, early=[])
    ten = AUDIT / 'ten_task_hypothesis_revalidation'
    for p in PROTOCOLS:
        path = ten / p / 'prototype_tracking_by_age.csv'
        out['age'][p] = rows(path)
        for r in out['age'][p]:
            metric('memory_age', p, 'mean_relative_L2', float(r['tracking_relative_l2_mean']),
                   'ratio', path, 'age='+r['class_age']+'; pooled class-task records')
        path = ten / p / 'oracle_abcd_final_task.csv'
        rr = rows(path); out['oracle'][p] = {r['condition']:float(r['seen_class_accuracy']) for r in rr}
        for r in rr:
            metric('oracle_CA', p, r['condition'], float(r['seen_class_accuracy']), '%', path, 'final seen micro; diagnostic head')
        path = ten / p / 'g256_r256_final_task.csv'
        rr=rows(path); out['source'][p]={r['condition']:float(r['seen_class_accuracy']) for r in rr}
        for r in rr:
            metric('matched_source', p, r['condition'], float(r['seen_class_accuracy']), '%', path,
                   '256/class/epoch; old source changed; current bank fixed')
        root=AUDIT/'scientific_audit'/p/'seed_1993'
        for name,field in [('prototype_oracle_audit.csv','prototype_relative_l2'),
                           ('covariance_oracle_audit.csv','frobenius_relative_error')]:
            path=root/name
            rr=[r for r in rows(path) if r['oracle_split']=='train' and r['is_old']=='True']
            metric('memory_tracking',p,field,float(np.mean([float(r[field]) for r in rr])),
                   'ratio',path,'all old class-task records; train oracle')
        path=root/'apr_oracle_drift_audit.csv'; rr=rows(path)
        metric('APR',p,'improved_records',sum(r['APR_result']=='improved' for r in rr),'count',path,
               'old class-task records; total='+str(len(rr)))
        register(root/'ca_regression_check.json')
    for p in ('IA5','CAR5'):
        root=AUDIT/'classifier_procedure_diagnostic_gpu2'/(p+'_T5')
        path=root/'classifier_probe_accuracy.csv'
        rr=[r for r in rows(path) if r['row_type']=='summary']
        out['classifier'][p]={r['condition']:float(r['seen_class_accuracy']) for r in rr}
        for r in rr:
            metric('classifier_probe',p,r['condition'],float(r['seen_class_accuracy']), '%',path,
                   'same frozen features; validation-selected; diagnostic')
        path=root/'g256_r256_accuracy.csv'
        for r in rows(path):
            if r['row_type']=='summary':
                metric('sampling_fairness',p,r['condition'],float(r['seen_class_accuracy']), '%',path,
                       'G_FIXED/R_FIXED exposure imbalance; do not compare as source-only effect')
    for p in ('IA10','CAR10'):
        root=AUDIT/'coverage_generalization_diagnostic'/p
        path=root/'probe_results.csv'
        rr=[r for r in rows(path) if r['row_type']=='selected']
        out['probe'][p]={r['representation']:r for r in rr}
        for r in rr:
            for field in ('train_accuracy','test_accuracy','class_macro_accuracy','support_margin_micro'):
                metric('pretrained_final',p,r['representation']+'_'+field,float(r[field]),
                       '%' if 'accuracy' in field else 'cosine gap',path,
                       'same split/optimizer/budget; test not used for selection')
        path=root/'paired_support_delta.csv';rr=rows(path)
        for field in ('delta_B_minus_A','delta_C_minus_B'):
            metric('coverage_source',p,field,float(np.mean([float(r[field]) for r in rr])),
                   'cosine margin',path,'equal per-class reference count; pooled 3 Gaussian draws, not model seeds')
        path=root/'support_summary.csv'
        for r in rows(path):
            if r['source']=='A_ACTUAL_CURRENT' and r['draw']=='0' and r['grouping'] in ('overall','classifier_correct'):
                metric('coverage_source',p,'actual_support_margin',float(r['support_margin_mean']),
                       'cosine margin',path,r['grouping']+'; classifier_correct='+r['classifier_correct'])
        root=WS/'error_trajectory_diagnostic'/p
        path=root/'sample_trajectory.csv'; tr=rows(path)
        assert len({r['sample_id'] for r in tr})==len(tr)
        retention=[r for r in tr if int(r['birth_task'])<=9]
        counts=Counter(r['error_type'][0] for r in retention if int(r['correct_T10'])==0)
        assert sum(counts.values())==sum(int(r['correct_T10'])==0 for r in retention)
        b1=sum(r['error_type'].startswith('B') and r['birth_restriction_restored']=='1' for r in retention)
        out['trajectory'][p]=dict(counts=counts,b1=b1,n=len(tr),retention_n=len(retention),
                                  final_correct=sum(int(r['correct_T10']) for r in tr))
        for k in ('A','B','C'):
            metric('trajectory',p,k,counts[k],'count',path,'birth<=T9; task-end checkpoints only')
        metric('trajectory',p,'B1_restored',b1,'count',path,'candidate set = all classes seen at birth')
        metric('trajectory',p,'checkpoint_final_accuracy',100*out['trajectory'][p]['final_correct']/len(tr),
               '%',path,'task_end checkpoint direct inference; not pre-after_task summary')
        path=WS/'b1_ranking_preserving_correction'/p/'metrics.json'; v=data(path)
        out['ranking'][p]=v
        for branch in ('C0','C1','C2'):
            for field in ('repair','harm','net','delta_pp'):
                metric('rank_preserving',p,branch+'_'+field,v['branches'][branch]['subsets']['overall'][field],
                       'pp' if field=='delta_pp' else 'count',path,'T10 whole seen micro')
            metric('rank_preserving',p,branch+'_independent_Gaussian_CE',v['gaussian_CE'][branch]['check_Gaussian_CE'],
                   'CE',path,'independent synthetic bank; NOT held-out image accuracy')
        path=WS/'cross_task_ca_reweighting'/p/'summary.json'; v=data(path); out['reweight'][p]=v
        for field in ('repair_n','harm_n','net_gain_n','delta_accuracy'):
            metric('cross_task_CA',p,field,v['overall'][field]*(100 if field=='delta_accuracy' else 1),
                   'pp' if field=='delta_accuracy' else 'count',path,'internal paired control; S0 parity FAIL')
        root=WS/'generation_local_all_transition_audit'/p
        out['transitions'][p]={}
        for task in ('T9','T10'):
            path=root/(task.lower()+'_transition_summary.csv')
            rr=rows(path)
            chosen=next(r for r in rr if r['cohort']=='T9_all' and r['comparison']=='G-S')
            mat=np.array([[int(chosen['n'+i+j]) for j in 'WCK'] for i in 'WCK'])
            assert mat.sum()==int(chosen['N'])
            assert int(chosen['local_net'])==mat[0,1]+mat[0,2]-mat[1,0]-mat[2,0]
            assert int(chosen['all_net'])==mat[0,2]+mat[1,2]-mat[2,0]-mat[2,1]
            out['transitions'][p][task]=dict(row=chosen,matrix=mat)
            for r in rr:
                if r['comparison']=='G-S':
                    for field in ('N','local_repair','local_harm','local_net','local_delta_pp',
                                  'all_repair','all_harm','all_net','all_delta_pp',
                                  'reference_macro_all_accuracy_pct','target_macro_all_accuracy_pct'):
                        metric('local_all_audit',p,task+'_'+field,float(r[field]),
                               'pp' if 'delta_pp' in field else ('%' if 'accuracy' in field else 'count'),
                               path,r['cohort']+'; S->G')
        # Recompute W/C/K summary from each of the immutable sample rows.
        path=root/'t9_sample_state_transitions.csv'; pairs=rows(path)
        for task in ('T9','T10'):
            mat=np.zeros((3,3),dtype=int)
            for r in pairs: mat['WCK'.index(r['S_'+task+'_state']), 'WCK'.index(r['G_'+task+'_state'])]+=1
            assert np.array_equal(mat,out['transitions'][p][task]['matrix'])
    path=WS/'type_a_generative_augmentation/birth_support_audit.csv'; rr=rows(path)
    for p in ('IA10','CAR10'):
        # ONE prespecified bank seed. It is not a model replicate; no CI.
        chosen=[r for r in rr if r['protocol']==p and int(r['task'])<=9 and r['subsample_seed']=='1993'
                and r['support_valid']=='True']
        class_groups=defaultdict(lambda:defaultdict(list))
        for r in chosen:
            group=r['A_location'] or r['group']
            class_groups[r['y']][group].append(float(r['own_support']))
        result={}
        for kind in ('A-within','A-cross'):
            matching=[g for g in class_groups.values() if kind in g and 'birth_correct' in g]
            result[kind]=dict(n=sum((r['A_location'] or r['group'])==kind for r in chosen),
                matched_classes=len(matching),a=float(np.mean([np.mean(g[kind]) for g in matching])),
                correct=float(np.mean([np.mean(g['birth_correct']) for g in matching])),
                class_deltas=[float(np.mean(g[kind])-np.mean(g['birth_correct'])) for g in matching])
            for field in ('n','matched_classes','a','correct'):
                metric('birth_support',p,kind+'_'+field,result[kind][field],
                       'count' if field in ('n','matched_classes') else 'cosine gap',path,
                       'birth<=T9; common-class macro own support; bank seed 1993 only')
        out['support'][p]=result
        path=WS/'type_a_generative_augmentation/audit'/p/'class_matched_support.csv'
        matched=[r for r in rows(path) if r['subsample_seed']=='1993' and r['A_minus_correct_own_support']]
        metric('birth_support',p,'all_A_class_matched_own_support_delta',
               float(np.mean([float(r['A_minus_correct_own_support']) for r in matched])),
               'cosine similarity',path,'all A; common-class macro; single bank seed 1993')
    path=WS/'type_a_generative_augmentation/metrics.json'; gen=data(path)
    out['generation']=gen
    for p in ('IA10','CAR10'):
        for key,subsets in gen[p]['comparisons'].items():
            if key.endswith(('G-S','GM-G')):
                for subset in ('overall','new','T9'):
                    for field in ('n','repair','harm','net','delta_pp'):
                        value=subsets[subset][field]
                        if value is not None:
                            metric('generation',p,key+'_'+field,value,
                                   'pp' if field=='delta_pp' else 'count',path,subset)
    # Independently aggregate source prediction CSV rather than trusting metrics alone.
    path=WS/'type_a_generative_augmentation/per_sample_predictions.csv'; rr=rows(path)
    groups=defaultdict(list)
    for r in rr: groups[(r['protocol'],r['location'],int(r['evaluated_after_task']),r['branch'])].append(r)
    checks=0
    for (p,loc,t,b),samples in groups.items():
        ids={r['sample_id'] for r in samples}; assert len(ids)==len(samples)
        cond=gen[p]['conditions'][f'{loc}_T{t}_{b}']
        assert len(samples)==cond['n']
        assert sum(r['pred']==r['y'] for r in samples)==cond['correct']; checks+=1
    out['generation_checks']=checks
    for root,filename in [('structure_multicenter_separation','reports/e0_comparison_20260916/tables/metrics_per_seed.csv'),
                          ('structure_e3_followup','reports/e3_followup_comparison_20260917/tables/metrics_per_seed.csv'),
                          ('structure_similarity_aware_followup','outputs/imageneta/metrics_per_seed.csv'),
                          ('structure_similarity_control_experiments','outputs/imageneta/metrics_per_seed.csv')]:
        path=WS/root/filename; rr=rows(path)
        for r in rr:
            out['early'].append(dict(project=root,**r))
            for field in ('Inc_Acc','Last_Acc','Forgetting_pp','final_old_accuracy','final_new_accuracy'):
                if r.get(field): metric(root,'IA5',r.get('method','unknown')+'_'+field,float(r[field]),
                                       'pp' if field=='Forgetting_pp' else '%',path,'single seed; within-study comparator')
            for field in ('Inc-Acc','Last-Acc','Forgetting','Old','New'):
                if r.get(field): metric(root,'IA5',r['method']+'_'+field,float(r[field]),
                                       'pp' if field=='Forgetting' else '%',path,'single seed; raw unrounded export')
    # Completed reduced matrix only; never pool interrupted or held 10-task sweep.
    for p in ('IA5','CAR5'):
        for method in ('Structure','A_P005','B_P020_A100','C_P020_L030','C_P020_L100'):
            path=WS/'structure_selective_similarity_sweep/outputs'/p/method/'seed_1993/summary.json'
            if path.exists():
                v=data(path)
                out['early'].append(dict(project='reduced',protocol=p,method=method,summary=v))
                for field in ('Inc-Acc','Last-Acc','Forgetting'):
                    metric('reduced_selective',p,method+'_'+field,v[field],
                           'pp' if field=='Forgetting' else '%',path,'completed 5-task selected matrix only')
                tpath=path.parent/'task_summary.csv'
                final=next(r for r in reversed(rows(tpath)) if r['eval_stage']=='task_final')
                for field in ('old_accuracy','new_accuracy'):
                    metric('reduced_selective',p,method+'_'+field,float(final[field]),'%',tpath,'final task single-head')
    for project in ('structure_rp_protoclad_task5','structure_stored_apr_rp'):
        for path in (WS/project/'outputs').glob('*/*/seed_1993/summary.json'):
            v=data(path)
            if v.get('status')!='complete': continue
            protocol=path.parents[2].name;method=path.parents[1].name
            for field in ('Inc-Acc','Last-Acc','Forgetting'):
                metric(project,protocol,method+'_'+field,v[field],'pp' if field=='Forgetting' else '%',path,'completed canonical run')
    for p in ('IA10','CAR10'):
        for kind in ('fecam','ceos'):
            path=WS/'fecam_boundary_followup'/kind/p/'metrics.json'; rr=data(path)
            for cond,values in rr.items():
                if isinstance(values,dict):
                    for field in ('seen_class_accuracy','old_accuracy','new_accuracy','q05','q10','negative_fraction'):
                        if field in values: metric('FeCAM_CEOS',p,cond+'_'+field,values[field],
                            '%' if 'accuracy' in field else ('ratio' if field=='negative_fraction' else 'logit margin'),path,
                            'final frozen features; matched comparator per condition')
        path=ten/p/'transition_summary.csv'; rr=rows(path)
        for r in rr:
            for field in ('positive_degradation_mass','improvement_mass'):
                metric('pair_transition',p,field,float(r[field]),'boundary violation count',path,r['transition'])
        path=ten/p/'error_concentration_by_task.csv'; rr=rows(path)
        final=next(r for r in rr if r['task']=='10')
        metric('pair_concentration',p,'top20_share',float(final['top20_error_share']),'ratio',path,'T10 directed error pairs')
    path=WS/'variation_consistency_pilot/cross_protocol/summary.csv';rr=rows(path)
    for r in rr:
        for field in ('t9_new','t10_seen','t10_newly_old_t9','retention_t9'):
            metric('JS_consistency',r['protocol'],r['branch']+'_'+field,float(r[field]),
                   'pp' if field=='retention_t9' else '%',path,'same two-view supervised control; lambda=.1, T=1 unscaled cosine')
    path=AUDIT/'paired_source_prediction_diagnostic/cross_protocol/final_summary.csv'
    for r in rows(path):
        if r['group']=='all':
            for field in ('repair_count','harm_count','net_gain_pp','class_macro_delta_pp'):
                metric('paired_source',r['protocol'],r['comparison']+'_'+field,float(r[field]),
                       'pp' if 'pp' in field else 'count',path,'A actual; B oracle Gaussian; C stored Gaussian; identical current bank')
    # Supplementary already-completed diagnostic numeric tables.
    for path in (WS/'fecam_boundary_followup').rglob('*.csv'):
        if any('snapshot' in s for s in path.parts): continue
        if path.name not in ('accuracy.csv','condition_accuracy.csv','summary.csv','summary_table.csv'): continue
        for r in rows(path):
            for field,value in r.items():
                if ('accuracy' in field or 'margin' in field) and value:
                    try: number=float(value)
                    except ValueError: continue
                    metric('FeCAM_CEOS',r.get('protocol',path.parent.name),r.get('condition','')+'_'+field,
                           number,'as stored',path,'see source units')
    return out


def make_figures(v):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':190})
    colors={'IA':'#28658c','CAR':'#bd7130'}
    figroot=ROOT/'figures';figroot.mkdir(exist_ok=True)
    def save(fig,name,title,sources,definition):
        fig.savefig(figroot/(name+'.png'),bbox_inches='tight',facecolor='white')
        fig.savefig(figroot/(name+'.svg'),bbox_inches='tight',facecolor='white')
        plt.close(fig)
        FIGURES.append(dict(name=name,title=title,sources=sources,definition=definition))
    # 01: graphical research map; boxes encode observations, not causal proof.
    fig,ax=plt.subplots(figsize=(12,7)); ax.set(xlim=(0,12),ylim=(0,7));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#64717b',lw=.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    weak='#eef0f1';observed='#e4eef3';open_color='#f8ebcc'
    for x,txt in [(0.1,'Structure\nSCC + APR + Gaussian CA'),(3.2,'Selective separation?\nNo common winner'),
                  (6.3,'Moment fidelity?\nAge error exists;\noracle gain small'),(9.4,'Source / boundary?\nSignals exist;\ninterventions fail')]:
        box(x,5.55,2.45,1.05,txt,observed if x==.1 else weak)
    for x in (2.55,5.65,8.75):ax.annotate('',xy=(x+.65,6.08),xytext=(x+.08,6.08),arrowprops=dict(arrowstyle='->'))
    box(4.5,3.9,3,1.1,'Error trajectory\nType A dominates\nretained final errors',observed)
    ax.annotate('',xy=(6,5.0),xytext=(10.65,5.5),arrowprops=dict(arrowstyle='->',connectionstyle='angle3'))
    box(.6,2.1,4.8,1.15,'Branch B: later competitor needed (B1)\n81.3% IA / 84.5% CAR of Type B\n2x cross-CA / task offsets: no common gain',weak)
    box(6.5,2.1,4.8,1.15,'Branch A: persistent birth-time errors\nWithin-task and cross-task; lower support\nGeneric generation: T1 gain, late-task failure',observed)
    for x in (3,9):ax.annotate('',xy=(x,3.28),xytext=(6,3.87),arrowprops=dict(arrowstyle='->'))
    box(3,.35,6,1.0,'OPEN: can real-only acquisition improve local AND all-seen,\nthen survive T10? Next: one matched CE + SupCon pilot.',open_color)
    for x in (3,9):ax.annotate('',xy=(6,1.4),xytext=(x,2.05),arrowprops=dict(arrowstyle='->'))
    ax.set_title('Research Decision Flow | observed / weakened / open',fontsize=15,pad=15)
    save(fig,'fig01_research_flow','연구 의사결정 지도',[], 'Evidence synthesis; supported phenomenon is not a proven cause.')
    # 02: age accumulation and causal moment replacement on separate axes.
    fig,axs=plt.subplots(1,2,figsize=(12,4.2))
    for p,rr in v['age'].items():
        rr=[r for r in rr if int(r['class_age'])>0]
        axs[0].plot([int(r['class_age']) for r in rr],[float(r['tracking_relative_l2_mean']) for r in rr],
                    marker='o',ms=4,label=p,color=colors['IA' if p.startswith('IA') else 'CAR'],ls='-' if p.endswith('10') else '--')
    axs[0].set(xlabel='Class age (tasks since birth)',ylabel='Mean relative L2 error',ylim=(0,.3),title='Memory error accumulates')
    axs[0].legend(ncol=2);axs[0].grid(alpha=.2)
    x=np.arange(4);width=.24
    for j,(cond,label,c) in enumerate([('B_ORACLE_MEAN','Mean only','#789fb4'),('C_ORACLE_COVARIANCE','Covariance only','#a9aeb4'),('D_FULL_ORACLE','Full moments','#b87d44')]):
        delta=[v['oracle'][p][cond]-v['oracle'][p]['A_STRUCTURE'] for p in PROTOCOLS]
        bars=axs[1].bar(x+(j-1)*width,delta,width,label=label,color=c)
        for bar,value in zip(bars,delta):axs[1].text(bar.get_x()+bar.get_width()/2,value+(.035 if value>=0 else -.05),f'{value:+.2f}',ha='center',va='bottom' if value>=0 else 'top',fontsize=8)
    axs[1].axhline(0,color='black',lw=.8);axs[1].set(xticks=x,xticklabels=PROTOCOLS,ylabel='Final seen delta vs matched A (pp)',ylim=(-.65,1.42),title='Better moments ≠ common downstream gain');axs[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig,'fig02_age_tracking','기억 오차 누적과 oracle 교체 이득',[str(AUDIT/'ten_task_hypothesis_revalidation'/p/n) for p in PROTOCOLS for n in ('prototype_tracking_by_age.csv','oracle_abcd_final_task.csv')],
         'Left: pooled old class-task mean relative L2 (not a longitudinal independent replicate). Right: final seen micro B/C/D minus A, pp.')
    # 03: fair source vs optimizer/head probes; zero line in both delta panels.
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    delta=[v['source'][p]['R256_FIXED']-v['source'][p]['G256_FIXED'] for p in PROTOCOLS]
    bars=axs[0].bar(PROTOCOLS,delta,color=[colors['IA'],colors['CAR']]*2)
    for b,d in zip(bars,delta):axs[0].text(b.get_x()+b.get_width()/2,d+.04,f'{d:+.3f}',ha='center')
    axs[0].set(ylabel='R256 actual − G256 oracle Gaussian (pp)',ylim=(-.2,1.15),title='Same 256/class exposure; old source only');axs[0].axhline(0,color='black',lw=.8)
    labels=['Long fit − 5ep','Linear − cosine','Bias linear − cosine'];x=np.arange(3)
    for j,p in enumerate(('IA5','CAR5')):
        c=v['classifier'][p]; d=[c['COSINE_CONVERGED']-c['COSINE_5EP'],c['LINEAR_NO_BIAS']-c['COSINE_CONVERGED'],c['LINEAR_WITH_BIAS']-c['COSINE_CONVERGED']]
        bars=axs[1].bar(x+(j-.5)*.34,d,.34,label=p,color=colors['IA' if j==0 else 'CAR'])
        for b,z in zip(bars,d):axs[1].text(b.get_x()+b.get_width()/2,z-.3,f'{z:+.2f}',ha='center',va='top',fontsize=9)
    axs[1].set(xticks=x,xticklabels=labels,ylabel='Matched probe accuracy difference (pp)',ylim=(-20,1),title='Head changes do not give a common rescue');axs[1].axhline(0,color='black',lw=.8);axs[1].legend()
    fig.tight_layout()
    save(fig,'fig03_classifier_source','공정한 source·classifier 교체의 효과',[str(AUDIT/'ten_task_hypothesis_revalidation'/p/'g256_r256_final_task.csv') for p in PROTOCOLS]+[str(AUDIT/'classifier_procedure_diagnostic_gpu2'/(p+'_T5')/'classifier_probe_accuracy.csv') for p in ('IA5','CAR5')],
         'All changes are within their own matched experiment, not minus an unrelated baseline. Actual old features are diagnostic-only.')
    # 04: show absolute train and test, so the generalization claim is visible.
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,p in zip(axs,('IA10','CAR10')):
        for rname,ls in [('PRETRAINED','--'),('STRUCTURE_FINAL','-')]:
            r=v['probe'][p][rname]
            y=[float(r['train_accuracy']),float(r['test_accuracy'])]
            ax.plot([0,1],y,marker='o',label=rname,ls=ls,lw=2,color='#9b9fa4' if rname=='PRETRAINED' else colors['IA' if p.startswith('IA') else 'CAR'])
            for xx,yy in enumerate(y):ax.text(xx,yy+1.5,f'{yy:.2f}%',ha='center',fontsize=9)
        ax.set(xticks=[0,1],xticklabels=['Historical train','Official test'],ylim=(35,104),ylabel='Cosine-probe accuracy (%)',title=p);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig,'fig04_representation_probe','Pretrained 대 final 표현의 일반화',[str(AUDIT/'coverage_generalization_diagnostic'/p/'probe_results.csv') for p in ('IA10','CAR10')],
         'Selected cosine probe: identical split, optimizer and budget; historical train is not representation-unseen.')
    # 05: final-error decomposition + B1 conditional recovery.
    fig,axs=plt.subplots(1,2,figsize=(11,4.2));x=np.arange(2);bottom=np.zeros(2)
    for kind,color in [('A','#356b8b'),('B','#bb8956'),('C','#969ba0')]:
        numbers=[v['trajectory'][p]['counts'][kind] for p in ('IA10','CAR10')]
        shares=np.array([100*n/sum(v['trajectory'][p]['counts'].values()) for p,n in zip(('IA10','CAR10'),numbers)])
        axs[0].bar(x,shares,bottom=bottom,label=kind,color=color)
        for xx,h,b,n in zip(x,shares,bottom,numbers):axs[0].text(xx,b+h/2,f'{kind}: {n} ({h:.1f}%)',ha='center',va='center',color='white' if kind=='A' else 'black',fontsize=10 if kind!='C' else 8)
        bottom+=shares
    axs[0].set(xticks=x,xticklabels=['IA10','CAR10'],ylim=(0,104),ylabel='Share of final errors (%)',title='Birth ≤ T9: Type A remains dominant');axs[0].legend(ncol=3,loc='lower right')
    restored=[100*v['trajectory'][p]['b1']/v['trajectory'][p]['counts']['B'] for p in ('IA10','CAR10')]
    axs[1].bar(x,restored,color='#bb8956')
    for xx,h,p in zip(x,restored,('IA10','CAR10')):axs[1].text(xx,h+2,f"{v['trajectory'][p]['b1']}/{v['trajectory'][p]['counts']['B']}\n{h:.2f}%",ha='center')
    axs[1].set(xticks=x,xticklabels=['IA10','CAR10'],ylim=(0,105),ylabel='B1 / Type B (%)',title='Birth candidate restriction (diagnostic only)')
    fig.tight_layout()
    save(fig,'fig05_error_trajectory','최종 오류 구성과 Type B 후보 제한',[str(WS/'error_trajectory_diagnostic'/p/'sample_trajectory.csv') for p in ('IA10','CAR10')],
         'Retained classes birth<=T9. A always wrong at task-end; B correct at birth/wrong finally; C remaining recovered-then-wrong. B1 final head with all birth-seen classes.')
    # 06: don't put CE and percentage points on the same axis.
    fig,axs=plt.subplots(1,2,figsize=(11,4));x=np.arange(2)
    for j,b in enumerate(('C0','C2')):
        ce=[v['ranking'][p]['gaussian_CE'][b]['check_Gaussian_CE'] for p in ('IA10','CAR10')]
        bars=axs[0].bar(x+(j-.5)*.34,ce,.34,label=b,color='#a0a7ac' if b=='C0' else '#427897')
        for bar,d in zip(bars,ce):axs[0].text(bar.get_x()+bar.get_width()/2,d+.01,f'{d:.3f}',ha='center')
    axs[0].set(xticks=x,xticklabels=['IA10','CAR10'],ylabel='Independent Gaussian check CE',ylim=(0,.72),title='Synthetic objective improves');axs[0].legend()
    deltas=[v['ranking'][p]['branches']['C2']['subsets']['overall']['delta_pp'] for p in ('IA10','CAR10')]
    bars=axs[1].bar(x,deltas,color=[colors['IA'],colors['CAR']]);axs[1].axhline(0,color='black',lw=.8)
    for bar,d,p in zip(bars,deltas,('IA10','CAR10')):
        r=v['ranking'][p]['branches']['C2']['subsets']['overall'];axs[1].text(bar.get_x()+bar.get_width()/2,d-.04,f"{d:+.3f} pp\nrepair {r['repair']} / harm {r['harm']}",ha='center',va='top')
    axs[1].set(xticks=x,xticklabels=['IA10','CAR10'],ylim=(-.86,.08),ylabel='Real-test C2 − C0 (pp)',title='But real-test accuracy declines')
    fig.tight_layout()
    save(fig,'fig06_ranking_transfer','Task 보정의 synthetic 개선·real-test 악화',[str(WS/'b1_ranking_preserving_correction'/p/'metrics.json') for p in ('IA10','CAR10')],
         'C2 centered conditional task offset; CE and pp use separate axes. No seed uncertainty bars.')
    # 07: A location and support, matched within class rather than pooled class imbalance.
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(2)
    for j,kind in enumerate(('A-within','A-cross')):
        ns=[v['support'][p][kind]['n'] for p in ('IA10','CAR10')]
        bars=axs[0].bar(x+(j-.5)*.34,ns,.34,label=kind,color='#3c708d' if j==0 else '#b88c60')
        for b,n in zip(bars,ns):axs[0].text(b.get_x()+b.get_width()/2,n+1,str(n),ha='center')
    axs[0].set(xticks=x,xticklabels=['IA10','CAR10'],ylim=(0,570),ylabel='Fixed Type A, birth ≤ T9 (count)',title='Birth-time failure is not one type');axs[0].legend()
    labels=[];values=[];cols=[];notes=[]
    for p in ('IA10','CAR10'):
        for kind in ('A-within','A-cross'):
            r=v['support'][p][kind];labels.append(p+'\n'+kind);values.append(r['a']-r['correct']);cols.append(colors['IA' if p.startswith('IA') else 'CAR']);notes.append(r['matched_classes'])
    for pos,(p,kind) in enumerate((p,k) for p in ('IA10','CAR10') for k in ('A-within','A-cross')):
        dd=v['support'][p][kind]['class_deltas']
        # Deterministic spread encodes individual classes, not random seed error bars.
        jitter=np.linspace(-.13,.13,len(dd))
        axs[1].scatter(pos+jitter,dd,s=13,alpha=.38,color=cols[pos])
        axs[1].plot([pos-.19,pos+.19],[values[pos],values[pos]],color='black',lw=2)
        axs[1].text(pos,-.49,f'{values[pos]:+.3f}\n{notes[pos]} shared classes',ha='center',fontsize=8)
    axs[1].axhline(0,color='black',lw=.8)
    axs[1].set(xticks=np.arange(4),xticklabels=labels,ylim=(-.52,.18),ylabel='Class-matched own-support: A − birth-correct',title='Dots = classes; black line = class-macro mean')
    fig.tight_layout()
    save(fig,'fig07_birth_support','Type A 분해와 class-matched support',[str(WS/'type_a_generative_augmentation/birth_support_audit.csv')],
         'Birth<=T9; one bank seed 1993; A-within/A-cross own-support versus birth-correct in shared classes. Dots are classes, not replicates. Association, not causal evidence.')
    # 08: compact stages plus raw paired count repair/harm.
    fig,axs=plt.subplots(1,2,figsize=(12,4.3));x=np.arange(4)
    for p in ('IA10','CAR10'):
        c=v['generation'][p]['comparisons'];delta=[c['T1_T1_G-S']['new']['delta_pp'],c['T9_T10_T9_G-S']['new']['delta_pp'],c['T9_T10_T10_G-S']['T9']['delta_pp'],c['T9_T10_T10_G-S']['overall']['delta_pp']]
        axs[0].plot(x,delta,marker='o',label=p,color=colors['IA' if p.startswith('IA') else 'CAR'])
        for xx,d in zip(x,delta):axs[0].annotate(f'{d:+.3f}',(xx,d),xytext=(0,9 if p=='IA10' else -16),textcoords='offset points',ha='center',fontsize=9)
    axs[0].axhline(0,color='black',lw=.8);axs[0].set(xticks=x,xticklabels=['T1 new','T9 new','T10: T9 cohort','T10 seen'],ylim=(-1.1,1.6),ylabel='G − strong augmentation S (pp)',title='Small T1 signal does not transfer');axs[0].legend()
    x=np.arange(2)
    for j,kind in enumerate(('repair','harm')):
        ns=[v['generation'][p]['comparisons']['T9_T10_T10_G-S']['overall'][kind] for p in ('IA10','CAR10')]
        bars=axs[1].bar(x+(j-.5)*.34,ns,.34,label=kind,color='#4e8395' if j==0 else '#b67f64')
        for b,n in zip(bars,ns):axs[1].text(b.get_x()+b.get_width()/2,n+1,str(n),ha='center')
    axs[1].set(xticks=x,xticklabels=['IA10 (N=1519)','CAR10 (N=8041)'],ylim=(0,61),ylabel='T10 whole-seen paired samples (count)',title='Harm outweighs repair');axs[1].legend()
    fig.tight_layout()
    save(fig,'fig08_generation','Generation의 단계별 효과와 repair/harm',[str(WS/'type_a_generative_augmentation/metrics.json'),str(WS/'type_a_generative_augmentation/per_sample_predictions.csv')],
         'G-S; T1 experiment and T8->T9->T10 suffix are separate branches, not one full augmented trajectory.')
    # 09: proportions share the same colour scale; raw N always annotated.
    fig,axs=plt.subplots(2,2,figsize=(9,7.3))
    for row,p in enumerate(('IA10','CAR10')):
        for col,t in enumerate(('T9','T10')):
            ax=axs[row,col];r=v['transitions'][p][t];mat=r['matrix'];n=mat.sum()
            ax.imshow(mat/n*100,cmap='Blues',vmin=0,vmax=85)
            for i in range(3):
                for j in range(3):ax.text(j,i,str(mat[i,j]),ha='center',va='center',color='white' if mat[i,j]/n>.5 else 'black',fontsize=13)
            a=r['row'];ax.set(xticks=range(3),xticklabels=list('WCK'),yticks=range(3),yticklabels=list('WCK'),xlabel='G state',ylabel='S state',title=f"{p} {t} | N={n}\nLocal net {int(a['local_net']):+d}; all net {int(a['all_net']):+d}")
    fig.suptitle('T9 cohort S → G | cells = raw count; colour = cohort %',fontsize=13);fig.tight_layout()
    save(fig,'fig09_local_all_transition','최신 T9 cohort W/C/K 전이',[str(WS/'generation_local_all_transition_audit'/p/n) for p in ('IA10','CAR10') for n in ('t9_sample_state_transitions.csv','t9_transition_summary.csv','t10_transition_summary.csv')],
         'Fixed T9 cohort. W local wrong; C local correct/all wrong; K all correct. Shared 0-85% colour scale, cells raw counts.')


def export_evidence(v):
    entries=[
      ('E0-E7 / multi-center','분포/거리 separation을 더하면 나은가?','IA5','structure_multicenter_separation/reports/e0_comparison_20260916/tables/metrics_per_seed.csv','E3 Last +.066 pp only; E4/E7 separation inactive','full covariance null is not an active intervention failure','WEAKENED / IMPLEMENTATION QUALIFIED',False),
      ('Selective separation','위험 관계를 더 분리하면 공통 이득인가?','IA5/CAR5','structure_selective_similarity_sweep/outputs/*/*/seed_1993/summary.json','ΔInc: all selected |delta|<0.08 pp; no common winner','operating point 이동; IA A/B New -1.476 pp','WEAKENED AS MAIN BOTTLENECK',False),
      ('AM-Drift / E3 / PIC','관계 loss가 stability/plasticity를 동시 개선하는가?','IA5','structure_e3_followup/reports/e3_followup_comparison_20260917/tables/metrics_per_seed.csv','AM-Drift Inc +0.17, Last +0.26, New -1.11 pp','one-dataset trade-off','DATASET-SPECIFIC',False),
      ('Similarity gates / controls','High-sim pair를 풀면 나은가?','IA5','structure_similarity_control_experiments/reports/additional_control/tables/main_performance.csv','High-Off worse than Low-Off; S2 route ~99.91% relative','gate degeneracy / selected location not universal solution','FAILED INTERVENTION',False),
      ('RP / StoredAPR','nonlinear mean definition correction helps?','IA5/CAR5','structure_stored_apr_rp/reports/stored_apr_rp_final_20260923/report.md','IA StoredAPR vs old RP Last -0.527; CAR old RP absent','definition discrepancy real; accurate statistic not guaranteed gain','DATASET-SPECIFIC',False),
      ('Memory / APR','Old moments 추적이 붕괴하는가?','IA5/CAR5/IA10/CAR10','structure_prototype_oracle_audit/scientific_audit','cosine ~.99; APR improves most records; error grows with age','accumulation observed, not prototype collapse','SUPPORTED AS PHENOMENON',True),
      ('Oracle moments','exact old moments cure downstream errors?','IA5/CAR5/IA10/CAR10','structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/*/oracle_abcd_final_task.csv','covariance deltas -0.066,+0.025,+0.132,+0.075 pp','exact moments not common dominant bottleneck','WEAKENED AS MAIN BOTTLENECK',True),
      ('Classifier exposure','old actual source intrinsically poor?','IA5/CAR5','structure_prototype_oracle_audit/classifier_procedure_diagnostic_gpu2','restoring old/class exposure changes +3.818/+5.721 pp','unmatched first diagnostic must not support source hypothesis','SUPERSEDED UNFAIR CONTRAST',True),
      ('G256/R256','actual vs Gaussian matched downstream?','IA5/CAR5/IA10/CAR10','structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/*/g256_r256_final_task.csv','R-G -0.066,+0.547,+0.066,+0.895 pp','larger CAR gain; not universal cure','DATASET-SPECIFIC',True),
      ('Classifier family / length','long fitting or linear head improves?','IA5/CAR5','structure_prototype_oracle_audit/classifier_procedure_diagnostic_gpu2/*/classifier_probe_accuracy.csv','long selected fit -0.922/-0.187; linear much worse','do not infer cosine-family universal bottleneck','FAILED INTERVENTION',True),
      ('Coverage / representation','final representation generalization collapse?','IA10/CAR10','structure_prototype_oracle_audit/coverage_generalization_diagnostic/*/probe_results.csv','final-pretrained test +11.390/+24.225 pp','matched diagnostic contradicts broad collapse claim','WEAKENED AS MAIN BOTTLENECK',True),
      ('Pair transition / risk','same-pair broad deterioration?','IA10/CAR10','structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/*/transition_summary.csv','boundary improvement mass exceeds deterioration in all nine transitions','some hard pairs exist; broad degradation claim weak','WEAKENED AS MAIN BOTTLENECK',True),
      ('FeCAM','covariance-sensitive readout rescues?','IA10/CAR10','fecam_boundary_followup/final_report.md','stored -4.81/-4.37 pp; oracle still below Structure','ported raw-space stats not official normalized-moment reproduction','FAILED INTERVENTION',True),
      ('Boundary / CEOS','harder Gaussian solves boundary gap?','IA10/CAR10','fecam_boundary_followup/final_report.md','CEOS matched seen -0.658/-0.473; new -4.032/-5.777 pp','hardness changes but transfer absent','FAILED INTERVENTION',False),
      ('Variation JS','two-view consistency helps current task?','IA10/CAR10','variation_consistency_pilot/final_report.md','T9 New 0/-0.264; T10 seen both zero pp','null at unscaled cosine T1 lambda.1 only','FAILED INTERVENTION',False),
      ('Error trajectory','birth failure or retention dominant?','IA10/CAR10','error_trajectory_diagnostic/*/sample_trajectory.csv','retained A shares 66.061/58.592%','prioritize acquisition without discarding retention','SUPPORTED AS PHENOMENON',True),
      ('B1 restriction','later candidates needed for Type B?','IA10/CAR10','error_trajectory_diagnostic/*/typeB_candidate_restriction.csv','126/155; 473/560 restored','test-derived class birth knowledge, not deployable correction','SUPPORTED AS PHENOMENON',True),
      ('Cross-task CA 2x','stronger cross-task competition loss?','IA10/CAR10','cross_task_ca_reweighting/*/summary.json','repair/harm 3/6 and 56/50; S0 parity FAIL','internal paired only; universal claim invalid','FAILED / QUALIFIED INTERVENTION',False),
      ('Rank-preserving correction','task shifts without local ranking damage?','IA10/CAR10','b1_ranking_preserving_correction/*/metrics.json','Gaussian CE falls; real-test -.527/-.460 pp','no within-task ranking drift, no successful transfer','FAILED INTERVENTION',False),
      ('Birth support','A-within vs A-cross and support?','IA10/CAR10','type_a_generative_augmentation/birth_support_audit.csv','birth<=T9 within/cross 169/158 IA;411/496 CAR; class-matched support lower','associative, test-defined subset, not acquisition causal proof','SUPPORTED AS PHENOMENON',True),
      ('Generation','fixed source beats strong conventional augmentation?','IA10/CAR10','type_a_generative_augmentation/metrics.json','T1 positive; late and T10 neutral/negative','only fixed generic recipe stopped; label fidelity unverified','STOP CURRENT RECIPE',False),
      ('Mixed moments GM','generated memory source helps at equal representation?','IA10/CAR10','type_a_generative_augmentation/statistics_comparison.json','T9 seen GM-G -.287/-.110 pp','statistics-only at fork; subsequent representation may diverge','FAILED INTERVENTION',False),
      ('Local/all WCK','hidden local improvement offset by competition?','IA10/CAR10','generation_local_all_transition_audit/*/t9_sample_state_transitions.csv','IA T9 local net+1 all0; CAR local-5 all-4; T10 local0/-8','no dataset-common hidden acquisition improvement','COMPLETED; STOP RECIPE',True),
    ]
    for entry in entries: evidence(*entry)
    # Register exact code provenance; these files supply equations, not results.
    codepaths=['structure_multicenter_separation/models/CDS_adapter.py','structure_multicenter_separation/models/base.py',
      'structure_multicenter_separation/network/classifier.py','structure_multicenter_separation/utils/inc_net.py',
      'structure_multicenter_separation/utils/loss.py','structure_multicenter_separation/utils/distill_loss.py',
      'structure_multicenter_separation/utils/adjust_distribution.py','structure_e3_followup/followup/losses.py',
      'structure_e3_followup/followup/adaptive_margin.py','structure_similarity_aware_followup/similarity/losses.py',
      'structure_selective_similarity_sweep/selective/losses.py','structure_rp_protoclad_task5/rp/losses.py',
      'structure_rp_protoclad_task5/rp/projection.py','structure_stored_apr_rp/stored_apr_rp/prototypes.py',
      'cross_task_ca_reweighting/losses.py','b1_ranking_preserving_correction/core.py',
      'type_a_generative_augmentation/learner.py','type_a_generative_augmentation/moments.py',
      'type_a_generative_augmentation/sanity_policy.json','type_a_generative_augmentation/exploratory_config.json',
      'generation_local_all_transition_audit/run.py','variation_consistency_pilot/pilot_learner.py',
      'fecam_boundary_followup/fecam/run.py','fecam_boundary_followup/ceos/run.py',
      'structure_prototype_oracle_audit/pair_boundary_risk/pipeline.py']
    for path in codepaths: register(WS/path)
    for path in ('structure_multicenter_separation/exps_jsons/imageneta_5task/E0.json',
                 'structure_selective_similarity_sweep/outputs/CAR5/Structure/seed_1993/run_metadata.json',
                 'structure_similarity_control_experiments/control/losses.py',
                 'structure_multicenter_separation/utils/distribution_regions.py',
                 'structure_e3_followup/followup/ceos.py',
                 'type_a_generative_augmentation/config.json',
                 'type_a_generative_augmentation/automatic_sanity_summary.json',
                 'type_a_generative_augmentation/run_manifest.json',
                 'generation_local_all_transition_audit/run_manifest.json'):
        register(WS/path)
    for path in (WS/'structure_stored_apr_rp/reports/stored_apr_rp_final_20260923/report.md',
                 WS/'error_trajectory_diagnostic/final_report.md',
                 WS/'type_a_generative_augmentation/reports/type_a_generative_augmentation_20261012/report.md',
                 WS/'generation_local_all_transition_audit/final_report.md',
                 WS/'structure_prototype_oracle_audit/scientific_audit/structure_oracle_scientific_final_report_20260928.md',
                 WS/'structure_prototype_oracle_audit/classifier_procedure_diagnostic_gpu2/classifier_procedure_diagnostic_final_report_20260929.md'):
        register(path)
    paper=WS/'논문/0_Exemplar-FreeClassIncrementalLearningviaPreservingClass-DiscriminativeStructure.pdf';register(paper)
    for r in EVIDENCE:
        name=r['experiment']
        if name in ('E0-E7 / multi-center','Selective separation','AM-Drift / E3 / PIC','Similarity gates / controls','RP / StoredAPR'):
            r['checkpoint_source']='Own completed method run; matched protocol/class order; seed 1993; no diagnostic refit in this report'
        elif name in ('Variation JS','Generation','Mixed moments GM'):
            r['checkpoint_source']='Vanilla Structure T8 -> matched T9/T10 suffix; generation T1 is a separate pretrained-init experiment; seed 1993'
        elif name in ('Error trajectory','B1 restriction','Birth support'):
            r['checkpoint_source']='Vanilla Structure T1-T10 task-end checkpoints / fixed per-sample trajectory; seed 1993'
        elif name=='Local/all WCK':
            r['checkpoint_source']='Already completed S/G generation T9/T10 paired sample/logits; vanilla subtypes fixed; no new inference'
        else:
            r['checkpoint_source']='Vanilla Structure frozen final checkpoint/pre-CA sidecar and existing feature cache; seed 1993; internal matched control'
    csv_write(ROOT/'key_results.csv',RESULTS); csv_write(ROOT/'evidence_map.csv',EVIDENCE)
    csv_write(ROOT/'appendix/early_family_numeric.csv',v['early'])
    lines=['# 주요 수치 전체표', '', '자동 집계: `build_report.py --stage all`. Accuracy %, delta pp, 원문 ratio/count/CE는 구분한다. 모두 model seed 1993이며 bank/draw seed를 model replicate로 간주하지 않는다.', '']
    by_exp=defaultdict(list)
    for r in RESULTS:by_exp[r['experiment']].append(r)
    for exp,rr in by_exp.items():
        lines += ['## '+exp,'','| Protocol | Metric | Value | Unit | Scope |','|---|---|---:|---|---|']
        for r in rr:
            val=f"{r['value']:.6f}" if isinstance(r['value'],float) else str(r['value'])
            lines.append(f"| {r['protocol']} | {r['metric']} | {val} | {r['unit']} | {r['scope']} |")
        lines+=['','원본: '+', '.join('`'+s+'`' for s in sorted({r['source_file'] for r in rr})), '']
    (ROOT/'appendix/full_numeric_tables.md').write_text('\n'.join(lines)+'\n')
    lines=['# Figure index','','본문은 9개 figure. PNG는 MD/PDF, SVG는 편집용. 모두 raw CSV/JSON 재집계; 기존 표 이미지 캡처 없음.','']
    for f in FIGURES:
        lines += [f"## {f['name']} — {f['title']}",'',f"![{f['title']}](figures/{f['name']}.png)", '',f['definition'],'']
    (ROOT/'figure_index.md').write_text('\n'.join(lines)+'\n')
    lines=['# Claim / Figure / Table source map','',('재현 명령: `${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python build_report.py --stage all` (CPU analysis only; no training/import of learners).').replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT)),
      '', '## 본문 claim / table routing','',
      '- Baseline / Table 1: `ten_task_hypothesis_revalidation/*/oracle_abcd_final_task.csv`, condition CHECKPOINT_MAIN. Class/task partition: 기존 run config / generation config.',
      '- Branch B Table 2: cross_task_ca_reweighting/*/summary.json and b1_ranking_preserving_correction/*/metrics.json. Each comparator is internal.',
      '- Branch A Table 3: birth_support_audit.csv, generation metrics.json, local/all transition summary CSV.',
      '- Verdict Table 4: evidence_map.csv '+str(len(EVIDENCE))+' rows; section-level evidence boundaries are explicit.',
      '- baseline / SCC / APR / CosFace / CA equations: exact source files listed below (not an assumed conceptual replacement).',
      '', '## Figures / query definitions','']
    for f in FIGURES:
        lines += ['### '+f['name'],'','- Script/query: `build_report.py::collect / make_figures`','- Definition: '+f['definition']]
        lines += ['- Source: `'+str(Path(s).relative_to(WS))+'`' for s in f['sources']]
        lines+=['']
    lines += ['## Integrity / numeric discrepancies','','- IA10 scientific CHECKPOINT_MAIN: 982/1519=64.647795%; task-end trajectory: 983/1519=64.713627%. Different evaluation timing (before/after `after_task`); do not silently equate. Exact 1-sample mechanism not proven here.',
      '- Early unmatched G_FIXED/R_FIXED are superseded by G256/R256 for source-only inference; exposure restoration alone gives +3.818302/+5.720682 pp.',
      '- Cross-task CA S0 weight parity failed (.011874 IA / .006949 CAR vs .001 tolerance). Internal paired results retained with caveat; no vanilla exact-reproduction claim.',
      '- Coverage report marks Gaussianization NOT SUPPORTED under its decision rule although support gaps are negative; distinguish observed geometry loss from an unproven dominant downstream bottleneck.',
      '- RP CAR5 original-RP comparator absent in final report: no fabricated direct CAR old-RP ablation.',
      '- Original generation human HOLD is superseded only for exploratory auto-QC mode; label_fidelity_verified remains false.',
      '', '## Read-only source fingerprints','', '| Source | Bytes | SHA-256 |','|---|---:|---|']
    for s,m in sorted(SOURCES.items()): lines.append(f"| `{Path(s).relative_to(WS)}` | {m['bytes']} | `{m['sha256']}` |")
    (ROOT/'appendix/source_map.md').write_text('\n'.join(lines)+'\n')
    unchanged=all(digest(Path(s))==m['sha256'] for s,m in SOURCES.items())
    assert unchanged,'Source changed while building report; stop.'
    validation=dict(status='PASS',source_files=len(SOURCES),source_unchanged=unchanged,
      generation_conditions_reaggregated=v['generation_checks'],WCK_sample_cohorts_reaggregated=4,
      figures=len(FIGURES),evidence_rows=len(EVIDENCE),numeric_rows=len(RESULTS),
      model_training_executed=False,model_inference_executed=False,
      created_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat())
    (ROOT/'appendix/build_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(validation,ensure_ascii=False),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['inventory','all'],default='inventory')
    args=parser.parse_args()
    inventory()
    if args.stage=='all':
        values=collect(); make_figures(values); export_evidence(values)
