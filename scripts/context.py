#!/usr/bin/env python3
"""Current-state and update-frequency research on explicitly viewed intervals."""
import argparse
import hashlib
import json
import logging
import shutil

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

import online_repaired as source
from stealth.conditional_blend import feature_state,prediction_state,predict

c=source.c;b=source.b;REPO=source.REPO


def load():
    source.load(True)  # this source's confirmation has already been consumed; no fresh-holdout claim
    root=c.RUN;parent=json.loads((root/'plan.json').read_text())
    c.CFG=yaml.safe_load((REPO/'configs/context.yaml').read_text());c.RUN=REPO/'artifacts'/c.CFG['name'];c.RUN.mkdir(parents=True,exist_ok=True)
    plan=dict(protocol='observable-state-daily-v1',config=c.CFG,data_sha256=parent['data_sha256'],
              viewed_intervals=[[4000,5299],[5340,5924]],source_plan_sha256=c.r.sha(root/'plan.json'),
              source_trace_sha256=c.r.sha(root/'adaptive_score_confirmation_trace.json'),
              native_markers={str(p.relative_to(root)):c.r.sha(p) for p in root.glob('*/*/done.json')},
              code_sha256={f:c.r.sha(REPO/f) for f in ['scripts/context.py','stealth/conditional_blend.py','docs/CONTEXT-15.md']})
    c.PLAN=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();plan['plan_sha256']=c.PLAN
    path=c.RUN/'plan.json'
    if path.exists() and json.loads(path.read_text())['plan_sha256']!=c.PLAN:raise RuntimeError('context source/config changed; use a new run ID')
    if not path.exists():c.r.write_json(path,plan)
    if c.FRAME.era.max()>=5925:raise RuntimeError('protected eras reached context reader')
    return root,parent


def copy_source(root,parent):
    for key,value in parent['config'].items():
        if key not in ('name','comparisons') and c.CFG[key]!=value:raise RuntimeError('native dependency or existing policy specification changed')
    for marker in root.glob('*/*/done.json'):
        doc=json.loads(marker.read_text())
        if doc['plan_sha256']!=parent['plan_sha256']:raise RuntimeError('context native source plan')
        for f,digest in doc['files'].items():
            if c.r.sha(marker.parent/f)!=digest:raise RuntimeError('context native source bytes')
        dest=c.RUN/marker.parent.parent.name/marker.parent.name;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'done.json').exists():c.checked_json(dest/'done.json');continue
        for f in doc['files']:shutil.copyfile(marker.parent/f,dest/f)
        c.r.write_json(dest/'done.json',{**doc,'plan_sha256':c.PLAN,'seconds':0.,'copied_from':'online14r','source_done_sha256':c.r.sha(marker)})


def seed_prediction(name,seed,phase):
    parts=[]
    for block in c.blocks(c.windows(phase),c.CFG['cadence'],c.CFG['gap']):
        directory=c.RUN/block['name']/name;c.checked_json(directory/'done.json')
        parts.append(pd.read_parquet(directory/f'prediction{seed}.parquet'))
    return pd.concat(parts)


def inputs():
    bases=[c.predictions('baseline'),c.predictions('baseline',True)]
    if len(bases[0].index.intersection(bases[1].index)):raise RuntimeError('overlapping OOS periods')
    base=pd.concat(bases);tables=[];statics=[];seeds={n:[[],[]] for n in c.CFG['gpu_recipes']}
    for phase,bb in zip((False,True),bases):
        tables.append(np.column_stack([c.rank_by_era(b.prediction(n,bb,phase),bb.era) for n in c.CFG['online']['components']]))
        statics.append(c.predictions('baseline',phase,static=True).reindex(bb.index).prediction.to_numpy())
        for name in seeds:
            for j,seed in enumerate((17,29)):
                ss=seed_prediction(name,seed,phase).reindex(bb.index)
                if not np.array_equal(ss.era,bb.era):raise RuntimeError('seed-state era mismatch')
                seeds[name][j].append(c.rank_by_era(ss.prediction,bb.era))
    P=np.concatenate(tables);static=np.concatenate(statics);frame=c.FRAME.loc[base.index]
    X=frame[[x for x in frame if x.startswith('feature_')]].to_numpy(float)
    core=[c.core_mean(X),c.core_mean(X[:,::2]),c.core_mean(X[:,1::2])]
    seed_arrays=[(np.concatenate(v[0]),np.concatenate(v[1])) for v in seeds.values()]
    era=base.era.to_numpy();unique=np.unique(era);prediction_states=[];feature_states=[]
    with threadpool_limits(limits=1):
        for e in unique:
            ids=np.flatnonzero(era==e)
            prediction_states.append(prediction_state(P[ids],[(a[ids],bb[ids]) for a,bb in seed_arrays]))
            feature_states.append(feature_state(X[ids]))
    return base,frame,P,static,core,np.asarray(prediction_states),np.asarray(feature_states)


def evaluate(prediction,data,which):
    base,frame,P,static,cores,_,_=data
    mask=base.era.between(4000,5299).to_numpy() if which=='discovery' else base.era.between(5340,5924).to_numpy()
    scenarios={};primary=baseline=None
    for name,ref in [('refreshed',P[:,0]),('static',static)]:
        for i,core in enumerate(cores):
            a=c.composite(prediction[mask],frame.target_everest.to_numpy()[mask],base.era.to_numpy()[mask],ref[mask],core[mask],bound=c.CFG['score_bound'])
            bb=c.composite(P[mask,0],frame.target_everest.to_numpy()[mask],base.era.to_numpy()[mask],ref[mask],core[mask],bound=c.CFG['score_bound'])
            scenarios[f'{name}/core_{i}']=float((a.score-bb.score).mean())
            if name=='refreshed' and i==0:primary,baseline=a,bb
    delta=primary-baseline
    windows=c.CFG['discovery'] if which=='discovery' else [dict(name='first_half',test_start=5340,test_end=5632),dict(name='second_half',test_start=5633,test_end=5924)]
    return dict(metrics=primary.mean().to_dict(),baseline=baseline.mean().to_dict(),
                paired={k:c.cv.paired_block_bootstrap(delta[k],**c.CFG['bootstrap']).as_dict() for k in ('fit','score')},
                windows={w['name']:{k:float(delta.loc[w['test_start']:w['test_end'],k].mean()) for k in ('fit','score')} for w in windows},
                scenario_score_gains=scenarios,
                block_sensitivity={str(block):c.cv.paired_block_bootstrap(delta.score,**{**c.CFG['bootstrap'],'block':block}).as_dict() for block in (40,60)}),primary


def gates(rows):
    g=c.CFG['promotion'];stopped=False
    for i,row in enumerate(sorted(rows,key=lambda r:r['paired']['score']['p_value'])):
        pair=row['paired'];threshold=g['alpha']/(len(rows)-i)
        ok=not stopped and pair['score']['inference_valid'] and pair['score']['p_value']<=threshold
        stopped=stopped or not ok
        row['holm_pass']=bool(ok);row['holm_threshold']=threshold
        row['passes_historical_gates']=bool(ok and pair['score']['diff_mean']>=g['min_score_gain'] and pair['fit']['diff_mean']>=g['min_fit_gain']
                                           and min(pair[k]['p_pos'] for k in ('fit','score'))>=g['p_pos']
                                           and min(row['scenario_score_gains'].values())>=0
                                           and all(min(v[k] for v in row['windows'].values())>=g['worst_window_gain'] for k in ('fit','score'))
                                           and sum(v['score']>0 for v in row['windows'].values())>=g['min_positive_windows'])


def run(root):
    done=c.RUN/'results.json'
    if done.exists():return c.checked_json(done)
    data=inputs();base,frame,P,_,cores,pstate,fstate=data
    cfg={**c.CFG['online'],**c.CFG['context']};rows={p:[] for p in ('discovery','viewed_transfer')};traces={}
    for name,state,conditional in [('daily',pstate,False),('prediction_state',pstate,True),('feature_state',fstate,True)]:
        pred,trace=predict(P,frame.target_everest.to_numpy(),base.era.to_numpy(),cores[0],state,cfg,conditional)
        traces[name]=trace
        pd.DataFrame(dict(era=base.era,prediction=pred),index=base.index).to_parquet(c.RUN/f'{name}-predictions.parquet')
        c.r.write_json(c.RUN/f'{name}-trace.json',dict(plan_sha256=c.PLAN,trace=trace))
        for phase in rows:
            row,scores=evaluate(pred,data,phase);rows[phase].append(dict(name=name,**row));scores.to_parquet(c.RUN/f'{name}-{phase}-scores.parquet')
    original=json.loads((root/'adaptive_score_confirmation_trace.json').read_text())['trace']
    daily={r['era']:r for r in traces['daily']};control=np.empty(len(base));max_error=0.
    for record in original:
        weights=np.asarray(record['weights']);new=np.asarray(daily[record['start']]['weights'])
        np.testing.assert_allclose(new,weights,atol=1e-12,rtol=1e-12);max_error=max(max_error,float(np.abs(new-weights).max()))
        mask=base.era.between(record['start'],record['end']).to_numpy();control[mask]=P[mask]@weights
    controls=[]
    for name,pred in [('original_100_era',control),('fixed_prior',.625*P[:,0]+.375*P[:,1])]:
        for phase in rows:
            row,_=evaluate(pred,data,phase);controls.append(dict(name=name,phase=phase,**row))
    for phase in rows:gates(rows[phase])
    # Descriptive association check: it never selects a feature, policy or parameter.
    associations={};unique=np.unique(base.era)
    basefit=c.r.metrics.per_era_fit_fast(pd.Series(P[:,0]),pd.Series(frame.target_everest.to_numpy()),pd.Series(base.era.to_numpy()))
    for j,name in enumerate(c.CFG['online']['components'][1:],start=1):
        fit=c.r.metrics.per_era_fit_fast(pd.Series(P[:,j]),pd.Series(frame.target_everest.to_numpy()),pd.Series(base.era.to_numpy()))-basefit
        for state_name,state in [('prediction',pstate),('feature',fstate)]:
            for w in c.CFG['discovery']+[dict(name='viewed_transfer',test_start=5340,test_end=5924)]:
                m=(unique>=w['test_start'])&(unique<=w['test_end']);vals=[]
                for col in state.T:
                    vals.append(float(spearmanr(col[m],fit.to_numpy()[m]).statistic) if np.std(col[m])>1e-12 else None)
                associations[f'{name}/{state_name}/{w["name"]}']=vals
    doc=dict(plan_sha256=c.PLAN,run_id=c.CFG['name'],label='adaptive historical research; discovery and transfer both viewed',
             comparisons=rows,controls=controls,source_boundary_weight_replay_max_error=max_error,
             state_associations=associations,promoted=False,reason='viewed-data investigation; no prospective qualification')
    c.r.write_json(done,doc);return doc


def report(doc):
    output=REPO/'handoffs/results';c.r.write_json(output/'context15-metrics.json',doc)
    lines=['# Daily and observable-state weighting 15','',doc['label'],'',
           '| interval | policy | FIT | baseline FIT | proxy score | baseline proxy | historical gates |',
           '|---|---|---|---|---|---|---|']
    for phase,rows in doc['comparisons'].items():
        for row in rows:
            lines.append(f'| {phase} | {row["name"]} | {row["metrics"]["fit"]:.8f} | {row["baseline"]["fit"]:.8f} | '
                         f'{row["metrics"]["score"]:.8f} | {row["baseline"]["score"]:.8f} | {row["passes_historical_gates"]} |')
    lines+=['',f'Source block-boundary weight replay max error: {doc["source_boundary_weight_replay_max_error"]:g}.',
            '', 'Native models are source-verified copies, not new fits. Label/state maturity guards are unchanged.',
            'No model was promoted. Baseline01 stays selected; no platform writes, uploads, submissions or staking.']
    (output/'context15-results.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,choices=['init','copy','run','report'])
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    root,parent=load()
    if args.stage=='copy':copy_source(root,parent)
    elif args.stage=='run':run(root)
    elif args.stage=='report':report(c.checked_json(c.RUN/'results.json'))


if __name__=='__main__':main()
