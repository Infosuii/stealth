#!/usr/bin/env python3
"""Fixed raw-target history controls with immutable native reference models."""
import argparse,hashlib,json,logging,shutil,time
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits
import objective as s
from stealth.objective_models import fit_gpu as original_gpu
from stealth.ranking_models import cpu_predict

ROOT=s.REPO;SOURCE=None


def filter_rows(spec,X,y,era,teacher):
    keep=np.asarray(era)>=spec['floor']
    return tuple(np.asarray(a)[keep] for a in (X,y,era,teacher))


def gpu_fit(spec,base,X,y,era,teacher,**kwargs):
    return original_gpu(spec,base,*filter_rows(spec,X,y,era,teacher),**kwargs)


def load():
    global SOURCE
    s.load();s.teacher_load();SOURCE=s.RUN;parent=json.loads((SOURCE/'plan.json').read_text())
    cfg=yaml.safe_load((ROOT/'configs/history.yaml').read_text());s.CFG=cfg;s.RUN=ROOT/'artifacts'/cfg['name'];s.RUN.mkdir(parents=True,exist_ok=True)
    markers=[SOURCE/b['name']/n/'done.json' for b in s.blocks(cfg['intervals'],100,40) for n in ('baseline','gpu_raw')]
    plan=dict(protocol='raw-history-v1',config=cfg,data_sha256=parent['data_sha256'],source_plan_sha256=s.r.sha(SOURCE/'plan.json'),teacher_sha256=s.TEACHER_SHA,
              source_markers={str(p.relative_to(SOURCE)):s.r.sha(p) for p in markers},
              code_sha256={f:s.r.sha(ROOT/f) for f in ['scripts/history_context.py','docs/HISTORY-18.md']})
    s.PLAN=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();plan['plan_sha256']=s.PLAN
    path=s.RUN/'plan.json'
    if path.exists() and json.loads(path.read_text())['plan_sha256']!=s.PLAN:raise RuntimeError('history experiment changed')
    if not path.exists():s.r.write_json(path,plan)
    s.FRAME=pd.read_parquet(ROOT/'data/futures_train.parquet',filters=[('exped','<','exped_5925')])
    s.FRAME['era']=s.FRAME.exped.str.rsplit('_',n=1).str[-1].astype(np.int32)
    if s.FRAME.era.max()>=5925 or not s.FRAME.index.is_unique:raise RuntimeError('history reader boundary/index')
    return parent


def copy_controls(parent):
    for key in ('baseline','seed','gap','cadence','inner_width','intervals'):
        if s.CFG[key]!=parent['config'][key]:raise RuntimeError('history control configuration changed')
    control=next(x for x in s.CFG['gpu_recipes'] if x['name']=='gpu_rolling')
    original=next(x for x in parent['config']['gpu_recipes'] if x['name']=='gpu_raw')
    if {k:v for k,v in control.items() if k not in ('name','floor')}!={k:v for k,v in original.items() if k!='name'}:raise RuntimeError('GPU rolling control changed')
    for block in s.blocks(s.CFG['intervals'],100,40):
        if block['train_end']-1024<400:raise RuntimeError('rolling control effective history changed')
        for old,new in [('baseline','cpu_expanding400'),('gpu_raw','gpu_rolling')]:
            src=SOURCE/block['name']/old;doc=json.loads((src/'done.json').read_text())
            if doc['plan_sha256']!=parent['plan_sha256'] or doc['teacher_manifest_sha256']!=s.TEACHER_SHA:raise RuntimeError('history control source plan/teacher')
            for f,digest in doc['files'].items():
                if s.r.sha(src/f)!=digest:raise RuntimeError('history control source bytes')
            dest=s.RUN/block['name']/new;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'done.json').exists():s.native_cached(dest/'done.json');continue
            for f in doc['files']:shutil.copyfile(src/f,dest/f)
            s.r.write_json(dest/'done.json',{**doc,'plan_sha256':s.PLAN,'seconds':0.,'copied_from':'objective16','source_done_sha256':s.r.sha(src/'done.json')})


def cpu_job(block,spec):
    directory=s.RUN/block['name']/spec['name']
    if (directory/'done.json').exists():s.native_cached(directory/'done.json');return f'{block["name"]}/{spec["name"]} reused'
    start=time.perf_counter();tr,te=s.splits(s.FRAME,block);tr=tr.loc[tr.era>=spec['floor']];X,Xt=s.arrays(tr,te)
    with threadpool_limits(limits=6):
        model,info=s.raw_fit(dict(history=spec['history'],half_life=None,score_aware=False),s.CFG['baseline'],X,tr.target_everest.to_numpy(),tr.era.to_numpy(),inner_width=200,gap=40,benchmark_trees=200,bound=s.CFG['score_bound'],seed=17,threads=6)
        prediction=s.lgbm.predict(model,Xt)
    info.update(declared_floor=spec['floor'],history=spec['history'],teacher_manifest_sha256=s.TEACHER_SHA)
    s.finish(directory,model,prediction,Xt,te,info,time.perf_counter()-start,'lgbm')
    return f'{block["name"]}/{spec["name"]} {time.perf_counter()-start:.1f}s, trees={info["n_iter"]}'


def summarize():
    path=s.RUN/'results.json'
    if path.exists():return s.checked(path)
    base=s.predictions('cpu_expanding400');frame=s.FRAME.loc[base.index]
    X=frame[[x for x in frame if x.startswith('feature_')]].to_numpy(float);cores=[s.core_mean(X),s.core_mean(X[:,::2]),s.core_mean(X[:,1::2])]
    teacher=s.TEACHER.reindex(base.index).to_numpy()
    components={r['name']:s.rank_by_era(s.predictions(r['name']).reindex(base.index).prediction,base.era) for r in s.CFG['cpu_recipes']+s.CFG['gpu_recipes']}
    cpu=sum(components[r['name']] for r in s.CFG['cpu_recipes'])/3;gpu=sum(components[r['name']] for r in s.CFG['gpu_recipes'])/3
    components['balanced_history']=(s.rank_by_era(cpu,base.era)+s.rank_by_era(gpu,base.era))/2
    names=['cpu_expanding3','gpu_expanding400','gpu_expanding3','balanced_history']
    for name in names:components[name+'_blend']=(s.rank_by_era(components[name],base.era)+components['cpu_expanding400'])/2
    rows={};controls=[]
    for w in s.CFG['intervals']:
        rr=[dict(name=n,**s.evaluate(components[n],base,teacher,frame,cores,w)) for n in s.CFG['comparisons']]
        g=s.CFG['promotion'];stopped=False
        for i,row in enumerate(sorted(rr,key=lambda r:r['paired']['score']['p_value'])):
            pair=row['paired'];limit=g['alpha']/(len(rr)-i);ok=not stopped and pair['score']['inference_valid'] and pair['score']['p_value']<=limit;stopped=stopped or not ok
            row['holm_pass']=bool(ok);row['holm_threshold']=limit
            row['passes_historical_gates']=bool(ok and pair['fit']['diff_mean']>=g['min_fit_gain'] and pair['score']['diff_mean']>=g['min_score_gain'] and min(pair[k]['p_pos'] for k in ['fit','score'])>=g['p_pos'] and min(row['scenario_score_gains'].values())>=0 and all(min(v[k] for v in row['windows'].values())>=g['worst_window_gain'] for k in ['fit','score']) and sum(v['score']>0 for v in row['windows'].values())>=g['min_positive_windows'])
        rows[w['name']]=rr
        for name in components:
            if name not in s.CFG['comparisons']:controls.append(dict(interval=w['name'],name=name,**s.evaluate(components[name],base,teacher,frame,cores,w)))
    records=[s.checked(p) for p in s.RUN.glob('*/*/done.json')]
    doc=dict(plan_sha256=s.PLAN,run_id=s.CFG['name'],label='adaptive historical raw-history research; all intervals viewed',comparisons=rows,controls=controls,promoted=False,
             checkpoint_groups=len(records),copied_groups=sum('copied_from' in r for r in records),serialized_replay_max_error=max(r['serialized_replay_max_error'] for r in records))
    s.r.write_json(path,doc);return doc


def report(doc):
    out=ROOT/'handoffs/results';s.r.write_json(out/'history18-metrics.json',doc)
    lines=['# Raw-target training history 18','',doc['label'],'','| interval | policy | FIT | baseline FIT | proxy score | baseline proxy | historical gates |','|---|---|---|---|---|---|---|']
    for phase,rows in doc['comparisons'].items():
        for r in rows:lines.append(f'| {phase} | {r["name"]} | {r["metrics"]["fit"]:.8f} | {r["baseline"]["fit"]:.8f} | {r["metrics"]["score"]:.8f} | {r["baseline"]["score"]:.8f} | {r["passes_historical_gates"]} |')
    lines+=['',f'{doc["checkpoint_groups"]} groups; {doc["copied_groups"]} copied immutable controls; serialization error {doc["serialized_replay_max_error"]:g}.',
            'No candidate was promoted and no platform state changed. Historical proxy scores do not establish future performance.']
    (out/'history18-results.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['init','copy','cpu','gpu','summarize','report','rehearse'],required=True);args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s');s.fit_gpu=gpu_fit;parent=load()
    if args.stage=='copy':copy_controls(parent)
    elif args.stage=='cpu':s.parallel([(b,r) for b in s.blocks(s.CFG['intervals'],100,40) for r in s.CFG['cpu_recipes']],cpu_job)
    elif args.stage=='gpu':
        for b in s.blocks(s.CFG['intervals'],100,40):
            for r in s.CFG['gpu_recipes']:s.gpu_job(b,r)
    elif args.stage=='summarize':print(json.dumps(summarize(),indent=2))
    elif args.stage=='report':report(summarize())
    elif args.stage=='rehearse':
        rng=np.random.default_rng(17);e=np.repeat(np.arange(3,1003),5);X=rng.integers(0,10,(len(e),8)).astype(np.float32);y=rng.integers(0,5,len(e))/4
        for r in s.CFG['gpu_recipes'][1:]:
            model,info=gpu_fit({**r,'cap':4,'depth':3,'checkpoint_step':2},s.CFG['baseline'],X,y,e,np.zeros(len(e)),width=40,gap=40,seed=17,bound=s.CFG['score_bound'])
            np.testing.assert_allclose(cpu_predict(model,X[:40]),s.cat.predict(s.cat.from_bytes(s.cat.to_bytes(model)),X[:40]),atol=1e-12)
            logging.info('expanded raw-history GPU rehearsal passed: floor %d',r['floor'])


if __name__=='__main__':main()
