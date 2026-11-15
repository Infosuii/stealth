#!/usr/bin/env python3
"""Matched deterministic target/count diagnostics; cannot select a candidate."""
import argparse,hashlib,json,logging,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import objective as s
from stealth.objective_models import prepare,labels
from stealth.temporal_models import rank_by_era,core_mean
from stealth.models import lgbm


def fixed(block,spec,count):
    tr,te=s.splits(s.FRAME,block);X,Xt=s.arrays(tr,te)
    X,y,e,teacher,_,_,core=prepare(spec,X,tr.target_everest.to_numpy(),tr.era.to_numpy(),s.TEACHER.reindex(tr.index).to_numpy(),s.CFG['inner_width'],40)
    p=dict(s.CFG['baseline']);p.pop('cap')
    with threadpool_limits(limits=6):
        model=lgbm.fit_fixed(p,X,labels(spec['target'],y,e,core,teacher),None,n_iter=count,seed=17,threads=6)
        prediction=lgbm.predict(model,Xt);blob=lgbm.to_bytes(model)
        np.testing.assert_allclose(prediction,lgbm.predict(lgbm.from_bytes(blob),Xt),atol=1e-12,rtol=1e-12)
    return blob,prediction,te


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,choices=['run','report']);args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s');s.load();s.teacher_load()
    root=s.RUN;out=s.REPO/'artifacts/objective17';out.mkdir(parents=True,exist_ok=True)
    blocks=s.blocks(s.CFG['intervals'],100,40)
    markers=[root/b['name']/name/'done.json' for b in blocks for name in ('cpu_core','cpu_full')]
    for marker in markers:s.native_cached(marker)
    plan=dict(protocol='cpu-target-count-factorial-v1',source_plan_sha256=s.r.sha(root/'plan.json'),teacher_sha256=s.TEACHER_SHA,
              native_sha256={str(p.relative_to(root)):s.r.sha(p) for p in markers},
              code_sha256={f:s.r.sha(s.REPO/f) for f in ['scripts/objective_counts.py','docs/OBJECTIVE-17-COUNTS.md']})
    digest=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();plan['plan_sha256']=digest
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())['plan_sha256']!=digest:raise RuntimeError('count diagnostic changed')
    if not (out/'plan.json').exists():s.r.write_json(out/'plan.json',plan)
    if args.stage=='report':
        doc=json.loads((out/'results.json').read_text());dest=s.REPO/'handoffs/results';s.r.write_json(dest/'objective17-metrics.json',doc)
        lines=['# Utility target/count controls 17','',doc['label'],'','| interval | cell | FIT | proxy score |','|---|---|---|---|']
        for row in doc['comparisons']:lines.append(f'| {row["interval"]} | {row["name"]} | {row["metrics"]["fit"]:.8f} | {row["metrics"]["score"]:.8f} |')
        lines+=['',f'Own-count replay max error: {doc["own_count_replay_max_error"]:g}.',
                'Diagnostics cannot select a new candidate or change previous gates. No platform writes.']
        (dest/'objective17-results.md').write_text('\n'.join(lines)+'\n');return
    maximum=0.;parts={'full_at_core':[],'core_at_full':[]};refits=0;reuses=0
    for block in blocks:
        own={name:json.loads((root/block['name']/name/'training.json').read_text())['n_iter'] for name in ('cpu_core','cpu_full')}
        if block['test_start'] in (2000,4000,5340):
            for name in own:
                spec=next(x for x in s.CFG['cpu_recipes'] if x['name']==name)
                _,pred,te=fixed(block,spec,own[name]);old=pd.read_parquet(root/block['name']/name/'prediction.parquet').reindex(te.index)
                np.testing.assert_allclose(pred,old.prediction,atol=1e-12,rtol=1e-12);maximum=max(maximum,float(np.abs(pred-old.prediction).max()))
        for name,original,donor in [('full_at_core','cpu_full','cpu_core'),('core_at_full','cpu_core','cpu_full')]:
            directory=out/block['name']/name;directory.mkdir(parents=True,exist_ok=True);marker=directory/'done.json'
            if marker.exists():
                record=json.loads(marker.read_text())
                if record['plan_sha256']!=digest:raise RuntimeError('changed diagnostic plan')
                for f,h in record['files'].items():
                    if s.r.sha(directory/f)!=h:raise RuntimeError('changed diagnostic bytes')
            else:
                count=own[donor];reuse=count==own[original]
                if reuse:
                    result=pd.read_parquet(root/block['name']/original/'prediction.parquet');files=[]
                else:
                    spec=next(x for x in s.CFG['cpu_recipes'] if x['name']==original)
                    blob,pred,te=fixed(block,spec,count);(directory/'model.bin').write_bytes(blob)
                    result=pd.DataFrame(dict(era=te.era,prediction=pred),index=te.index);files=['model.bin']
                result.to_parquet(directory/'prediction.parquet');files+=['prediction.parquet']
                record=dict(plan_sha256=digest,count=count,own_count=own[original],source_target=original,count_donor=donor,reused_equal_count=reuse,files={f:s.r.sha(directory/f) for f in files})
                s.r.write_json(marker,record)
            reuses+=record['reused_equal_count'];refits+=not record['reused_equal_count'];parts[name].append(pd.read_parquet(directory/'prediction.parquet'))
        logging.info('matched count controls: %s',block['name'])
    base=s.predictions('baseline');frame=s.FRAME.loc[base.index];X=frame[[x for x in frame if x.startswith('feature_')]].to_numpy(float)
    cores=[core_mean(X),core_mean(X[:,::2]),core_mean(X[:,1::2])];teacher=s.TEACHER.reindex(base.index).to_numpy()
    predictions={name:pd.concat(data).reindex(base.index).prediction.to_numpy() for name,data in parts.items()}
    predictions.update({name:s.predictions(name).reindex(base.index).prediction.to_numpy() for name in ('cpu_core','cpu_full')})
    rows=[]
    for interval in s.CFG['intervals']:
        for name,pred in predictions.items():rows.append(dict(interval=interval['name'],name=name,**s.evaluate(pred,base,teacher,frame,cores,interval)))
    s.r.write_json(out/'results.json',dict(plan_sha256=digest,label='descriptive matched CPU target/count controls; no selection or promotion',comparisons=rows,
                                        own_count_replay_max_error=maximum,changed_count_refits=refits,equal_count_reuses=reuses))


if __name__=='__main__':main()
