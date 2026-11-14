#!/usr/bin/env python3
"""Matched raw-input preservation intervention on the saved latent models."""
import json,logging,time
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
import diagnose_models as d
from stealth import _runtime as rt
from stealth.era_boost import centred_target
from stealth.representation_models import params_for
from stealth.research_models import to_bytes


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    out=d.REPO/'artifacts/forensics06b';out.mkdir(parents=True,exist_ok=True)
    plan={'source_plan_sha256':d.sha(d.REPO/'artifacts/forensics06/plan.json'),'code_sha256':d.sha(__file__),
          'source_data_sha256':d.sha(d.REPO/'data/futures_train.parquet'),'reader_ceiling':5300}
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:raise RuntimeError('intervention run changed')
    d.write(out/'plan.json',plan)
    frame,columns=d.frame_data()
    windows=json.loads((d.REPO/'artifacts/representations04/plan.json').read_text())['config']['discovery']
    with threadpool_limits(limits=6):
        for w in windows:
            folder,done,models,cached=d.checkpoint('representations04',w['name'],'latent_trees')
            _,_,_,base=d.checkpoint('representations04',w['name'],'baseline')
            tr,te,X,Xt=d.arrays(frame,columns,w);era=te.era.to_numpy();train_era=tr.era.to_numpy()
            records=json.loads((folder/'training.json').read_text())['selected']
            original=np.array([d.predict('basis',m,Xt,era) for m in models]);new=[];changed=0
            np.testing.assert_allclose(original.mean(axis=0),cached.prediction.to_numpy(),atol=1e-10)
            dst=out/w['name'];dst.mkdir(exist_ok=True)
            for i,(model,record) in enumerate(zip(models,records)):
                candidate={k:v for k,v in model.items() if not k.startswith('_')}
                if not bool(model['add_raw']):
                    changed+=1;candidate['add_raw']=True
                    xx=rt.latent_design(candidate,X,train_era).astype(np.float32)
                    yy,weights=centred_target(tr.target_everest.to_numpy(),train_era)
                    booster=lgb.train(params_for(record['params'],len(yy),17,6),lgb.Dataset(xx,label=yy,weight=weights,params={'max_bin':63}),num_boost_round=record['count'])
                    candidate['tree_blob']=np.frombuffer(booster.model_to_string().encode(),dtype=np.uint8)
                pred=d.predict('basis',candidate,Xt,era);new.append(pred)
                if bool(model['add_raw']):np.testing.assert_allclose(pred,original[i],atol=1e-10)
                (dst/f'model{i}.bin').write_bytes(to_bytes(candidate))
            cases={'original/raw':original.mean(axis=0),'original/rank':d.rank_ensemble(original,era),
                   'preserved/raw':np.mean(new,axis=0),'preserved/rank':d.rank_ensemble(new,era)}
            scores,fits=d.score_cases(cases,base.prediction.to_numpy(),te)
            d.pd.DataFrame(fits).to_parquet(dst/'fits.parquet')
            d.write(dst/'ablation.json',{'window':w['name'],'changed_members':changed,'results':scores})
            logging.info('%s raw-input intervention finished; changed %d members',w['name'],changed)
    d.write(out/'done.json',{'status':'done','plan_sha256':d.sha(out/'plan.json')})


if __name__=='__main__':main()
