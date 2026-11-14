#!/usr/bin/env python3
"""Long neural fits in the GPU interpreter. Numeric, primary-head-only portable exports."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import campaign as c
from stealth._runtime import basis_design,basis_forward
from stealth.research_models import era_target,fit_basis,to_bytes
from stealth.metrics import per_era_fit_fast
from research_gpu import export

log=logging.getLogger('campaign-gpu')


def targets(y,era):
    y=np.asarray(y,dtype=float)
    if y.ndim==1:y=y[:,None]
    mask=np.isfinite(y)
    result=np.zeros_like(y)
    for j in range(y.shape[1]):
        keep=mask[:,j]
        if keep.any():result[keep,j]=era_target(y[keep,j],np.asarray(era)[keep])[0]
    return result,mask.astype(np.float32)


def tensors(design,y,era,skip,mean,std):
    order=np.argsort(era,kind='stable')
    d,y,e=design[order],np.asarray(y)[order],np.asarray(era)[order]
    unique,group,count=np.unique(e,return_inverse=True,return_counts=True)
    pos=np.arange(len(e))-np.repeat(np.cumsum(count)-count,count)
    t,m=targets(y,e)
    shape=(len(unique),int(count.max()))
    features=np.zeros((*shape,d.shape[1]),dtype=np.float32)
    target=np.zeros((*shape,t.shape[1]),dtype=np.float32)
    mask=np.zeros_like(target)
    base=np.zeros(shape,dtype=np.float32)
    features[group,pos]=(d-mean)/std
    target[group,pos]=t
    mask[group,pos]=m
    base[group,pos]=d@skip
    return [torch.as_tensor(a,device='cuda') for a in (features,target,mask,base)]


def train(spec,X,y,era,columns,seed,epochs,valid=None,selection=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    design=basis_design(X,'linear')
    skip=fit_basis({'kind':'linear','alpha':spec['skip_alpha']},X,np.asarray(y)[:,0],era,columns,seed=seed,threads=2)['coef']
    mean,std=design.mean(axis=0),np.maximum(design.std(axis=0),.1)
    batches=tensors(design,y,era,skip,mean,std)
    hidden=int(spec['hidden'])
    model=nn.Sequential(nn.Linear(design.shape[1],hidden),nn.GELU(approximate='tanh'),nn.Dropout(spec['dropout']),
        nn.Linear(hidden,hidden//2),nn.GELU(approximate='tanh'),nn.Dropout(spec['dropout']),nn.Linear(hidden//2,np.asarray(y).shape[1])).cuda()
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    average=copy.deepcopy(model).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=spec['weight_decay'])
    def lr_factor(epoch):return .03+.97*.5*(1+math.cos(math.pi*epoch/spec['epochs']))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    head=torch.zeros((1,1,np.asarray(y).shape[1]),device='cuda')
    head[:,:,0]=1
    rng=np.random.default_rng(seed)
    best,best_epoch,best_state=-float('inf'),0,None
    curves=[]
    if valid is not None:
        vd=basis_design(valid[0],'linear')
        vx=torch.as_tensor((vd-mean)/std,dtype=torch.float32,device='cuda')
        vb=vd@skip
        def evaluate():
            with torch.no_grad():
                pred=average(vx)[:,0].cpu().numpy()+vb
            return float(per_era_fit_fast(pd.Series(pred),pd.Series(valid[1]),pd.Series(valid[2])).mean())
        best=evaluate()
        best_state=copy.deepcopy(average.state_dict())
        curves.append({'epoch':0,'inner_fit':best,'train_mse':None})
    for epoch in range(1,epochs+1):
        model.train()
        loss_sum,steps=0.,0
        ids=rng.permutation(len(batches[0]))
        for first in range(0,len(ids),spec['batch_eras']):
            inputs,target,mask,base=[a[ids[first:first+spec['batch_eras']]] for a in batches]
            prediction=model(inputs)+base[:,:,None]*head
            count=mask.sum(dim=1).clamp_min(1)
            prediction=prediction-(prediction*mask).sum(dim=1)[:,None,:]/count[:,None,:]
            mse=(((prediction-target)*mask)**2).sum(dim=1)/count
            loss=mse[:,0].mean()
            if mse.shape[1]>1:loss=loss+spec['aux_weight']*mse[:,1:].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for a,p in zip(average.parameters(),model.parameters()):a.lerp_(p,1-spec['ema'])
            loss_sum+=float(loss.detach())
            steps+=1
        scheduler.step()
        if epoch%spec['eval_every']==0 or epoch==epochs:
            score=evaluate() if valid is not None else None
            curves.append({'epoch':epoch,'inner_fit':score,'train_mse':loss_sum/max(steps,1)})
            if score is not None and score>best+1e-8:
                best,best_epoch,best_state=score,epoch,copy.deepcopy(average.state_dict())
            if epoch%50==0 or epoch==epochs:
                log.info('%s seed %d epoch %d/%d train MSE %.5f inner %s best epoch %d',spec['name'],seed,epoch,epochs,loss_sum/max(steps,1),None if score is None else round(score,5),best_epoch)
    if valid is not None:average.load_state_dict(best_state)
    average.eval()
    result=export(average,skip,mean,std,columns)
    result['W2']=result['W2'][:,:1]
    result['b2']=result['b2'][:1]
    # Scale comes solely from this training fit and is frozen before any outer prediction.
    scale=max(float(np.std(basis_forward(result,X))),1e-6)
    result['coef']=result['coef']/scale
    result['W2']=result['W2']/scale
    result['b2']=result['b2']/scale
    with torch.no_grad():
        xx=torch.as_tensor((design[:256]-mean)/std,dtype=torch.float32,device='cuda')
        reference=(average(xx)[:,0].cpu().numpy()+design[:256]@skip)/scale
    error=float(np.max(np.abs(reference-basis_forward(result,X[:256]))))
    if error>1e-4:raise RuntimeError(f'GPU/export arithmetic mismatch: {error}')
    info={'best_epoch':best_epoch if valid is not None else epochs,'trained_epochs':epochs,'curves':curves,
          'seconds':time.perf_counter()-start,'export_error':error,'peak_gpu_mb':torch.cuda.max_memory_allocated()/2**20}
    del optimizer,model,average,batches,design
    if valid is not None:del vx
    torch.cuda.empty_cache()
    return result,info


def fit_window(window,spec,full=False,trainer=None):
    trainer=trainer or train
    r=c.r
    if r.cached(window,spec['name']):return
    start=time.perf_counter()
    if full:
        columns=[x for x in r.FRAME if x.startswith('feature_')]
        tr,te=r.FRAME,r.FRAME.iloc[:0]
        X=tr[columns].to_numpy(dtype=float,copy=True);X[X<0]=np.nan
        Xt=X[:0]
    else:columns,tr,te,X,Xt=r.arrays(window)
    targets_cols=['target_everest']
    if spec.get('aux_weight',0):
        targets_cols+=[x for x in tr if x.startswith('target_') and x!='target_everest']
    y,era=tr[targets_cols].to_numpy(dtype=float),tr.era.to_numpy()
    unique=np.unique(era);cut=unique[int(len(unique)*.8)]
    inner,valid=era<cut-40,era>=cut
    blobs,predictions,epochs=[],[],[]
    for seed in spec['seeds']:
        name=f'{spec["name"]}/seed{seed}'
        folder=r.cache_dir(window,name)
        if r.cached(window,name):
            blob=(folder/'model0.bin').read_bytes()
            pred=r.read_prediction(window,name).prediction.to_numpy()
            info=json.loads((folder/'done.json').read_text())
            count=info['epochs']
        else:
            _,inner_info=trainer(spec,X[inner],y[inner],era[inner],columns,seed,spec['epochs'],valid=(X[valid],y[valid,0],era[valid]))
            count=inner_info['best_epoch']
            model,full_info=trainer(spec,X,y,era,columns,seed,count,selection=inner_info)
            pred=basis_forward(model,Xt) if len(Xt) else np.empty(0)
            blob=to_bytes(model)
            folder.mkdir(parents=True,exist_ok=True)
            r.write_json(folder/'training.json',{'inner':inner_info,'refit':full_info,'targets':targets_cols})
            r.save_fit(window,name,'basis',[blob],pred,te,inner_info['seconds']+full_info['seconds'],
                {'epochs':count,'trained_inner_epochs':spec['epochs'],'training_sha256':r.sha(folder/'training.json')})
            log.info('%s/%s seed %d finished; selected %d after %d inner epochs',window['name'],spec['name'],seed,count,spec['epochs'])
        blobs.append(blob);predictions.append(pred);epochs.append(count)
    r.save_fit(window,spec['name'],'basis',blobs,np.mean(predictions,axis=0),te,time.perf_counter()-start,
        {'epochs':epochs,'trained_inner_epochs_per_seed':spec['epochs'],'device':torch.cuda.get_device_name(0)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmation',action='store_true')
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no silent CPU fallback')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.smoke:
        rng=np.random.default_rng(1)
        X=rng.integers(0,10,size=(256,4)).astype(float)
        y=rng.normal(size=(256,2));era=np.repeat(np.arange(16),16)
        spec={'name':'smoke','hidden':16,'dropout':.4,'weight_decay':.03,'lr':.0003,'epochs':5,'batch_eras':8,'eval_every':1,'ema':.99,'aux_weight':.3,'skip_alpha':10000}
        model,info=train(spec,X,y,era,list('abcd'),17,5,valid=(X[:64],y[:64,0],era[:64]))
        assert np.isfinite(basis_forward(model,X)).all() and info['trained_epochs']==5
        log.info('Long-neural CUDA smoke passed; export error %.3g',info['export_error'])
        return
    c.load('confirmation' if args.confirmation or args.full else 'discovery')
    specs=[s for s in c.r.CONFIG['candidates'] if s['kind']=='long_net']
    if args.confirmation or args.full:
        selected=c.selection()['selected']
        if not selected or selected['name'] not in [s['name'] for s in specs]:raise RuntimeError('no frozen neural winner')
        specs=[s for s in specs if s['name']==selected['name']]
    if args.full:
        if not json.loads((c.r.RUN/'confirmation.json').read_text())['promoted']:raise RuntimeError('no promoted neural winner')
        windows=[{'name':'full','train_end':5925,'test_start':5925,'test_end':5924}]
    else:windows=[c.r.CONFIG['confirmation']] if args.confirmation else c.r.CONFIG['discovery']
    for spec in specs:
        for window in windows:fit_window(window,spec,full=args.full)
    c.r.write_json(c.r.RUN/('gpu-full.json' if args.full else 'gpu-confirmation.json' if args.confirmation else 'gpu-discovery.json'),{'status':'done','plan_sha256':c.r.PLAN_SHA})


if __name__=='__main__':main()
