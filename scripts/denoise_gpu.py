#!/usr/bin/env python3
"""Masked, target-free GPU feature learning with a chronological ridge prediction head."""
import argparse,copy,json,logging,math,os,time
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import pandas as pd
import torch
from torch import nn
import representations as rep
from campaign_gpu import fit_window
from stealth._runtime import basis_design,basis_forward
from stealth.representation_models import design_path
from stealth.metrics import per_era_fit_fast

log=logging.getLogger('denoise-gpu')


def train(spec,X,y,era,columns,seed,epochs,valid=None,selection=None):
    start=time.perf_counter()
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    design=basis_design(X,'linear');n=len(columns);width=design.shape[1]
    mean,std=design.mean(axis=0),np.maximum(design.std(axis=0),.1)
    raw=torch.as_tensor(design,dtype=torch.float32,device='cuda')
    mu=torch.as_tensor(mean,dtype=torch.float32,device='cuda')
    sd=torch.as_tensor(std,dtype=torch.float32,device='cuda')
    encoder=nn.Sequential(nn.Linear(width,spec['hidden']),nn.GELU(approximate='tanh'),nn.Dropout(spec['dropout']),
                           nn.Linear(spec['hidden'],spec['latent']),nn.GELU(approximate='tanh')).cuda()
    decoder=nn.Sequential(nn.Linear(spec['latent'],spec['hidden']),nn.GELU(approximate='tanh'),nn.Linear(spec['hidden'],n)).cuda()
    average=copy.deepcopy(encoder).eval()
    optimizer=torch.optim.AdamW([*encoder.parameters(),*decoder.parameters()],lr=spec['lr'],weight_decay=spec['weight_decay'])
    curves=[]
    # No labels, targets or target-validation features are used in encoder updates.
    for epoch in range(1,epochs+1):
        encoder.train();decoder.train()
        rate=spec['lr']*(.03+.97*.5*(1+math.cos(math.pi*(epoch-1)/spec['epochs'])))
        for group in optimizer.param_groups:group['lr']=rate
        order=torch.randperm(len(raw),device='cuda')
        total,steps=0.,0
        for first in range(0,len(raw),spec['batch']):
            original=raw[order[first:first+spec['batch']]]
            observed=original[:,n:]==0
            masked=(torch.rand_like(original[:,:n])<spec['corruption'])&observed
            corrupted=original.clone()
            corrupted[:,:n].masked_fill_(masked,0)
            corrupted[:,n:].masked_fill_(masked,1)
            predicted=decoder(encoder((corrupted-mu)/sd))
            weight=observed*(masked.to(torch.float32)+.1*(~masked).to(torch.float32))
            loss=(((predicted-original[:,:n])**2)*weight).sum()/weight.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_([*encoder.parameters(),*decoder.parameters()],1.,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for a,p in zip(average.parameters(),encoder.parameters()):a.lerp_(p,1-spec['ema'])
            total+=float(loss.detach());steps+=1
        if epoch%50==0 or epoch==epochs:
            value=total/max(steps,1);curves.append({'epoch':epoch,'reconstruction_mse':value})
            log.info('denoiser seed %d epoch %d/%d reconstruction MSE %.5f',seed,epoch,epochs,value)
    average.eval()
    def encode(d):
        output=[]
        with torch.no_grad():
            for first in range(0,len(d),8192):
                xx=torch.as_tensor(d[first:first+8192],dtype=torch.float32,device='cuda')
                output.append(average((xx-mu)/sd).cpu().numpy())
        return np.concatenate(output).astype(float)
    hidden=encode(design)
    hm,hs=hidden.mean(axis=0),np.maximum(hidden.std(axis=0),.1)
    combined=np.concatenate([design,(hidden-hm)/hs],axis=1)
    lambdas=spec['lambdas'] if valid is not None else [selection['selected_lambda']]
    coefs,_=design_path(combined,np.asarray(y)[:,0],era,lambdas)
    scores=None
    if valid is not None:
        vd=basis_design(valid[0],'linear')
        vh=encode(vd)
        prediction=np.concatenate([vd,(vh-hm)/hs],axis=1)@coefs
        scores=[float(per_era_fit_fast(pd.Series(prediction[:,j]),pd.Series(valid[1]),pd.Series(valid[2])).mean()) for j in range(len(lambdas))]
        best=int(np.argmax(scores))
    else:best=0
    coef=coefs[:,best]
    result={'kind':'net','columns':columns,'coef':coef[:width],'x_mean':mean,'x_std':std,
            'W0':average[0].weight.detach().cpu().numpy().T.copy().astype(float),'b0':average[0].bias.detach().cpu().numpy().copy().astype(float),
            'W1':average[3].weight.detach().cpu().numpy().T.copy().astype(float),'b1':average[3].bias.detach().cpu().numpy().copy().astype(float),
            'W2':(coef[width:]/hs)[:,None],'b2':np.array([-hm@(coef[width:]/hs)])}
    scale=max(float(np.std(basis_forward(result,X))),1e-6)
    result['coef']=result['coef']/scale;result['W2']=result['W2']/scale;result['b2']=result['b2']/scale
    reference=combined[:256]@coef/scale
    error=float(np.max(np.abs(reference-basis_forward(result,X[:256]))))
    if not np.isfinite(error) or error>1e-4:raise RuntimeError(f'denoiser numeric export mismatch: {error}')
    info={'trained_epochs':epochs,'best_epoch':epochs,'selected_lambda':float(lambdas[best]),'head_inner_scores':scores,
          'curves':curves,'seconds':time.perf_counter()-start,'export_error':error,'peak_gpu_mb':torch.cuda.max_memory_allocated()/2**20}
    del raw,optimizer,encoder,decoder,average
    torch.cuda.empty_cache()
    return result,info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--confirmation',action='store_true');parser.add_argument('--full',action='store_true')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.smoke:
        rng=np.random.default_rng(1);X=rng.integers(0,10,size=(256,4)).astype(float);y=rng.normal(size=(256,1));era=np.repeat(np.arange(16),16)
        spec={'hidden':16,'latent':4,'dropout':.1,'corruption':.25,'lr':.0003,'weight_decay':.01,'epochs':5,'batch':64,'ema':.99,'lambdas':[1,100]}
        _,info=train(spec,X,y,era,list('abcd'),17,5,valid=(X[:64],y[:64,0],era[:64]))
        model,info=train(spec,X,y,era,list('abcd'),17,5,selection=info)
        assert np.isfinite(basis_forward(model,X)).all()
        log.info('Denoising CUDA smoke passed; export error %.3g',info['export_error']);return
    rep.load('confirmation' if args.confirmation or args.full else 'discovery')
    c=rep.c;spec=next(s for s in c.r.CONFIG['candidates'] if s['kind']=='denoised')
    if args.confirmation or args.full:
        selected=c.selection()['selected']
        if not selected or selected['name']!=spec['name']:raise RuntimeError('no frozen denoising winner')
    if args.full:
        if not json.loads((c.r.RUN/'confirmation.json').read_text())['promoted']:raise RuntimeError('no promoted winner')
        windows=[{'name':'full','train_end':5925,'test_start':5925,'test_end':5924}]
    else:windows=[c.r.CONFIG['confirmation']] if args.confirmation else c.r.CONFIG['discovery']
    for w in windows:fit_window(w,spec,full=args.full,trainer=train)
    c.r.write_json(c.r.RUN/('gpu-full.json' if args.full else 'gpu-confirmation.json' if args.confirmation else 'gpu-discovery.json'),{'status':'done','plan_sha256':c.r.PLAN_SHA})


if __name__=='__main__':main()
