#!/usr/bin/env python3
"""Balanced denoising-head repair, with verified target-free encoder reuse."""
import argparse,json,logging
import repairs as rep
from campaign_gpu import fit_window
from denoise_gpu import train
import diagnose_models as forensic


def encoder_source(window,spec,seed):
    source=rep.c.REPO/'artifacts'/spec['reuse_encoders_from']
    seed_name=f'denoised_ridge/seed{seed}'
    if not (source/window['name']/seed_name/'done.json').exists():return None
    plan=json.loads((source/'plan.json').read_text())
    current=json.loads((rep.c.r.RUN/'plan.json').read_text())
    if plan['data_sha256']!=current['data_sha256']:raise RuntimeError('encoder source data mismatch')
    prior=next(w for w in plan['config']['discovery'] if w['name']==window['name'])
    if prior!=window or plan['config']['train_min_era']!=rep.c.r.CONFIG['train_min_era']:
        raise RuntimeError('encoder fitting partition changed')
    folder,done,models,_=forensic.checkpoint(spec['reuse_encoders_from'],window['name'],seed_name)
    record=json.loads((folder/'training.json').read_text())['refit']
    if record['trained_epochs']!=spec['epochs']:raise RuntimeError('source encoder schedule changed')
    return {'model':models[0],'sha256':forensic.sha(folder/'model0.bin'),'trained_epochs':record['trained_epochs']}


def main():
    # Import torch only inside the existing dedicated GPU modules.
    from denoise_gpu import torch
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--confirmation',action='store_true');parser.add_argument('--full',action='store_true');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.smoke:
        from denoise_gpu import np
        rng=np.random.default_rng(9);X=rng.integers(0,10,size=(256,4)).astype(float);y=rng.normal(size=(256,1));era=np.repeat(np.arange(16),16)
        s={'hidden':16,'latent':4,'dropout':.1,'corruption':.25,'lr':.0003,'weight_decay':.01,'epochs':5,'batch':64,'ema':.99,'lambdas':[.001,.1,10],'raw_group_weights':[0.,.5,1.]}
        old,info=train(s,X,y,era,list('abcd'),17,5,valid=(X[:64],y[:64,0],era[:64]))
        new,check=train(s,X,y,era,list('abcd'),17,5,selection=info,encoder_source={'model':old,'trained_epochs':5,'sha256':'synthetic'})
        for k in ['W0','b0','W1','b1']:np.testing.assert_array_equal(new[k],old[k])
        assert check['trained_epochs']==0 and check['reused_encoder_epochs']==5
        logging.info('Balanced-head CUDA/reuse smoke passed; error %.3g',check['export_error']);return
    rep.load('confirmation' if args.confirmation or args.full else 'discovery');c=rep.c
    spec=next(s for s in c.r.CONFIG['candidates'] if s['kind']=='denoised')
    if args.confirmation or args.full:
        selected=c.selection()['selected']
        if not selected or selected['name']!=spec['name']:raise RuntimeError('no frozen denoising winner')
    if args.full:
        if not json.loads((c.r.RUN/'confirmation.json').read_text())['promoted']:raise RuntimeError('no promoted winner')
        windows=[{'name':'full','train_end':5925,'test_start':5925,'test_end':5924}]
    else:windows=[c.r.CONFIG['confirmation']] if args.confirmation else c.r.CONFIG['discovery']
    for window in windows:
        def trainer(sp,X,y,era,columns,seed,epochs,valid=None,selection=None):
            cached=encoder_source(window,sp,seed) if valid is None else None
            return train(sp,X,y,era,columns,seed,epochs,valid=valid,selection=selection,encoder_source=cached)
        fit_window(window,spec,full=args.full,trainer=trainer)
    c.r.write_json(c.r.RUN/('gpu-full.json' if args.full else 'gpu-confirmation.json' if args.confirmation else 'gpu-discovery.json'),{'status':'done','plan_sha256':c.r.PLAN_SHA})


if __name__=='__main__':main()
