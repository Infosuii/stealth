import numpy as np
from stealth.objective_models import labels,fit_cpu
from stealth.models import lgbm


def test_utility_targets_center_scale_and_do_not_invent_benchmark_information():
    rng=np.random.default_rng(1);era=np.repeat(np.arange(20),8);y=rng.normal(size=len(era));core=rng.normal(size=len(era))
    a=labels('core',y,era,core,np.zeros(len(y)));b=labels('full',y,era,core,np.ones(len(y)))
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(a,labels('core',y+100*era,era,core,np.zeros(len(y))),atol=1e-12)
    centered=y-np.repeat(y.reshape(20,8).mean(axis=1),8)
    np.testing.assert_allclose(a.reshape(20,8).mean(axis=1),0,atol=1e-14)
    np.testing.assert_allclose(a.std(),centered.std(),atol=1e-14)
    np.testing.assert_array_equal(labels('raw',y,era,core,np.zeros(len(y))),y)


def test_native_full_utility_fit_has_own_purged_count_and_portable_booster():
    rng=np.random.default_rng(2);era=np.repeat(np.arange(400,800),5);X=rng.integers(0,10,(len(era),5));y=rng.normal(size=len(era));ref=rng.normal(size=len(era))
    spec=dict(history=350,target='full');base=dict(cap=5,num_leaves=3,min_data_in_leaf=20,early_stopping_rounds=2)
    model,info=fit_cpu(spec,base,X,y,era,ref,width=40,gap=40,seed=17,threads=1,bound=2/np.pi)
    assert info['train_min']==450 and info['inner_valid_min']-info['inner_fit_max']>40
    np.testing.assert_allclose(lgbm.predict(model,X[:30]),lgbm.predict(lgbm.from_bytes(lgbm.to_bytes(model)),X[:30]),atol=1e-12)


def test_outer_job_partition_cannot_expose_future_labels(monkeypatch):
    import importlib.util
    from pathlib import Path
    import pandas as pd
    path=Path(__file__).resolve().parents[1]/'scripts/objective.py';monkeypatch.syspath_prepend(str(path.parent))
    spec=importlib.util.spec_from_file_location('objective_runner_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    frame=pd.DataFrame(dict(era=np.arange(400,2100),target_everest=np.arange(1700)/1700))
    block=dict(train_end=1960,test_start=2000,test_end=2099)
    before,_=module.splits(frame,block);changed=frame.copy();changed.loc[changed.era>=1960,'target_everest']=999
    after,_=module.splits(changed,block)
    pd.testing.assert_frame_equal(before,after)
    assert before.era.max()<2000-40
