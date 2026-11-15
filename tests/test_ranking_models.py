import numpy as np
import pytest
from catboost import CatBoostRegressor,CatBoostRanker
from stealth.models import cat
from stealth.ranking_models import checkpoint_counts,checkpoint_predictions,cpu_predict,ordered,params,pool


def test_grouping_is_contiguous_and_keeps_feature_label_pairs():
    era=np.array([3,1,3,2,1,2],dtype=np.int32)
    X=np.arange(12).reshape(6,2);y=X[:,0]*.1
    xx,yy,ee=ordered(X,y,era)
    np.testing.assert_array_equal(ee,[1,1,2,2,3,3])
    np.testing.assert_allclose(yy,xx[:,0]*.1)
    assert checkpoint_counts(100,32)==[1,4,16,32,64,96,100]


@pytest.mark.parametrize('loss',['RMSE','QueryRMSE','YetiRank'])
def test_native_objective_and_serialized_cpu_inference(loss):
    rng=np.random.default_rng(17)
    era=np.repeat(np.arange(20,dtype=np.int32),10)
    X=rng.integers(0,10,(200,6)).astype(np.float32)
    X[::11,0]=np.nan
    y=np.tile(np.linspace(0,1,10),20)
    spec=dict(loss=loss,cap=4,learning_rate=.02,depth=3,l2_leaf_reg=20,threads=1,checkpoint_step=2)
    cls=CatBoostRegressor if loss=='RMSE' else CatBoostRanker
    model=cls(**params(spec,seed=17,device='CPU'))
    model.fit(pool(X,y,era,loss))
    prediction=cpu_predict(model,X)
    np.testing.assert_allclose(prediction,cat.predict(cat.from_bytes(cat.to_bytes(model)),X),atol=1e-12)
    stages=dict(checkpoint_predictions(model,X,4,2,1))
    assert sorted(stages)==checkpoint_counts(4,2)
    for count,pred in stages.items():np.testing.assert_allclose(pred,cpu_predict(model,X,count=count),atol=1e-12)
    assert np.isfinite(prediction).all()


def test_copied_dependencies_are_bound_to_source_and_destination_bytes(tmp_path,monkeypatch):
    import importlib.util,json
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/ranking.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec=importlib.util.spec_from_file_location('ranking_runner_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'REPO',tmp_path)
    cfg=dict(baseline={'cap':1},seed=17,threads=6,inner_width=200,gap=40,cadence=100,
             inner_benchmark_trees=200,train_min_era=2800,ensembles={})
    monkeypatch.setattr(module.c,'CFG',cfg)
    monkeypatch.setattr(module.c,'PLAN','new-plan')
    destination=tmp_path/'artifacts/new';destination.mkdir(parents=True)
    monkeypatch.setattr(module.c,'RUN',destination)
    root=tmp_path/'artifacts/temporal11';checkpoint=root/'early_4000/baseline';checkpoint.mkdir(parents=True)
    (root/'plan.json').write_text(json.dumps(dict(data_sha256='same-data',config=cfg,code_sha256={},plan_sha256='old-plan')))
    (destination/'plan.json').write_text(json.dumps(dict(data_sha256='same-data')))
    blob=checkpoint/'model0.bin';blob.write_bytes(b'original native model')
    (checkpoint/'done.json').write_text(json.dumps(dict(plan_sha256='old-plan',files={'model0.bin':module.c.r.sha(blob)},seconds=3.)))
    module.copy_controls()
    target=destination/'early_4000/baseline/done.json'
    copied=module.c.checked_json(target)
    assert copied['copied_from']=='temporal11' and copied['source_plan_sha256']=='old-plan' and copied['seconds']==0
    module.copy_controls()  # immutable same-plan reuse
    (target.parent/'model0.bin').write_bytes(b'corrupted destination')
    with pytest.raises(RuntimeError,match='corrupt checkpoint'):module.copy_controls()
    (target.parent/'model0.bin').write_bytes(blob.read_bytes())
    blob.write_bytes(b'changed source')
    with pytest.raises(RuntimeError,match='dependency bytes'):module.copy_controls()
