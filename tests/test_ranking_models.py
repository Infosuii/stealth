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
