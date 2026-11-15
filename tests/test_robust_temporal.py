import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stealth.robust_temporal import core_columns,fit,median_count
from stealth.temporal_models import rank_by_era


def test_core_partition_is_label_free_and_group_columns_cover_original():
    X = np.arange(48).reshape(8,6)
    even,odd = (core_columns(dict(training_core=k),X) for k in ('even','odd'))
    out = np.empty_like(X);out[:,::2],out[:,1::2] = even,odd
    np.testing.assert_array_equal(out,X)
    assert median_count([1,100,8])==8
    with pytest.raises(ValueError):median_count([0,100,8])


def test_multi_window_counts_have_only_own_purged_sources():
    rng = np.random.default_rng(22)
    era = np.repeat(np.arange(400),4)
    X = rng.integers(0,10,(len(era),5)).astype(float)
    y = .5+X[:,0]*.01+rng.normal(0,.1,len(era))
    spec = dict(history=400,half_life=None,score_aware=True,stopping_windows=3)
    model,info = fit(spec,dict(num_leaves=3,min_data_in_leaf=30,cap=5,early_stopping_rounds=2),X,y,era,
                     inner_width=30,gap=40,benchmark_trees=3,bound=2/np.pi,seed=17,threads=1)
    assert info['n_iter']==median_count(info['counts']) and len(info['counts'])==3
    sources = info['count_sources']
    assert all(s['valid_min']-s['fit_max']>40 for s in sources)
    assert sources[0]['valid_max']<sources[1]['valid_min']<sources[2]['valid_min']
    assert max(s['valid_max'] for s in sources)<=era.max()
    assert np.isfinite(model.predict(X[:20],num_threads=1)).all()


def test_nested_groups_rank_each_group_before_averaging(monkeypatch):
    path = Path(__file__).resolve().parents[1]/'scripts/robust_temporal.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('robust_temporal_runner_test',path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg = dict(ensembles={'group':{'a':1,'b':1},'outer':{'group':1,'c':1}})
    monkeypatch.setattr(module.c,'CFG',cfg)
    arrays = dict(a=np.array([0.,1,2,3,4]),b=np.array([0,1,4,2,3.]),c=np.array([4,1,0,3,2.]))
    base = pd.DataFrame(dict(era=[1]*5))
    def predictions(name,confirmation=False):return pd.DataFrame(dict(prediction=arrays[name]))
    monkeypatch.setattr(module.c,'predictions',predictions)
    group = (rank_by_era(arrays['a'],base.era)+rank_by_era(arrays['b'],base.era))/2
    expected = (rank_by_era(group,base.era)+rank_by_era(arrays['c'],base.era))/2
    np.testing.assert_allclose(module.prediction('outer',base),expected)
    assert module.dependencies('outer')=={'a','b','c'}
    arrays['a'] = 1e6+arrays['a']*999
    np.testing.assert_allclose(module.prediction('outer',base),expected)
