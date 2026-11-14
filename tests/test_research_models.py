"""Era-balanced objective, exported arithmetic, research boundaries and isolated runtime portability."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stealth import _runtime as rt, bundle, pkl_check
from stealth.research_models import era_target, fit_basis, to_bytes


def fixture_arrays():
    rng = np.random.default_rng(3)
    X = rng.integers(0, 10, size=(160, 4)).astype(float)
    X[rng.random(X.shape) < .1] = np.nan
    era = np.repeat(np.arange(20), 8)
    y = np.nan_to_num(X[:, 0], nan=4.5) * .1 + rng.normal(0, .2, len(X))
    return X, y, era, ['a', 'b', 'c', 'd']


def test_centred_objective_equals_direct_normal_equations():
    X, y, era, columns = fixture_arrays()
    model = fit_basis({'kind':'linear', 'alpha':100}, X, y, era, columns)
    design = rt.basis_design(X, 'linear')
    target, weights, group, counts = era_target(y, era)
    centered = design.copy()
    for i in range(len(counts)):
        centered[group == i] -= centered[group == i].mean(axis=0)
    direct = np.linalg.solve(centered.T @ (weights[:,None] * centered) + 100*np.eye(design.shape[1]),
                              centered.T @ (weights * target))
    np.testing.assert_allclose(model['coef'], direct, rtol=1e-6, atol=1e-9)
    changed = fit_basis({'kind':'linear','alpha':100}, X, y + era * 100, era, columns)
    np.testing.assert_allclose(model['coef'], changed['coef'], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize('kind', ['linear', 'polynomial', 'quadratic', 'gam', 'residual'])
def test_basis_roundtrip_and_row_permutation(kind):
    X, y, era, columns = fixture_arrays()
    model = fit_basis({'kind':kind,'alpha':100,'projections':3,'rounds':3}, X, y, era, columns, threads=1)
    decoded = rt.load_basis(to_bytes(model))
    actual = rt.basis_forward(decoded, X)
    np.testing.assert_allclose(actual, rt.basis_forward(model, X), atol=1e-12)
    order = np.random.default_rng(9).permutation(len(X))
    np.testing.assert_allclose(rt.basis_forward(decoded, X[order]), actual[order], atol=1e-12)
    assert np.isfinite(actual).all()


def test_smooth_tables_obey_the_registered_penalty():
    X, y, era, columns = fixture_arrays()
    ordinary = fit_basis({'kind':'gam','alpha':10}, X, y, era, columns)
    smooth = fit_basis({'kind':'gam','alpha':10,'smooth':100000}, X, y, era, columns)
    roughness = lambda m: np.sum(np.diff(m['coef'][:,:10], n=2, axis=1)**2)
    assert roughness(smooth) < roughness(ordinary)


@pytest.mark.parametrize('kind', ['linear','polynomial','gam','quadratic','residual','net'])
def test_basis_pickle_is_portable(kind, live_frame, data_dir, tmp_path):
    clean = Path(__file__).resolve().parents[1] / 'artifacts/pklcheck_venv/bin/python'
    if not clean.exists():
        pytest.skip('clean venv missing')
    columns = [c for c in live_frame if c.startswith('feature_')]
    X = live_frame[columns].to_numpy(dtype=float, copy=True)
    X[X<0] = np.nan
    rng = np.random.default_rng(4)
    spec = {'kind': 'linear' if kind=='net' else kind, 'alpha':100, 'projections':4, 'rounds':3}
    model = fit_basis(spec, X, rng.normal(size=len(X)), np.repeat(np.arange(3),30), columns, threads=1)
    if kind=='net':
        n = len(model['coef'])
        model.update(kind='net', x_mean=np.zeros(n), x_std=np.ones(n),
                     W0=rng.normal(0,.01,(n,8)), b0=np.zeros(8), W1=rng.normal(0,.01,(8,4)), b1=np.zeros(4),
                     W2=rng.normal(0,.01,(4,1)), b2=np.zeros(1))
    blob = to_bytes(model)
    b = {'version':'test', 'run_id':'basis-test', 'feature_cols':columns, 'missing_cols':[],
         'members':[{'name':'candidate','family':'basis','input':'nan','feature_cols':columns,'blobs':[blob]}],
         'weights':{'candidate':1.0},'p_b':0.0,'p_c':0.0,'core_proxy_cols':[]}
    predict = rt.make_predict(b)
    shuffled = live_frame.sample(axis=1, frac=1, random_state=2)
    pd.testing.assert_frame_equal(predict(live_frame), predict(shuffled))
    path = tmp_path / 'basis.pkl'
    bundle.write_pickle(b,path)
    result = pkl_check.run_check(path,python=clean,data_dir=data_dir,eras=0)
    assert result['passed'], pkl_check.format_report(result)


def research_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/research_run.py'
    spec = importlib.util.spec_from_file_location('research_test_runner',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_research_reader_excludes_the_previous_audit(tmp_path, monkeypatch):
    import yaml

    runner = research_module()
    monkeypatch.setattr(runner,'REPO',tmp_path)
    monkeypatch.setattr(runner,'__file__',str(tmp_path/'scripts/research_run.py'))
    monkeypatch.setattr(runner,'sha',lambda p:'fixed')
    config = {'name':'test','train_min_era':2800,'excluded_from_research':[5925,6524]}
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    calls=[]
    def read(path, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame({'exped':['exped_4000','exped_5000']})
    monkeypatch.setattr(runner.pd,'read_parquet',read)
    runner.load(path)
    assert calls==[{'filters':[('exped','<','exped_5925')]}]
    monkeypatch.setattr(runner.pd,'read_parquet',lambda *a,**k:pd.DataFrame({'exped':['exped_5945']}))
    with pytest.raises(RuntimeError,match='old audit reached'):
        runner.load(path)
