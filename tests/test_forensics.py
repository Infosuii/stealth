import importlib.util,json,sqlite3
from pathlib import Path
import numpy as np
import pytest


def runner():
    p=Path(__file__).resolve().parents[1]/'scripts/diagnose_models.py'
    spec=importlib.util.spec_from_file_location('forensic_test',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_rank_ensemble_is_invariant_to_each_models_positive_affine_scale():
    r=runner();era=np.repeat([1,2],5)
    p=np.array([[1,2,2,4,5,8,7,6,5,4],[5,4,3,2,1,5,6,7,8,9]],dtype=float)
    original=r.rank_ensemble(p,era);changed=p.copy();changed[0]=100*p[0]+17
    np.testing.assert_allclose(r.rank_ensemble(changed,era),original,atol=1e-15)
    assert not np.allclose(p.mean(axis=0),changed.mean(axis=0))
    changed[0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):r.rank_ensemble(changed,era)


def test_count_reader_uses_trial_number_and_preserves_database(tmp_path):
    r=runner();p=tmp_path/'search.sqlite'
    with sqlite3.connect(p) as db:
        db.execute('create table trials(trial_id integer, number integer)')
        db.execute('create table trial_user_attributes(trial_id integer,key text,value_json text)')
        db.execute('insert into trials values(91,7)')
        db.execute('insert into trial_user_attributes values(?,?,?)',(91,'counts',json.dumps([1,3,20])))
    before=r.sha(p);counts,digest=r.inner_counts(tmp_path,7)
    assert counts==[1,3,20] and digest==before and r.sha(p)==before
    with pytest.raises(RuntimeError,match='missing'):r.inner_counts(tmp_path,91)
