import json,sqlite3
import numpy as np
import pytest
from stealth.campaign_models import prior_proposals,initial_proposals
from stealth import pool_models


def test_prior_population_is_fixed_before_any_labels_or_scores():
    a=prior_proposals(5,17);b=prior_proposals(5,17);other=prior_proposals(5,29)
    assert a==b and a[:2]==initial_proposals() and a[2:]!=other[2:]
    assert len(a)==5 and all(set(p)==set(a[0]) for p in a)


def test_source_study_copy_is_isolated_and_partition_checked(tmp_path,monkeypatch):
    monkeypatch.setattr(pool_models,'REPO',tmp_path)
    root=tmp_path/'artifacts/original';src=root/'early/trees/search.sqlite';src.parent.mkdir(parents=True)
    with sqlite3.connect(src) as db:db.execute('create table evidence(value integer)');db.execute('insert into evidence values(7)')
    inner={'windows':3}
    config={'seed':17,'inner':inner,'train_min_era':2800,'discovery':[{'name':'early','train_end':3960}],
            'candidates':[{'name':'trees','kind':'tuned_trees','trials':2,'cap':8,'patience':3}]}
    (root/'plan.json').write_text(json.dumps({'config':config,'data_sha256':'data','plan_sha256':'source'}))
    destination=tmp_path/'artifacts/new/early/pool';destination.mkdir(parents=True)
    spec={'reuse_search_from':'original','trials':2,'cap':8,'patience':3}
    before=pool_models.sha(src)
    proof=pool_models.copy_source(spec,destination,np.arange(2800,3960),inner,17,'data')
    assert proof['source_sha256']==before and pool_models.sha(src)==before
    with sqlite3.connect(destination/'search.sqlite') as db:db.execute('insert into evidence values(9)')
    with sqlite3.connect(src) as db:assert db.execute('select value from evidence').fetchall()==[(7,)]
    with pytest.raises(RuntimeError,match='data/seed'):pool_models.copy_source(spec,destination,np.arange(2800,3960),inner,29,'data')
