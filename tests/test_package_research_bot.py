import importlib.util
from pathlib import Path

import pytest


def load(monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/package_research_bot.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec=importlib.util.spec_from_file_location('package_research_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_only_registered_policies_can_be_packaged(monkeypatch):
    m=load(monkeypatch)
    assert m.components('cpu_expanding3','history18')==['cpu_expanding3']
    assert m.components('gpu_core3_blend','utility19')==['gpu_core3','cpu_raw400']
    for name,run in [('cpu_expanding3_blend','history18'),('gpu_core3_rawcount','utility19')]:
        with pytest.raises(ValueError):m.components(name,run)


def test_export_preserves_equal_gaussian_member_and_seed_rules(monkeypatch):
    m=load(monkeypatch)
    members=[dict(name='gpu_core3',arm='cat',blob_rule='gaussian_mean'),dict(name='cpu_raw400',arm='lgbm')]
    p=m.payload(members,['feature_1'],'fixed-plan')
    assert p['members']==members
    assert p['weights']=={'gpu_core3':.5,'cpu_raw400':.5}
    assert p['p_b']==p['p_c']==0
    assert p['source_run_fingerprint']=='fixed-plan'
