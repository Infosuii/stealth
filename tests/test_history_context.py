import importlib.util
from pathlib import Path
import numpy as np


def test_history_filter_removes_only_declared_earlier_rows(monkeypatch):
    p=Path(__file__).resolve().parents[1]/'scripts/history_context.py';monkeypatch.syspath_prepend(str(p.parent))
    spec=importlib.util.spec_from_file_location('history_context_test',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    era=np.arange(3,1200);X=np.column_stack([era,era+1]);y=era*.1;teacher=era*.2
    x,t,e,b=module.filter_rows(dict(floor=400),X,y,era,teacher)
    np.testing.assert_array_equal(e,np.arange(400,1200));np.testing.assert_allclose(t,e*.1);np.testing.assert_allclose(b,e*.2)
    np.testing.assert_array_equal(x[:,0],e)
    for original,filtered in zip((X,y,era,teacher),module.filter_rows(dict(floor=3),X,y,era,teacher)):np.testing.assert_array_equal(original,filtered)
