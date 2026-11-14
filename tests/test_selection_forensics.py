import importlib.util,json,sqlite3,sys
from pathlib import Path
import pytest


def module(monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/diagnose_selection.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec=importlib.util.spec_from_file_location('selection_forensic_test',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def test_stored_distributions_preserve_categories(monkeypatch):
    mod=module(monkeypatch)
    categorical=json.dumps({'name':'CategoricalDistribution','attributes':{'choices':[None,False,8]}})
    assert mod.decode_parameter(0.,categorical) is None
    assert mod.decode_parameter(1.,categorical) is False
    assert mod.decode_parameter(2.,categorical)==8
    assert mod.decode_parameter(4.,json.dumps({'name':'IntDistribution'}))==4
    assert mod.decode_parameter(.2,json.dumps({'name':'FloatDistribution'}))==.2
    with pytest.raises(ValueError):mod.decode_parameter(1.,json.dumps({'name':'Unsupported'}))


def test_trial_reader_keeps_parameters_counts_and_objective_without_writes(tmp_path,monkeypatch):
    mod=module(monkeypatch);path=tmp_path/'trial.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('create table trials(trial_id integer,number integer,state text)')
        db.execute('create table trial_params(trial_id integer,param_name text,param_value real,distribution_json text)')
        db.execute('create table trial_user_attributes(trial_id integer,key text,value_json text)')
        db.execute('create table trial_values(trial_id integer,objective integer,value real)')
        db.execute("insert into trials values(91,7,'COMPLETE')")
        db.execute('insert into trial_params values(?,?,?,?)',(91,'context',1.,json.dumps({'name':'CategoricalDistribution','attributes':{'choices':[False,True]}})))
        for key,value in [('counts',[1,3,20]),('scores',[.01,.02,.03])]:db.execute('insert into trial_user_attributes values(?,?,?)',(91,key,json.dumps(value)))
        db.execute('insert into trial_values values(91,0,.017)')
    before=mod.d.sha(path);trials,digest=mod.read_trials(path)
    assert trials==[{'number':7,'params':{'context':True},'counts':[1,3,20],'inner_scores':[.01,.02,.03],'objective':.017}]
    assert digest==before and mod.d.sha(path)==before
