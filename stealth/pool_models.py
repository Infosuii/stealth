"""Broad rank consensus and immutable source-study reuse."""
import hashlib,json,sqlite3
from pathlib import Path
import numpy as np
from stealth._runtime import average_blob_scores
from stealth.campaign_models import fit_trees

REPO=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def copy_source(spec,directory,era,inner,seed,data_sha):
    if not spec.get('reuse_search_from'):return None
    root=REPO/'artifacts'/spec['reuse_search_from'];name=directory.parent.name
    plan=json.loads((root/'plan.json').read_text())
    window=next((w for w in plan['config']['discovery'] if w['name']==name),None)
    if window is None:return None
    if plan['data_sha256']!=data_sha or plan['config']['seed']!=seed or plan['config']['inner']!=inner:
        raise RuntimeError('source-study data/seed/inner layout changed')
    if window['train_end']!=int(np.max(era))+1 or plan['config']['train_min_era']!=int(np.min(era)):
        raise RuntimeError('source-study fitting boundaries changed')
    old=next(s for s in plan['config']['candidates'] if s['kind']=='tuned_trees')
    if any(old[k]!=spec[k] for k in ('trials','cap','patience')):raise RuntimeError('source training budget changed')
    source=root/name/old['name']/'search.sqlite';fingerprint=sha(source)
    destination=directory/'search.sqlite';record=directory/'source-study.json'
    proof={'source_sha256':fingerprint,'source_plan_sha256':plan['plan_sha256'],'window':name}
    if record.exists() and json.loads(record.read_text())!=proof:raise RuntimeError('source study changed during reuse')
    if destination.exists() and not record.exists():raise RuntimeError('unverified partial source-study clone')
    if not destination.exists():
        with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as src,sqlite3.connect(destination) as dst:src.backup(dst)
        record.write_text(json.dumps(proof,indent=2)+'\n')
    if sha(source)!=fingerprint:raise RuntimeError('source study changed while copying')
    return proof


def fit_pool(spec,X,y,era,columns,inner,*,seed,threads,directory,data_sha):
    proof=copy_source(spec,directory,era,inner,seed,data_sha)
    models,info=fit_trees(spec,X,y,era,inner,seed=seed,threads=threads,storage=directory/'search.sqlite')
    if len(models)!=spec['trials']:raise RuntimeError('pool does not contain every registered completed model')
    info['source_study']=proof;info['sampler']=spec['sampler']
    return models,info


def predict_pool(models,X,era):
    values=np.array([m.predict(np.asarray(X,dtype=np.float32),num_threads=1) for m in models])
    result=np.empty(len(X))
    for e in np.unique(era):
        keep=np.asarray(era)==e
        result[keep]=average_blob_scores(list(values[:,keep]),'gaussian_mean')
    return result
