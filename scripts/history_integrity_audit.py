#!/usr/bin/env python3
'Audit completed history18 native checkpoints and training boundaries.'
import sys,json,hashlib
from pathlib import Path
import numpy as np
import history_context as h
h.load()
s=h.s
records=[];groups=[]
for block in s.blocks(s.CFG['intervals'],100,40):
 for recipe in s.CFG['cpu_recipes']+s.CFG['gpu_recipes']:
  d=s.RUN/block['name']/recipe['name']
  group=s.native_cached(d/'done.json');groups.append(group)
  paths=[d/'training.json'] if recipe['name'].startswith('cpu') else [d/f'training{seed}.json' for seed in recipe['seeds']]
  for p in paths:
   r=json.loads(p.read_text())
   floor=max(recipe['floor'],block['train_end']-recipe['history']) if recipe['history'] else recipe['floor']
   assert r['train_min']==floor,(p,r['train_min'],floor)
   assert r['train_max']<block['test_start']-40,(p,'future fit')
   assert r['inner_valid_min']-r['inner_fit_max']>40,(p,'inner gap')
   assert r['teacher_manifest_sha256']==s.TEACHER_SHA
   if recipe['name'].startswith('gpu'):
    assert r['native_task_type']=='GPU'
    assert r['inner_cap']==2048
    assert r['curve'][-1]['count']==2048
   records.append(dict(block=block['name'],recipe=recipe['name'],copied='copied_from' in group,**r))
assert len(groups)==234
assert len(records)==351
new=[r for r in records if not r['copied']]
gpu=[r for r in new if r['recipe'].startswith('gpu')]
doc=dict(plan_sha256=s.PLAN,groups=len(groups),native_models=len(records),copied_groups=sum('copied_from' in r for r in groups),new_native_models=len(new),new_gpu_seed_models=len(gpu),full_inner_gpu_iterations=sum(r['inner_cap'] for r in gpu),all_boundaries_passed=True,actual_serialization_replay_max_error=max(r['serialized_replay_max_error'] for r in groups),inner_gpu_diagnostics={})
for name in ['gpu_expanding400','gpu_expanding3']:
 rr=[r for r in gpu if r['recipe']==name]
 gains=[max(v['score'] for v in r['curve'])-r['curve'][-1]['score'] for r in rr]
 doc['inner_gpu_diagnostics'][name]=dict(fits=len(rr),selected_count_min=min(r['n_iter'] for r in rr),selected_count_median=float(np.median([r['n_iter'] for r in rr])),selected_count_max=max(r['n_iter'] for r in rr),cap_selected=sum(r['n_iter']==2048 for r in rr),endpoint_worse_than_best=sum(v>1e-12 for v in gains),median_endpoint_proxy_loss=float(np.median(gains)))
p=s.RUN/'integrity-audit.json';p.write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps(doc,indent=2))
