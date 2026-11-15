#!/usr/bin/env python3
'Post-result descriptive history contrasts; no additional promotion tests.'
import sys,json
from pathlib import Path
import history_context as h
h.load();s=h.s
base=s.predictions('cpu_expanding400');frame=s.FRAME.loc[base.index]
X=frame[[x for x in frame if x.startswith('feature_')]].to_numpy(float)
cores=[s.core_mean(X),s.core_mean(X[:,::2]),s.core_mean(X[:,1::2])]
teacher=s.TEACHER.reindex(base.index).to_numpy()
p={r['name']:s.rank_by_era(s.predictions(r['name']).reindex(base.index).prediction,base.era) for r in s.CFG['cpu_recipes']+s.CFG['gpu_recipes']}
doc=dict(plan_sha256=s.PLAN,label='post-result descriptive matched-history contrasts; no new promotion tests',comparisons=[])
for w in s.CFG['intervals']:
 for a,b in [('cpu_expanding400','cpu_rolling'),('cpu_expanding3','cpu_expanding400'),('gpu_expanding400','gpu_rolling'),('gpu_expanding3','gpu_expanding400')]:
  row=dict(interval=w['name'],challenger=a,comparator=b,**s.evaluate(p[a],base,teacher,frame,cores,w,comparison=p[b]))
  doc['comparisons'].append(row)
  print(w['name'],a,b,'fit delta',row['paired']['fit']['diff_mean'],'p_pos',row['paired']['fit']['p_pos'],flush=True)
s.r.write_json(s.RUN/'matched-history-diagnostics.json',doc)
