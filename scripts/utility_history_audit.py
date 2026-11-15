#!/usr/bin/env python3
"""Read-only verification of all utility19 histories and matched count donors."""
import json

import numpy as np
import utility_history as u


def main():
    u.load();s=u.s
    groups, fits, own_gpu = [], [], []
    for b in s.blocks(s.CFG['intervals'],100,40):
        for family in ('cpu','gpu'):
            for floor in (400,3):
                donor=s.RUN/b['name']/f'{family}_raw{floor}'
                for kind in ('raw','core','core_fixed'):
                    name=f'{family}_{"core" if kind=="core_fixed" else kind}{floor}'
                    if kind=='core_fixed':name+='_rawcount'
                    d=s.RUN/b['name']/name
                    group=s.native_cached(d/'done.json');groups.append(group)
                    for seed in ([17,29] if family=='gpu' else [17]):
                        suffix=str(seed) if family=='gpu' else ''
                        info=json.loads((d/f'training{suffix}.json').read_text())
                        assert info['train_min']==floor,(d,'floor')
                        assert info['train_max']==b['train_end']-1,(d,'outer gap')
                        assert info['inner_valid_min']-info['inner_fit_max']>40,(d,'inner gap')
                        assert info['teacher_manifest_sha256']==s.TEACHER_SHA,(d,'reference provenance')
                        if kind=='core_fixed':
                            raw=json.loads((donor/f'training{suffix}.json').read_text())
                            assert info['n_iter']==raw['n_iter'],(d,'count')
                            assert info['count_source_done_sha256']==s.r.sha(donor/'done.json'),(d,'count donor')
                        if kind!='raw':
                            assert info['target']=='core',(d,'supervision')
                        if family=='gpu' and kind=='core':
                            assert info['native_task_type']=='GPU'
                            assert info['inner_cap']==2048 and info['curve'][-1]['count']==2048
                            own_gpu.append(info)
                        fits.append(info)
    assert len(groups)==468 and len(fits)==702
    replay=s.checked(s.RUN/'cpu-own-count-replay.json')
    assert replay['replays']==6 and replay['max_error']==0
    doc=dict(plan_sha256=s.PLAN,all_boundaries_and_count_donors_passed=True,
             checkpoint_groups=len(groups),native_models=len(fits),
             copied_groups=sum('copied_from' in g for g in groups),
             serialization_max_error=max(g['serialized_replay_max_error'] for g in groups),
             own_cpu_count_replay=replay,new_gpu_inner_iterations=sum(i['inner_cap'] for i in own_gpu),
             gpu_inner_diagnostics=dict(fits=len(own_gpu),selected_cap=sum(i['n_iter']==2048 for i in own_gpu),
                endpoint_worse=sum(max(v['score'] for v in i['curve'])>i['curve'][-1]['score']+1e-12 for i in own_gpu),
                selected_count_median=float(np.median([i['n_iter'] for i in own_gpu]))))
    s.r.write_json(s.RUN/'integrity-audit.json',doc)
    print(json.dumps(doc,indent=2))


if __name__=='__main__':main()
