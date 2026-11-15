#!/usr/bin/env python3
"""Source-bound support-only repair of the observable-state experiment."""
import argparse,hashlib,json,logging
import yaml
import context as old
from stealth.conditional_supported import predict

c=old.c;REPO=old.REPO


def load():
    root,parent=old.load();source_run=c.RUN;source_plan=json.loads((source_run/'plan.json').read_text())
    c.CFG=yaml.safe_load((REPO/'configs/context-supported.yaml').read_text());c.RUN=REPO/'artifacts'/c.CFG['name'];c.RUN.mkdir(parents=True,exist_ok=True)
    plan=dict(protocol='state-effective-support-v1',config=c.CFG,data_sha256=source_plan['data_sha256'],
              source_plan_sha256=c.r.sha(source_run/'plan.json'),source_results_sha256=c.r.sha(source_run/'results.json'),
              code_sha256={f:c.r.sha(REPO/f) for f in ['scripts/context_supported.py','stealth/conditional_supported.py','docs/CONTEXT-15-REPAIR.md']})
    c.PLAN=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();plan['plan_sha256']=c.PLAN
    path=c.RUN/'plan.json'
    if path.exists() and json.loads(path.read_text())['plan_sha256']!=c.PLAN:raise RuntimeError('support experiment changed')
    if not path.exists():c.r.write_json(path,plan)
    old.copy_source(root,parent)
    return root


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['init','run','report'],required=True);args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    old.predict=predict;root=load()
    if args.stage=='run':old.run(root)
    elif args.stage=='report':
        doc=c.checked_json(c.RUN/'results.json');out=REPO/'handoffs/results';c.r.write_json(out/'context15r-metrics.json',doc)
        lines=['# Support-constrained state conditioning','',doc['label'],'',
               '| interval | policy | FIT | proxy score | historical gates |','|---|---|---|---|---|']
        for phase,rows in doc['comparisons'].items():
            for row in rows:lines.append(f'| {phase} | {row["name"]} | {row["metrics"]["fit"]:.8f} | {row["metrics"]["score"]:.8f} | {row["passes_historical_gates"]} |')
        lines+=['','Native models and previous gates are unchanged. No promotion or platform writes.']
        (out/'context15r-results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
