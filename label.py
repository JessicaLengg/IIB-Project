"""Label generated geometry using independent periodic Timoshenko FE."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from fe import analytic_checks, homogenize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args(); folder = args.dataset.parent
    target = folder / 'labels.npz'
    if target.exists(): raise FileExistsError('Preserve existing FE labels.')
    with np.load(args.dataset) as source: data = {k: source[k].copy() for k in source.files}
    n = len(data['adjacency']); c = np.full((n, 9), np.nan); matrix = np.full((n, 6, 6), np.nan)
    valid = np.zeros(n, bool); rows = []; started = time.perf_counter()
    checks = analytic_checks()
    # All splits are fixed before fitting or inspecting reconstruction/search outcomes.
    order = np.random.default_rng(20261108).permutation(n)
    subsets = np.full(n, 'train', dtype='<U10')
    subsets[order[int(.8*n):int(.9*n)]] = 'validation'; subsets[order[int(.9*n):]] = 'test'
    np.savez_compressed(folder / 'splits.npz', subset=subsets, sample_ids=data['sample_ids'], seed=20261108)
    with threadpool_limits(limits=2), (folder / 'FE_journal.jsonl').open('w') as journal:
        for i, (a, p) in enumerate(zip(data['adjacency'], data['positions'])):
            row = {'sample_id':i, 'status':'failed', 'reason':''}
            try:
                values, info = homogenize(a, p, weld_intersections=False)
                if info['minimum_stiffness_eigenvalue'] <= 1e-10: raise ValueError('Non-positive effective stiffness')
                c[i] = values; matrix[i] = info['c_matrix']; valid[i] = True
                row.update(status='passed', E22=float(1/np.linalg.inv(matrix[i])[1,1]),
                           residual=info['equilibrium_relative_residual'])
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error: row['reason'] = str(error)
            rows.append(row); journal.write(json.dumps(row)+'\n'); journal.flush()
            if (i+1) % 500 == 0: print(f'FE {i+1}/{n}; passed {valid[:i+1].sum()}', flush=True)
    np.savez_compressed(target, stiffness=c, c_matrix=matrix, valid_FE=valid, sample_ids=data['sample_ids'])
    with (folder/'FE_results.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=['sample_id','status','reason','E22','residual'], lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    summary={'structures':n,'passed':int(valid.sum()),'failed':int((~valid).sum()),'analytic_checks':checks,
             'weld_intersections':False,'Es':1.,'nu':.3,'density':.15,'seconds':time.perf_counter()-started,
             'generation_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
             'solver_sha256':hashlib.sha256(Path(__file__).with_name('fe.py').read_bytes()).hexdigest()}
    (folder/'FE_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(summary,flush=True)


if __name__ == '__main__': main()
