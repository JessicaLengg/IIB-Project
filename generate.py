"""Small, reproducible runner for the authors' public generation functions."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random
import time
from importlib.metadata import version

import networkx as nx
import numpy as np
import upstream as up

FAMILIES = ['A1', 'A2', 'B2', 'C2']
UPPER = np.triu_indices(27)
EDGES = np.triu_indices(27, 1)
AXES = [1, 0, 2]  # Explicit global x/y exchange used by our existing ML exports.


def connected(adjacency):
    active = np.flatnonzero(adjacency.sum(0))
    return bool(len(active) and nx.is_connected(nx.from_numpy_array(adjacency[np.ix_(active, active)])))


def symmetric(adjacency):
    result = (adjacency + adjacency.swapaxes(-1, -2)) > 0
    result[..., np.arange(27), np.arange(27)] = False
    return result.astype(np.uint8)


def parent_library(attempts, seed):
    np.random.seed(seed); random.seed(seed)
    seeds = [np.loadtxt(up.DATA / name, delimiter=',')
             for name in ['bcc.csv', 'sc.csv', 'sc_subdivided.csv', 'octet.csv']]
    # Preserve the release's otherwise unused 1,000 draws for RNG continuity.
    for _ in range(1000): np.random.choice(FAMILIES, 2)
    raw, families = [], []
    for _ in range(attempts):
        family = random.choices(FAMILIES, k=1)[0]
        index, _, _ = up.str2stru(family)
        raw.append(up.moveTop(seeds[index], up.nodesInit, index)); families.append(family)
    raw = np.asarray(raw)
    positions = np.tile(up.nodesInit, (attempts, 1))
    repaired, positions = up.check_connection(raw.copy().reshape(-1, 27), positions, attempts)
    repaired = symmetric(repaired.reshape(-1, 27, 27))
    connected_ids = [i for i, a in enumerate(repaired) if connected(a)]
    selected_positions = positions.reshape(-1, 27, 3)[connected_ids].reshape(-1, 3)
    boundary_failures = set(up.check_bc_connected(repaired[connected_ids].astype(float).reshape(-1, 27),
                                                selected_positions, len(connected_ids)).tolist())
    keep = [i for j, i in enumerate(connected_ids) if j not in boundary_failures]
    if not keep: raise ValueError('No valid parents; inspect seed and attempt count.')
    _, unique = np.unique(repaired[keep][:, UPPER[0], UPPER[1]], axis=0, return_index=True)
    ids = np.asarray(keep)[unique]
    return repaired[ids], ids, np.asarray(families)[ids], raw


def generate(parents, pairs, variants, seed):
    # Check the union of BOTH parents; the release's pair-check repeats index 1.
    valid = [(i, j) for i, j in itertools.combinations_with_replacement(range(len(parents)), 2)
             if connected(parents[i] | parents[j])]
    if pairs > len(valid): raise ValueError(f'Only {len(valid)} valid pairs; reduce --pairs or increase --attempts.')
    np.random.seed(seed + 1); random.seed(seed + 1)
    chosen = np.asarray(valid)[np.random.choice(len(valid), pairs, replace=False)]
    adjacency, offsets, positions = [], [], []
    seen, duplicate_draws = set(), 0
    for i, j in chosen:
        union = parents[i] | parents[j]
        for _ in range(variants):
            for attempt in range(100):
                offset = up.ptbNodes(union)
                key = hashlib.sha256(union.tobytes() + offset[:, AXES].tobytes()).hexdigest()
                if key not in seen:
                    seen.add(key)
                    break
                duplicate_draws += 1
            else:
                raise ValueError('Repeated geometry draws exhausted; reduce variants or change the seed.')
            points = up.nodesInit.copy() if not offset.any() else up.ptb2nodes(offset)
            adjacency.append(union.copy()); offsets.append(offset[:, AXES]); positions.append(points[:, AXES])
    return np.asarray(adjacency), np.asarray(offsets), np.asarray(positions), chosen, len(valid), duplicate_draws


def check(adjacency, offsets, positions, chosen, parents, variants):
    rows = []; keys = set()
    reference = up.nodesInit[:, AXES]
    for i, (a, x, p) in enumerate(zip(adjacency, offsets, positions)):
        pair_id = i // variants; parent1, parent2 = chosen[pair_id]
        np.testing.assert_array_equal(a, parents[parent1] | parents[parent2])
        np.testing.assert_allclose(up.ptb2nodes(x[:, AXES])[:, AXES], p, atol=1e-12)
        active = np.flatnonzero(a.sum(0)); degree = a.sum(0)[active]
        lengths = np.linalg.norm(p[EDGES[1]] - p[EDGES[0]], axis=1)[a[EDGES] > 0]
        boundary = bool(np.all(np.isclose(p[active].min(0), 0)) and np.all(np.isclose(p[active].max(0), 1)))
        key = hashlib.sha256(a.tobytes() + x.tobytes()).hexdigest()
        assert key not in keys, f'Duplicate structure {i}'; keys.add(key)
        assert connected(a) and boundary and degree.min() >= 2
        assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1 and abs(x).max() <= .5
        assert np.all(x[reference != .5] == 0) and lengths.min() > 1e-8
        rows.append(dict(sample_id=i, pair_id=pair_id, variant=i % variants, parent1=int(parent1), parent2=int(parent2),
                         topology_key=hashlib.sha256(a.tobytes()).hexdigest(), connected=True, boundary_span=boundary,
                         minimum_active_degree=int(degree.min()), shortest_beam=float(lengths.min()),
                         longest_beam=float(lengths.max()), representation_key=key))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('generated'))
    parser.add_argument('--attempts', type=int, default=500)
    parser.add_argument('--pairs', type=int, default=100)
    parser.add_argument('--variants', type=int, default=20)
    parser.add_argument('--seed', type=int, default=202611011)
    args = parser.parse_args()
    if min(args.attempts, args.pairs, args.variants) < 1: parser.error('Counts must be positive.')
    if args.output.exists(): parser.error('Output exists; use a fresh --output folder.')
    started = time.perf_counter()
    parents, parent_ids, families, raw = parent_library(args.attempts, args.seed)
    a, x, p, chosen, valid_pairs, duplicate_draws = generate(parents, args.pairs, args.variants, args.seed)
    rows = check(a, x, p, chosen, parents, args.variants)
    reference = up.nodesInit[:, AXES]; args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'structures.npz', adjacency=a, raw_offsets=x, positions=p,
                        offsets=x[:, reference == .5].astype(np.float32), reference_positions=reference,
                        sample_ids=np.arange(len(a)))
    np.savez_compressed(args.output / 'parents.npz', adjacency=parents, raw_attempt_id=parent_ids, seed_family=families)
    np.savez_compressed(args.output / 'raw_attempts.npz', adjacency=raw)
    np.savetxt(args.output / 'selected_pairs.csv', chosen, delimiter=',', fmt='%d', header='parent1,parent2', comments='')
    with (args.output / 'provenance.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    summary = dict(structures=len(a), unique_structures=len(rows), unique_topologies=len({r['topology_key'] for r in rows}),
                   parent_topologies=len(parents), valid_parent_pairs=valid_pairs, topology_attempts=args.attempts,
                   selected_pairs=args.pairs, variants_per_pair=args.variants, seed=args.seed,
                   duplicate_geometry_draws_resampled=duplicate_draws,
                   all_checks_passed=True, labels_included=False, pretrained_weights_used=False,
                   seconds=round(time.perf_counter() - started, 3),
                   tested_versions={name: version(name) for name in ['numpy', 'scipy', 'networkx']})
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
