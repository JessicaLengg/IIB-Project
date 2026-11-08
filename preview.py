"""Draw four fixed octant examples; no trained model is involved."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, default=Path('examples.png'))
    args = parser.parse_args()
    with np.load(args.dataset) as data:
        adjacency, positions = data['adjacency'], data['positions']
    _, first = np.unique(adjacency.reshape(len(adjacency), -1), axis=0, return_index=True)
    ids = sorted(first)[:4]
    fig = plt.figure(figsize=(11, 3.2), layout='constrained')
    for column, i in enumerate(ids, 1):
        ax = fig.add_subplot(1, len(ids), column, projection='3d')
        for u, v in zip(*np.nonzero(np.triu(adjacency[i], 1))):
            ax.plot(*positions[i, [u, v]].T, color='#256482', linewidth=1.8)
        active = adjacency[i].sum(0) > 0
        ax.scatter(*positions[i, active].T, color='#dc7547', s=9)
        ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), box_aspect=(1, 1, 1), title=f'Own sample {i}',
               xlabel='x', ylabel='y', zlabel='z')
        ax.tick_params(labelsize=7)
    fig.suptitle('First four distinct generated octant topologies', fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180); plt.close(fig)


if __name__ == '__main__': main()
