# Truss Data Generation

A small, reproducible **data-generation-only** workflow adapted from Zheng et al.'s public [UnifyingTrussDesignSpace code](https://github.com/li-zhengz/UnifyingTrussDesignSpace). It generates new octant graphs and constrained node positions from elementary lattice seeds. No published final dataset, pretrained weights, PyTorch or network access is needed to run it.

![Generation workflow](docs/assets/workflow.svg)

![Four generated octant examples](docs/assets/examples.png)

## What is included?

- `generate.py`: one readable entry point for generation, checks and export.
- `upstream.py`: 12 selected upstream functions, with their original AST bodies retained; source attribution and MIT license are included.
- `data/`: the four upstream elementary adjacency graphs and the 27 reference-node coordinates.
- `preview.py`: a small plotting helper.
- `example_data/`: an actually generated, checked 2,000-structure example with provenance.
- `docs/`: a GitHub Pages landing page for this generation workflow.

FE labels, model training, checkpoint comparisons and latent optimisation are outside this package. The example contains **geometry and connectivity, not stiffness labels**.

## Quick start

Use Python 3.10 or newer, then run from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate.py --output generated
python preview.py generated/structures.npz --output examples.png
```

The defaults request 500 topology attempts, 100 selected parent pairs and 20 independent geometry draws per pair. The supplied example has **2,000 distinct structures, 100 distinct final topologies and 177 retained parent topologies**. A new output path is required; existing results are never overwritten.

The dependency versions used to verify the supplied example are in `requirements-tested.txt`.

## Generation steps

1. Load the upstream SC, subdivided SC, BCC and octet seed graphs on 27 candidate nodes in one octant.
2. Apply the upstream stochastic topology changes and repair rules. Retain connected parent graphs meeting the upstream boundary check, then remove duplicate topologies.
3. Enumerate unordered parent pairs, allowing self-pairs. Check the connectivity of the union of **both** parents and sample valid pairs without replacement.
4. Combine the two parent adjacencies with Boolean OR. For each selected pair, call the upstream node-perturbation function independently 20 times.
5. Verify connectivity of active nodes, boundary spanning, union provenance, unique complete structures, allowed offset directions, coordinate bounds and nonzero beam lengths. Save the arrays, seeds, parent IDs and per-sample checks.

Vertex positions are fixed. Edge nodes move only along their edge, face nodes within their face, and the body node in three dimensions. The representation keeps 27 candidate slots; inactive slots are not connected graph nodes. This exports the **octant**, not an already mirrored full-cell FE mesh. Beam interiors do not become joints merely by crossing.

## Output arrays

```python
import numpy as np

with np.load("example_data/structures.npz") as data:
    adjacency = data["adjacency"]  # (2000, 27, 27), symmetric, zero diagonal
    offsets = data["offsets"]      # (2000, 27), movable natural-coordinate features
    positions = data["positions"]  # (2000, 27, 3), Cartesian coordinates
```

`raw_offsets` stores `(N, 27, 3)` natural offsets; these are **not generally Cartesian displacements**. `reference_positions` stores the reference arrangement. The export applies one explicit global x/y exchange to both coordinates and offset components, matching our existing ML exports. It does not alter adjacency, lengths or connectivity. `generate.check` verifies the coordinate conversion using the inverse exchange and the original upstream mapping.

`parents.npz`, `raw_attempts.npz`, `selected_pairs.csv` and `provenance.csv` retain the generation history. `summary.json` reports actual unique counts, rather than assuming every selected pair produces a new topology.

## What was adapted?

This is a **code-derived, smaller-scale generation workflow**, not a recreation of the authors' complete historical dataset.

- The original pair-validation loop reads the first parent index twice. Our runner checks the actual union of the two different parent indices.
- We use fixed NumPy/Python seeds, serial execution and smaller configurable counts. The original release's 1,000 unused NumPy draws are retained to preserve RNG continuity with our archived run.
- We extract only the functions used by this route. Their bodies are unchanged; their provenance hashes are in `upstream_functions.json`.
- We add independent checks and provenance export. No VAE or stiffness predictor is used to select accepted examples.

The public code generates geometry **after pairing**. The paper's prose describes geometry perturbation within the preliminary library **before superposition**, and describes inserting joints at beam intersections. We follow the selected executable graph-generation route and preserve explicit graph connections. Therefore, the package should be cited as **adapted from the public code**, not as a literal reproduction of every paper-described step. No maximum perturbed-beam-length filter or intersection welding is silently added.

See [PROVENANCE.md](PROVENANCE.md) and [VERIFICATION.md](VERIFICATION.md).

## Source and citation

Li Zheng, Konstantinos Karapiperis, Siddhant Kumar and Dennis M. Kochmann. *Unifying the design space and optimizing linear and nonlinear truss metamaterials by generative modeling*. Nature Communications 14, 7563 (2023). [Paper](https://doi.org/10.1038/s41467-023-42068-x) · [Original code](https://github.com/li-zhengz/UnifyingTrussDesignSpace).

Upstream revision: `8bb0aa401d6b5c2560e8d2121f11e2e0a7c26f77`. Upstream MIT copyright and license are retained in [LICENSE](LICENSE). This adaptation is not the authors' official repository.
