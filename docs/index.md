---
layout: default
title: Truss Data Generation
---

# Truss Data Generation

**Own structure samples, generated with an adapted public-code workflow.**

This small project makes the generation route in Zheng et al.'s public code easy to run and inspect: elementary seed graphs, stochastic topology changes, parent superposition, constrained node perturbations and checked exports. It is a generation-only package with no neural-network weights or final published dataset dependency.

![Workflow](assets/workflow.svg)

## A checked 2,000-structure example

| Item | Actual supplied example |
| --- | ---: |
| Raw topology attempts | 500 |
| Unique retained parents | 177 |
| Selected parent pairs | 100 |
| Geometry draws per pair | 20 |
| Distinct complete structures | 2,000 |
| Distinct final topologies | 100 |

![First four distinct octant topologies](assets/examples.png)

Each structure has a 27-by-27 adjacency matrix and 27 three-dimensional candidate node positions within one octant. Vertices are fixed; edge, face and interior nodes obey their respective movement constraints. The generator checks active-graph connectivity, boundary spanning, coordinate conversion, nonzero beam lengths and provenance.

## Run the workflow

```bash
python -m pip install -r requirements.txt
python generate.py --output generated
python preview.py generated/structures.npz --output examples.png
```

The generator saves geometry, connectivity, source-parent indices, per-sample checks and a summary. Its default output is a new folder; existing results are preserved. [Open the repository for code, the sample dataset and full instructions]({{ site.github.repository_url }}).

## Adapted from the authors' code

Selected source functions retain their original bodies. The small runner corrects a parent-pair validity index, fixes random seeds and reduces the job size. It follows the public code's **pair first, then independently perturb geometry** route. Explicit graph connections are preserved; crossing beam interiors are not automatically welded. These choices and the global export coordinate-frame exchange are documented in the repository.

The package is not a literal reproduction of every step in the paper's prose or of its full historical dataset. It contains no FE labels, training or optimisation results. Those belong to later stages of the research pipeline.

## Reference

Zheng, L., Karapiperis, K., Kumar, S. & Kochmann, D. M. *Unifying the design space and optimizing linear and nonlinear truss metamaterials by generative modeling*. Nature Communications **14**, 7563 (2023).

[Paper](https://doi.org/10.1038/s41467-023-42068-x) · [Original code](https://github.com/li-zhengz/UnifyingTrussDesignSpace)

Upstream MIT copyright is retained. This adaptation is not the authors' official repository.
