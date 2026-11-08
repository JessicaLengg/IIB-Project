# Source and adaptations

Upstream: https://github.com/li-zhengz/UnifyingTrussDesignSpace

Revision: `8bb0aa401d6b5c2560e8d2121f11e2e0a7c26f77`.

The four elementary graph CSVs come from `dataGeneration/baseLattices`: BCC `A_S.csv`, SC `A.csv`, SC `A_S.csv`, and the `fcc/A_S.csv` octet graph. `nodes_init.csv` is the original `nodesInit.csv`.

## Retained functions

| Original file | Functions copied to upstream.py |
| --- | --- |
| dataGeneration/generation.py | moveTop, check_bc_connected, check_connection |
| dataGeneration/moveNodes.py | str2stru, randrange |
| dataGeneration/main.py | get_dof_idx, randomPtb, ptbNodes, bary_coor1D, bary_coor2D, bary_coor3D, ptb2nodes |

Only these function definitions and the required constants are used. Python's AST unparser removes comments and reformats the source; function AST bodies are unchanged. Hashes are recorded in `upstream_functions.json`. This avoids importing the original scripts, which execute large jobs and load other files at import time.

## Runner changes

`generate.py` replaces the original top-level orchestration with a small command-line runner. It uses 500 default parent attempts, serial dispatch, explicit NumPy/Python seeds and the original topology-deduplication order. The original 1,000 unused NumPy draws are preserved for RNG continuity with the archived small-data run.

The release's parent-pair validity loop assigns both indices from `all_combinations[k][0]`. The runner checks `parents[i] | parents[j]`, using both indices. Thus its acceptance test corresponds to the pair actually used for generation. No upstream function body is patched for this correction.

Geometry is independently perturbed after pair superposition. The exported Cartesian coordinates and natural-offset components undergo the same global x/y exchange used in our prior ML dataset. Both coordinate conversion and parent unions are checked for every exported row.

The runner does not read the authors' final dataset or any neural-network weights. It does not use reconstruction scores to reject samples. It does not run FE, generate stiffness labels, weld beam-interior intersections, filter by stiffness, or impose an extra maximum perturbed-beam-length rule.

## Scope

New sample instances are generated from the authors' elementary seed graphs and adapted public-code workflow. This is not a new generation algorithm and does not claim exact historical dataset reproduction. The paper's prose and the public code differ in some ordering and intersection details; those differences remain explicit rather than being combined silently.
