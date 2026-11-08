# Verification

Verified locally on 8 November 2026.

## Executed checks

- The default command actually generated 2,000 distinct complete structures, 100 distinct final topologies and 177 retained parent topologies from 500 attempts.
- The four exported arrays (`adjacency`, `offsets`, `raw_offsets`, `positions`) match our archived own-data 2,000-structure export exactly, element by element. This is a regeneration of that cohort, not an additional independent cohort or a match to the authors' historical final dataset.
- All 12 selected upstream function AST bodies match the recorded original revision. The five upstream CSV assets and MIT license are retained.
- The generator checks active-graph connectivity, boundary spanning, parent union identity, coordinate conversion, offset directions/bounds, finite coordinates, positive beam lengths, minimum active degree and duplicate complete structures.
- A copied package was run from a temporary directory outside the research workspace with a different seed, 80 parent attempts, four selected pairs and three geometry variants. It generated 12 checked structures successfully.
- Re-running against an existing output directory was rejected before any data changed.
- No pretrained model was loaded, no FE calculation was run and no final published sample database was required.

`verification.json` contains the machine-readable result. Package file hashes are in `MANIFEST.sha256`; the source-function hashes are in `upstream_functions.json`.

## Boundaries of these checks

These are generation, provenance and portability checks. Graph/geometry acceptance does not prove FE stability, physical strength, manufacturability, or neural reconstruction performance. Intersecting beam interiors remain separate unless the explicit graph supplies a joint. Very short or long nonzero beams are not silently removed by an additional manufacturing rule.

The optional GitHub Pages Markdown configuration and its local image links are included. Deployment to GitHub and the final hosted theme rendering have not been performed here.

## Own-data reference pipeline

The new datasets/own_10000 cohort contains 10,000 distinct full structures and 499 distinct topologies. All 10,000 FE labelling attempts passed. Sample IDs align across geometry, labels, split metadata and CSV. Effective matrices are symmetric positive definite; E22 values recomputed from saved matrices agree with the CSV. The saved random split is 8,000/1,000/1,000. No full input duplicates the earlier 2,000-row example. Dataset verification is saved alongside the arrays.

The VAE class AST matches the upstream class definition. Twenty hard-decoding coordinate conversions agree with the upstream utilities; E22 gradients through soft decoder/encoder/predictor are finite and nonzero on three fixed nonzero latent probes. The exactly zero code in an untrained zero-bias ReLU network is stationary, so optimisation starts use encoder means rather than that artificial probe. Training and actual search results are recorded separately after execution.
