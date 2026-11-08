# Own-data training and latent stiffness search

Fixed before the new run, 8 November 2026.

## Dataset and FE

Use this repository's generator: 1,500 parent attempts, 500 pairs, 20 geometry draws each, generation seed 202611081. Report actual unique topology count and generation failures. Preserve previous datasets. Label all accepted structures with the independent periodic linear Timoshenko FE solver, explicit graph joints, Es=1, nu=.3, density=.15. Failed labels remain failures and are excluded transparently; never fabricate labels.

Randomly split sample IDs 80/10/10 with seed 20261108 before training. Do not group by topology. Save indices, hashes and actual topology overlap. This is new-geometry sample evaluation, not a guarantee of unseen-topology generalization. Normalization uses valid training labels only. The test partition does not choose models or settings.

## Task 1

One scratch joint fit; no author checkpoint is read. Preserve the upstream 48-dimensional VAE class and its native shared-noise/logvar implementation. Xavier initialization, posterior-scale heads zero weights/bias -3. New head 48-128-128-9. Train for 200 epochs: 30 mean warmup, then native sampled reconstruction. Batch 128, Adam 3e-4 then 1e-4; gradient norm cap 10. Loss is adjacency BCE + 10 offset MSE + .1 training-std-normalized stiffness MSE + KL, annealed to 1e-4 over 50 sampled epochs. Select every ten epochs plus epoch 1 by validation exact mean topology rate, wrong edges, offset RMSE and normalized property MSE. Evaluate test after selection. Report connection accuracy, exact topologies, coordinates, nine-C errors and predicted-matrix validity. FE-check twelve preselected test reconstructions after selection.

## Task 2

Freeze that own-trained model. Select eight training source structures with different topologies by descending actual E22. Start from encoder means. Max 400 Adam steps, lr=1e-3, each latent component within .5 of its start; target E22=.15 is a fixed search reference, not an attainable-performance claim. Soft decoder -> encoder mean -> head -> E22. Stop a trajectory on non-positive-definite predicted C or nonfinite values. Hard-decoded FE at start, step 200 and selected endpoint; retain failures and distinguish predictor score from FE improvement. One fixed random latent displacement per start is a diagnostic reference, not an efficiency benchmark. At most 64 new post-training FE evaluations including reconstruction checks; generation labelling cost is accounted separately. No VAE retraining during search, stress constraints, GP search or hyperparameter sweeps. Report negative outcomes if any.

## Reporting

Commit executable code, dataset provenance, concise results, figures and one own-trained selected checkpoint. Keep logs/checkpoint intermediates local. Do not add upload instructions or dataset-sufficiency discussion to the public pages. Do not claim a global optimum, faster convergence, unseen-topology recovery or original author-FE equivalence.
