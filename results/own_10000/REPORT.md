# Own-data reconstruction and stiffness prediction

One scratch joint fit; selected epoch 160. No author checkpoint is read. The upstream VAE architecture and native sampler are retained, with our small stiffness head and loss.

| Partition | Mode | Exact topology | Matrix accuracy | Offset RMSE | Nine-C RMSE | Positive-definite predicted C |
| --- | --- | --- | --- | --- | --- | --- |
| train | mean | 8000/8000 | 100.0000% | 0.01252 | 0.0015967 | 7609/8000 |
| validation | mean | 1000/1000 | 100.0000% | 0.016681 | 0.0018252 | 943/1000 |
| test | mean | 1000/1000 | 100.0000% | 0.017005 | 0.0018197 | 944/1000 |
| train | native_sample | 7997/8000 | 99.9998% | 0.012777 | 0.0015967 | 7609/8000 |
| validation | native_sample | 999/1000 | 99.9978% | 0.016965 | 0.0018252 | 943/1000 |
| test | native_sample | 998/1000 | 99.9986% | 0.017229 | 0.0018197 | 944/1000 |

Test contains 438 distinct topologies; 1000/1000 test rows share their topology with training. This is random sample splitting, not a topology-disjoint benchmark. Test did not select checkpoints or hyperparameters.

Test median active coordinate MAE: 0.00299146; median maximum active coordinate component error: 0.0319849, relative to octant edge length 1. Exact connections do not imply exact positions.

Prespecified reconstructed-geometry FE checks: 12/12 passed. Median absolute E22 change among successful solves: 0.5451835228670876%. These twelve checks are not all-test FE verification.

Property R2/RMSE by component are in property_metrics.csv. One dataset and training seed do not establish broad generative validity. Native VAE sampling variation is not calibrated property uncertainty. FE stability is not manufacturing or strength validation. See EXPERIMENT.md for the fixed recipe.
