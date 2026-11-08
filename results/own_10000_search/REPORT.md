# Own-model latent stiffness search

Freeze the VAE and stiffness head trained from scratch on our own FE-labelled structures. Eight training-source topologies initialize the fixed bounded searches. No test examples choose starts, trajectories or settings.

| Source ID | Original FE E22 | Decoded-start FE E22 | Selected-endpoint FE E22 | Gain versus original (%) | Selected step | Stop reason |
| --- | --- | --- | --- | --- | --- | --- |
| 5466 | 0.0582523 | 0.05833460652592833 | 0.05153138096471951 | -11.537654258360098 | 259 | non_positive_surrogate_C |
| 7465 | 0.0533709 | 0.052986582465382744 | 0.049275682109999185 | -7.673120695522595 | 400 | budget_completed |
| 4782 | 0.0512747 | 0.05135073020210644 | None | None | 400 | budget_completed |
| 9221 | 0.051064 | 0.050897379715514966 | 0.05398881328947843 | 5.727676546705118 | 400 | budget_completed |
| 6750 | 0.0484508 | 0.04872777935533999 | 0.051740910889763576 | 6.790525328044916 | 400 | budget_completed |
| 8044 | 0.048264 | 0.047451511589673964 | 0.053717386251906016 | 11.299151325841162 | 400 | budget_completed |
| 3765 | 0.0476229 | 0.046837834146340775 | 0.05291356537372639 | 11.109494964304002 | 400 | budget_completed |
| 8222 | 0.0475556 | 0.047108536967189404 | 0.05387944629403352 | 13.297788168486392 | 400 | budget_completed |

Endpoints after at least one latent update: 7/8. Those exceeding their own original FE source: 5/8. Step-zero fallbacks are retained but are not optimised results.

New post-training search FE attempts: 32; earlier reconstructed-test FE attempts: 12; combined cap: 64. The full dataset FE labelling is a separate upfront cost. Exact repeated decoded geometries reuse the search cache.

The endpoint is selected by surrogate prediction before its FE check. Midpoint checks do not update the optimisation path. Both gains versus the original structure and versus the decoded start are saved; recovering a starting reconstruction loss is not necessarily a better design than the original.

The random displacement is one fixed diagnostic per source, not an equally tuned optimisation benchmark. No faster-convergence, global-optimality, novel-topology, stress-limit or manufacturing claim is made. Invalid predicted stiffness and actual FE failures remain visible. Negative results are valid outcomes. The saved own-trained checkpoint hash is unchanged.

## Verification and reference comparisons

The largest relative improvement is source 8222: 0.04755560 to 0.05387945 (+13.2978%). The highest absolute endpoint is source 9221: 0.05398881 (+5.7277% versus its own source). Neither exceeds the best original training source, 0.05825233. These are different comparisons and must not be conflated.

The largest-relative-gain source and decoded endpoint were independently FE-replayed from the saved own-model latent. Its topology is unchanged; geometry changes. Search performed 32 new geometry checks, including one disconnected graph rejected before solving, hence 31 actual FE solves. Including twelve reconstruction solves and two final replay solves gives 45 actual post-training solves (46 conservative charged checks), below the cap of 64. Dataset labelling cost is separate.
