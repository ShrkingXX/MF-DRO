V6 first pass, DISCARDED as a controlled comparison: run_mf_mes built its own
KennedyOHaganGP with the default dkl_threshold=30, so deep kernel learning
activated at n_hf=36. Every other experiment in this project passes
dkl_threshold=9999. Kept for the record, not used for any claim.

Numbers were: MF-MES 1.3727/0.5967/0.3680 (mean 0.7791),
              SF-MES 0.9010/1.4814/1.5106 (mean 1.2977) on seeds 42/43/44.
Mechanics verified healthy in this pass: clamp rate 0.000%, cost lands on 548,
SF-MES does exactly 25 HF queries, L-BFGS-B improved 500/500 starts.
