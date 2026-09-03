# h197 pre-launch: real-query b, before vs after the candidate-pool fix

Borehole_8D seed 42, identical except the pool used to fit b.

| pool | net change in b | steps down/up | monotone |
|---|---|---|---|
| uniform, 200 pts (wrong) | +0.36 (+2.7%) | 13/12 | no |
| ROI-Q10, 600 pts (correct) | -3.73 (-27.9%) | 14/9 | no |

roi_accept at inference spanned 0.086-0.111 across 24 real queries (target 0.10),
so the ROI filter binds at inference as it does in training.

b is NOT monotone after the fix and is not expected to be: rtg[tau] is an
endpoint difference H(y*|D_tau) - H(y*|D_T), which carries no monotonicity
guarantee. This is why PAVA was removed rather than retuned -- training applies
no projection, so imposing one at inference would itself be a mismatch.

```
    q 0: b =  13.3593   roi_accept = 0.1038
    q 1: b =  14.6003   roi_accept = 0.0963
    q 2: b =  14.4241   roi_accept = 0.1005
    q 3: b =  15.2716   roi_accept = 0.0985
    q 4: b =  15.1709   roi_accept = 0.1032
    q 5: b =  14.6731   roi_accept = 0.1011
    q 6: b =  14.0844   roi_accept = 0.0999
    q 7: b =  14.1980   roi_accept = 0.1065
    q 8: b =  11.0386   roi_accept = 0.0988
    q 9: b =  10.8268   roi_accept = 0.1042
    q10: b =   9.8362   roi_accept = 0.1010
    q11: b =  10.6947   roi_accept = 0.1012
    q12: b =  11.3313   roi_accept = 0.0931
    q13: b =  10.6793   roi_accept = 0.0944
    q14: b =  11.2588   roi_accept = 0.1015
    q15: b =  11.0519   roi_accept = 0.0971
    q16: b =  12.0668   roi_accept = 0.0917
    q17: b =  11.0531   roi_accept = 0.0975
    q18: b =  10.5235   roi_accept = 0.1078
    q19: b =  10.2291   roi_accept = 0.1113
    q20: b =  10.3947   roi_accept = 0.0862
    q21: b =  10.2069   roi_accept = 0.1047
    q22: b =   9.4725   roi_accept = 0.0971
    q23: b =   9.6334   roi_accept = 0.1043
  steps where b FELL   : 14/23
  steps where b ROSE   : 9/23
  net change over run  : -3.7259  (-27.9%)
  -> monotone non-increasing? NO
```
