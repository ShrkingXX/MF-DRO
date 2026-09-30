# MF-DRO — paper design for AISTATS 2027 (8 pp main text)

Status: narrative + structure only. No section is drafted here. Every number below is a
measured result on disk (experiment id given); ⟨·⟩ marks a number that does not yet exist
and may not be written into any submitted text until it does.

Deadlines: abstract Sep 29 '26 AoE (open until ~08:00 CST Sep 30), paper Oct 6 '26 AoE.
Format: 8 pages excl. references/appendix/checklist/AI-use statement; double-blind;
reciprocal-reviewer nomination required. Source: virtual.aistats.org/Conferences/2027.

---

## 1. The one-sentence contribution

> MF-DRO extends Direct Regret Optimization to multi-fidelity Bayesian optimization by
> simulating cross-fidelity rollouts on a Kennedy–O'Hagan ensemble, labelling them with a
> max-value-entropy return that is comparable across fidelities, and training a decision
> transformer with a fidelity head on expert and random rollouts together; on four
> multi-fidelity benchmarks it ⟨matches or improves on⟩ the strongest multi-fidelity
> acquisition at a matched budget while spending no real evaluations on training data.

Nanda's three pillars, as the introduction must land them:

| pillar | content |
|---|---|
| **What** | (i) the MF extension (rollouts, KO-coupled reward, fidelity head); (ii) MES-entropy return as a dense, cross-fidelity, telescoping RTG; (iii) random-rollout augmentation; (iv) a measured account of what the return channel does and does not do in DT-based BO |
| **Why** | budget-matched final simple regret on 4 benchmarks × ⟨10⟩ seeds against MF-MES, MF-GP-UCB, MF-MI-Greedy, SF-DRO; ablations that isolate each component; two-sided measurement of the return channel |
| **So what** | a DT-based optimizer that needs no real trajectories to train, is fidelity-aware, and comes with a mechanistic account that tells practitioners which parts of the DT machinery carry weight |

---

## 2. Two abstracts (Farquhar's five sentences)

### 2a. Target abstract — fill only with measured values

> We introduce MF-DRO, a multi-fidelity extension of Direct Regret Optimization in which
> a decision transformer learns where and at what fidelity to query from rollouts
> simulated on a Kennedy–O'Hagan Gaussian-process ensemble, so that no real evaluations
> are spent on training data. Multi-fidelity optimization must trade information against
> cost across fidelities, and return-conditioned sequence models have no native notion of
> either. MF-DRO simulates rollouts that switch fidelity, labels every step with the
> max-value-entropy information gain of the high-fidelity optimum — a dense return that is
> comparable across fidelities because the ensemble's cross-fidelity coupling scales
> low-fidelity credit by how much it informs the high-fidelity posterior — and adds a
> fidelity head to the transformer. Training on random rollouts alongside acquisition-
> guided ones improves final regret on the benchmark where the base method previously
> lost. On Currin-2D, Hartmann-6D, Borehole-8D and Ackley-10D at a matched cost budget,
> MF-DRO reaches final simple regret ⟨X⟩ against ⟨Y⟩ for the strongest multi-fidelity
> baseline (MF-MES), over ⟨10⟩ seeds. A two-sided measurement shows the return channel of
> the underlying decision transformer is inert — the rollout return is independent of the
> first action it labels, and the location head averages a bimodal target under both
> squared and absolute loss — which localizes where the method's gains come from and
> where the remaining headroom lies.

### 2b. Abstract the evidence supports today

Identical to 2a with the results sentence replaced by:

> On Borehole-8D, the benchmark where the base method previously lost to every baseline,
> MF-DRO closes the gap to MF-MES (6.43 vs 6.40 final simple regret, 20/20 seeds across
> four configurations for the augmentation effect); on Hartmann-6D the base configuration
> already improves on MF-MES (5.93 vs 6.62), and on Currin-2D and Ackley-10D the methods
> are within noise.

2b is a mechanism paper with a partial method result. It is defensible on the current
record. 2a requires the experiments in §8.

---

## 3. What is new relative to `papers/Old_dro.py`

The original: single-fidelity BoTorch GP ensemble varied by lengthscale; rollouts under
one acquisition per step (`rotate_acq` re-draws the acquisition *every call*); per-step
clamped improvement reward summed into RTG; DT with a location head only; K=1 inference
with a **hardcoded** `target_rtg = 1.0`; candidate search around the incumbent with a
fixed-κ UCB ≥ max-LCB filter; early stop when a step fails to improve.

| component | Old_dro.py | MF-DRO | user listed? |
|---|---|---|---|
| surrogate | single-fidelity SingleTaskGP ensemble, lengthscale grid | **KO two-fidelity ensemble**, ρ ~ U(0.3, 0.95) × lengthscale grid | (1) partly |
| rollout action | x only | **(x, ℓ) jointly**; teacher scores `[α_L/c_L, α_H/c_H]` and takes the cost-normalized argmax | (1) |
| LF/HF comparability in the reward | n/a | an LF fantasy rebuilds `gp_lf`, recomputes `Y_δ = Y_hf − ρ·μ_L`, rebuilds `gp_δ`; the HF max-value entropy moves by exactly what the LF point told the HF posterior, scaled by ρ (`ko_gp.py:632–646`) | (1) "discounting by KO" — now stated precisely |
| reward | clamped per-step improvement, ≥0 | **MES-entropy**: `rtg[τ] = log b_τ − log b_T` | (2) |
| DT heads | location | **location + fidelity** (BCE, `lambda_fid`) | (1) |
| training data | acquisition rollouts only | acquisition **+ random-in-ROI** rollouts, 1:1 | (3) |
| real-budget use | simulated (inherited) | simulated (inherited) | (4) — inherited, see §8 |
| **state** | GP hyperparams + best value + step + best position | + **fixed reference-grid posterior features** (μ_H, σ_H, μ_L, σ_L at R points from the rollout's own conditioned model), recent-fidelity window, cost ratio | **not listed** — this is what gives within-batch states any variation at all (h207 STATE-DIAG: without it every τ=0 state in a batch is identical) |
| **inference RTG target** | hardcoded 1.0, unnormalized | `max(batch_max, α·running_max)` from the **current** rollout batch | **not listed** — and measured to matter: a target pinned outside the training range costs 5.27 (h212) |
| **candidate pool** | UCB ≥ max-LCB, fixed κ, around incumbent | UCB ≥ max-LCB with **β calibrated to a target acceptance quantile** | **not listed** — the largest single measured gain in the project: Borehole 15.82 → 11.59, 9/10 seeds (h84/h90) |
| RTG shape | forward sum of clamped rewards | **telescoping** endpoint difference = a future-only return-to-go in DT's own definition `R_t = Σ_{t'≥t} r_t'` | not listed |
| ensemble diversity | lengthscale only | lengthscale **and ρ** | not listed |
| evaluation | iterations | **cost-matched budget**, simple regret of the best HF query at cost 200 | not listed — the metric is the multi-fidelity claim |

Items marked "not listed" are candidates for the contribution bullets. The reference-grid
state block and the calibrated ROI are the two with the largest measured effects and the
two most likely to be asked about.

Inherited, not novel, and should be cited as DRO's: GP-simulated rollouts (claim 4),
Bayesian early stopping, K=1 inference, the DT architecture, the UCB≥LCB filter itself.

---

## 4. Claim (2): what the MES-entropy return buys — grounded in Wang & Jegelka (2017)

1. **Density.** Every observation changes the max-value posterior, so every step earns a
   non-zero reward. Under the original's improvement reward, h208 measured that **97.5% of
   the original's rollouts terminate at step 1 with `rtg[0] = 0`**: the label is zero for
   almost all training data.
2. **Cross-fidelity comparability.** Both fidelities are scored by their effect on one
   quantity — the entropy of the *high-fidelity* optimum — and the KO coupling sets LF's
   credit automatically (table above). No hand-set discount.
3. **A proper return-to-go.** `Σ_{t≥τ}(log b_t − log b_{t+1}) = log b_τ − log b_T`, so
   `rtg[τ]` is the remaining information gain from τ onward: future-only and decreasing,
   which is DT's definition. The flat/terminal label we tried (h207–h213) is not, and
   cost +3.06 on Hartmann RNG-matched.
4. **Positivity where it matters.** `rtg[0]` is positive in practice (targets 0.30–1.13),
   so the floored target schema is valid. A signed label broke it (target pinned at 0.5 for
   108 of 117 iterations; fixing it recovered 5.27, h212).
5. **Cost.** A 1-D entropy over y\* via the Gumbel approximation (MES §3.1), not a
   d-dimensional entropy over x\*.
6. **Theory in hand.** MES has a simple-regret bound (Thm 3.2) and a formal equivalence to
   UCB/PI/EST at one sample (Lemma 3.1). The reward and the teacher share that basis.
7. **Consistency between teacher and label.** The teacher's one-step greedy argmax *is* the
   argmax of the one-step reward, so the training data optimizes the quantity it is
   labelled with.

Items 4–7 are the ones you had not listed.

---

## 5. Section plan and page budget (8 pages)

| § | title | pp | job |
|---|---|---|---|
| 1 | Introduction | 1.25 | problem, DRO in one paragraph, the four contributions as bullets, Figure 1 |
| 2 | Background | 0.75 | MFBO and the KO model; DT and return conditioning; MES (Eq. 4–5) |
| 3 | MF-DRO | 2.0 | 3.1 KO ensemble and state; 3.2 cross-fidelity rollouts and the joint teacher; 3.3 the MES-entropy return; 3.4 the transformer with a fidelity head; 3.5 random-rollout augmentation; 3.6 inference (target schema, ROI pool). Algorithm box. |
| 4 | What the return channel does | 1.0 | the two-sided measurement (label: adj R² ≈ 0; head: averages under both losses) and the horizon result (L8 ≈ L8-TRUNC ≈ L1). One figure, three panels. |
| 5 | Experiments | 2.0 | 5.1 setup (benchmarks, cost ratios, budget-matched metric, seeds, baselines); 5.2 main comparison (Table 1, Figure 2); 5.3 ablations (Table 2: fidelity head, reward, random half, ROI, state block); 5.4 where the augmentation helps and why (cost-ratio ordering) |
| 6 | Related work | 0.5 | MFBO acquisitions; DT / return-conditioned policies; learned/meta BO (OptFormer, NAP, PFNs — all `[CITATION NEEDED — verify]`) |
| 7 | Limitations and conclusion | 0.5 | one place, stated once |

Appendix: reproducibility checklist, AI-use statement, per-seed tables, protocol/retraction
log summary (the 216-experiment record is itself a credibility asset — one paragraph).

---

## 6. Figures and tables

**Figure 1 (the method).** Left to right: real data → KO ensemble (M members, ρ varied) →
per-member rollouts that switch fidelity, drawn by the joint MES teacher (blue) and the
random teacher (grey) → each step labelled `log b_τ − log b_T` → DT with location and
fidelity heads → one real (x, ℓ) query. Caption states the four contributions in one
sentence each. Must read in grayscale.

**Figure 2 (does it work).** Regret vs cost, 4 panels, all methods; endpoint direct-
labelled. Existing code: `experiments/h210-mean-vs-median/code/plot_curves.py`.

**Figure 3 (what the return channel does).** (a) adj R² of `rtg[0]` on the first action for
MES and random rollouts, with and without CRN — all ≈ 0. (b) |x − x\*| of the emitted
query under a 2:1 bimodal target, MSE vs L1, against the two modes at 0.67 and 0. (c)
h208's L1 / L8 / L8-TRUNC on the original implementation, 10 seeds. One message: the
return does not carry the first action's value, the head does not select on it, and
steps after the first do not reach the query.

**Table 1.** Final simple regret at cost 200, 4 benchmarks × {MF-DRO, MF-MES, MF-GP-UCB,
MF-MI-Greedy, SF-DRO}, mean ± se over ⟨10⟩ seeds, best bold, ↓.

**Table 2 (ablations, Borehole + Hartmann).** −fidelity head; improvement reward instead
of MES-entropy (**RNG-matched** — the parity tool exists); −random half; −ROI calibration;
−reference-grid state; rollout length 1 vs 8.

---

## 7. Experiments → claims map

| claim in the abstract | evidence in hand | still needed |
|---|---|---|
| MF extension works | Table 1 rows for CTRL-K1 (Hartmann 5.93 < MF-MES 6.62; Currin 0.13 < 0.35) | n=10; Ackley (3.74 vs MF-MES 4.28 but SF-DRO 3.43 wins) |
| MES-entropy reward is the right label | h213 (label swap, RNG-matched: +3.06 on Hartmann); h212 (target pinning) | the **improvement-reward ablation on the current base, RNG-matched** — never run cleanly |
| random half helps | Borehole 11.59 → 7.17, 20/20 seeds, 4 configs (h207/h209/h211) | see §8.1: it hurts Hartmann |
| fidelity head matters | — | **the head ablation has never been run** |
| no real budget on training | inherited from DRO | a real-rollout DT arm (`use_real_rollout_queries=True` exists as a flag) if claim (4) is stated as a result rather than a design property |
| return channel is inert, two-sided | h221 + h220 + h208 | done; n=100 rollouts / 5 seeds / 10 seeds respectively |

---

## 8. What must be run before abstract 2a can be submitted

Ordered by how much of the abstract each one unblocks.

**8.1 One configuration that is ≤ the base method on all four benchmarks.** Today the best
configuration differs by benchmark (MIXR-L1 on Borehole 6.43; CTRL-K1 on Hartmann 5.93).
Reporting the best per benchmark is the single thing most likely to sink the paper. The
augmentation's effect is monotone in the HF:LF cost ratio (2:1 helps −4.42, 3:1 and 5:1
neutral, 8:1 hurts +5.26; h217) and it shifts real queries toward HF (LF −0.136 / −0.095).
The arm: **`random_p_hf` matched to the acquisition half's own HF fraction, on Hartmann and
Borehole, CTRL-K1 base.** h210's mix-matched arm preserved the Borehole gain (−4.72, 5/5)
on the old base; it has not been run on Hartmann or on this base. 10 runs. This is the
only cheap route to "one configuration".

**8.2 Beat, not tie, MF-MES on Borehole.** MIXR-L1 is 6.43 vs 6.40 at n=5 with MF-MES
ahead on 4 of 5 seeds. A win needs either a real margin or n=10 showing the tie holds.

**8.3 n=10 on the final configuration, all four benchmarks.** Standard errors at n=5 run
1.5–3.0 rel%; the claims in 2a are inside that.

**8.4 The two ablations that back contributions (1) and (2) and do not exist:**
fidelity head off; improvement reward vs MES-entropy on the current base with
`rtg_rng_parity=True`. Without them the contribution bullets are asserted, not shown.

**8.5 Claim (4) as a measured result.** Either state it as an inherited design property
(no experiment; one sentence citing DRO) or run the real-rollout DT arm at matched real
budget. The flag exists; the baseline does not.

**8.6 Ackley.** SF-DRO (3.43) beats every MF method including ours (3.74/4.00). "Outperforms
all baselines on all benchmarks" is false there today by a margin larger than the se.
Either a configuration closes it or the sentence changes to the multi-fidelity baselines.

Not required for 2a and should not be started before Oct 6: the DAgger-shaped state-
coverage arm, the multi-acquisition arm, a mixture-density head.

---

## 9. Framing decisions already settled by the record

- **K=1.** The K=8 window costs nothing and gains nothing at matched label (h206, h211),
  and its earlier apparent damage was a positional-embedding defect plus a label
  confound. The paper uses K=1 and says so in one sentence.
- **MES-entropy label.** `terminal_improvement` never helped anywhere and needed three
  repairs (h212, h213, h215-stood-down). One sentence in §4 or the appendix.
- **Rollout length.** L=1 matches or beats L=8 on both implementations (h172, h208). Use
  it, cite the measurement, and let §4 explain why.
- **The ROI and the state block are contributions**, not plumbing. They carry the largest
  measured effects in the project.

## 10. Limitations (write once, in §7)

Four synthetic benchmarks; simple regret of the best HF query as the single metric;
the augmentation's benefit depends on the fidelity cost ratio; the return channel is
inert, so return conditioning is not yet a control knob the user can turn; ⟨10⟩ seeds.
