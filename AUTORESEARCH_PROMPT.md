# Autoresearch restart prompt

Paste the block below into a fresh session at the repo root.

---

/autoresearch

**Goal.** Determine whether MF-DRO's incumbent-freeze on Hartmann 6D has a fix
*within the DRO frame*, and whether that fix yields lower final simple regret
than MF-MI-Greedy and MF-GP-UCB at matched real cost. This is a question, not a
target — "no within-frame fix closes the gap" is a valid, reportable outcome.

**Read first, before running anything:** `PROTOCOL.md` (frozen pre-registration —
you may change the method, never the evaluation), `findings.md`,
`research-state.yaml`.

**Frozen evaluation (do not modify):** Hartmann 6D · baselines MF-MI-Greedy and
MF-GP-UCB · 10 seeds, identical across methods · matched real cost · final
simple regret · success = MF-DRO mean+SE strictly below best-baseline mean−SE ·
identical initial design across all methods within a comparison. Report every
run including gate failures, and individual seed traces alongside mean ± SE.

**Fix scope.** In: simulated-trajectory generation, rollout, Decision
Transformer (architecture/training/conditioning/reward/RTG), DRO acquisition and
its ambiguity set. Initial-design changes allowed *only* if applied identically
to MF-DRO and all baselines. Out: replacing MF-DRO with an existing method (not
a fix), changing the frozen evaluation, selecting seeds/budgets/basin widths on
the basis of results.

**Do not re-derive these — they are established:**

1. H0 (narrow-basin coverage at initialization) is **REFUTED**. Freeze rate on
   Hartmann 6D, from 1280 inventoried prior runs: SF-DRO 12/12, MF-GP-UCB 10/10,
   MF-DRO 9/12, versus MF-MI-Greedy 0/10, Greedy-MES 0/12, KO-MES / Additive-MES
   / SF-MES 0/5 each. The same initial design does not freeze MI-Greedy, so init
   coverage is not the primary mechanism.
2. H1 (freeze originates in the DRO/DT component, not the multi-fidelity
   machinery) is **SUPPORTED**: SF-DRO freezes more than MF-DRO.
3. H2 (DRO × narrow-basin interaction) is **SUPPORTED**: MF-DRO does not freeze
   on Ackley 10D (incumbent-improvement 7/9/8 vs 3–4 for baselines).
4. `REVISION_LOG.md`'s Hartmann initialization claim is verified **wrong** —
   6.2% not 12%, an 86% gate-failure rate, and seed=42 sits at the 30th
   percentile. Do not build on it.
5. The inventory already exists at `data/results_inventory.csv`
   (`src/analysis/inventory.py`). Do not rebuild it.

**Leading open question.** There may be two distinct freeze mechanisms. A
fidelity-head threshold bug in `src/model/decisionTransformer.py` (`p_val > 0.5`
against a measured τ=0 HF label rate of 0.371, forcing LF selection every
iteration) would explain MF-DRO — but *cannot* explain SF-DRO, which has no
fidelity head and freezes 12/12. Resolve whether this is one mechanism or two
before designing the fix.

**Feasibility, so you calibrate.** Mean final simple regret: MF-MI-Greedy 0.279,
Greedy-MES 0.36–0.45, MF-DRO 1.31, MF-GP-UCB 1.99–2.46. MF-DRO already beats
MF-GP-UCB; the binding target is MI-Greedy at roughly 4× lower regret, and
MF-DRO's best single seed (0.576) is still worse than MI-Greedy's mean. Prior
runs used very different iteration counts (MI-Greedy median 53, MF-DRO 100,
MF-GP-UCB 800), so existing cross-method numbers may not be cost-matched —
re-verify rather than inherit them.

**Practical.** Use `.venv/bin/python`. Work on a dedicated branch. Lock each
experiment protocol in a git commit *before* running it; never combine protocol
and results in one commit. Record negative results with what they rule out.
Update `findings.md`, `research-log.md`, `research-state.yaml` on real progress.
Write progress reports to `to_human/`.

---

## Two things to know before you paste it

**The loop only fires when the session is idle.** It never interrupts a query,
so if you stay in conversation it makes zero progress. Start it, then leave the
session alone.

**Use a git worktree if you want to work in parallel.** The loop owns the working
tree it runs in — it commits with `git add -A` and writes `findings.md`,
`research-state.yaml`, `research-log.md`, `experiments/`, `data/`, `to_human/`,
and new `results/` dirs. Give yourself a separate checkout or your in-progress
edits will land in its commits.
