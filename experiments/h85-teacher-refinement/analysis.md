# H85 analysis — 24/24 runs, 0 failures. Control PASSED 4/4 bit-identical.

| benchmark | arm | mean | sd | paired vs control |
|---|---|---|---|---|
| Hartmann_6D | REFINE-0 (control) | 7.99% | 5.85 | — |
| Hartmann_6D | REFINE-100 | 6.05% | 2.61 | **-1.93 pts (3/5)** |
| Hartmann_6D | HF-FLOOR | 6.62% | **2.00** | **-1.37 pts (3/5)** |
| Borehole_8D | REFINE-0 (control) | 15.82% | 2.36 | — |
| Borehole_8D | REFINE-100 | **9.96%** | 2.78 | **-5.85 pts (5/5)** |
| Borehole_8D | HF-FLOOR | 16.00% | 2.34 | +0.18 pts (0/5) |

## Bars

- **P1 MET** (disproportionality). Refinement helps 3x more on Borehole
  (-5.85) than Hartmann (-1.93). This was the bar designed so that a generic
  "refinement helps everywhere" result would FAIL, and it did not fail.
- **P2 MET** (mechanism). Borehole's near-bound coordinate fraction rises
  **8.93% -> 16.32%**, from the uniform null to well above it. Refinement makes
  MF-DRO reach the boundary where Borehole's optimum lives. This bar could have
  failed even with the regret improvement, and did not.
- **P3 MET.** REFINE-100 lowers Borehole regret on 5/5.
- **P4 MET.** Wall-clock 1.89x (Hartmann) and 1.97x (Borehole), inside the 2x
  bar -- though the in-flight estimate read 2.02x, so it passes narrowly and
  only on the final measurement.
- **P5 MET** (variance). HF-FLOOR cuts Hartmann across-seed sd 5.85 -> 2.00.
- **P6 REFUTED.** I registered NEGATIVE that the floor would not improve the
  Hartmann mean by >= 1pt. It improved it by 1.37.
- **P7 confirmed literally.** HF-FLOOR is +0.18 on Borehole and four of five
  runs are BIT-IDENTICAL to the control, because the floor never fires there.

## The mechanism claim that held

Before any of these runs, I measured at a matched model state that the ROI moves
teacher action quality +0.010 while refinement moves it +0.046, and argued the
teacher's flat argmax over uniform random candidates -- not the region those
candidates come from -- was the binding constraint. P1 and P2 both confirm it:
refinement helps disproportionately where the optimum sits on a boundary, and it
demonstrably moves queries onto that boundary.

Six mechanism claims were refuted earlier in this project. This is the one that
survived, and it survived a bar designed to break it.

## Where my prediction failed, and why

P6. I argued the floor could not improve the mean because HF COUNT does not
predict outcome across h83's seeds. The floor does not change HF count so much
as HF STREAK STRUCTURE: the control runs 131, 73, 75 and 50-query consecutive
low-fidelity streaks on Hartmann, and the floor bounds every one at 3. A budget
with 12 HF queries evenly spread is a different object from one with 12 HF
queries and a 131-query LF gap, and my seed table could not see the difference.

## What this does NOT establish

1. **MF-DRO still loses.** Borehole with refinement is 9.96% against MF-MES's
   6.40%. The h83 headline stands.
2. **Nothing here is a finding yet.** Amendment 2: h89 confirms both
   interventions at fresh seeds 52-56, registered before these numbers were
   written up, with falsifiers requiring withdrawal.
3. **The Hartmann arms are weak.** Both are 3/5 seeds, and h87 showed exactly
   that shape failing to replicate.
