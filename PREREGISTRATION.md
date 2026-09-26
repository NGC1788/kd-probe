# Pre-registration: P1 probe (written 2026-09-26, before any stage-1 result)

This file may be edited only until the first stage-1 result exists. Every later change goes to the deviation log at the bottom, with a date and a reason.

## Question
In supervised KD with a **frozen** pretrained teacher, does removing the ground-truth CE term increase the **extra** loss caused by teacher-input masking (MaskedKD)?

MaskedKD's DINO experiment (Table 4) is label-free but uses an EMA teacher, so it does not answer this question. No prior probability of success is stated.

## Fixed settings
| Item | Value |
|---|---|
| Data | ImageNet-100, CMC (Tian et al.) class list, from HF `clane9/imagenet-100` (126,689 train / 5,000 val) |
| Teacher | DeiT III-B/16, official IN-1k checkpoint `deit_3_base_224_1k.pth`, frozen. Its logits are restricted to the 100 classes before the softmax |
| Student | DeiT-S/16, trained from scratch |
| Code | MaskedKD commit `96d052d` with the patches in `probe/launch.py` |
| Loss, CE on | `0.5 * CE_ls(y) + 0.5 * KL(p_teacher || p_student)`, T = 1 (MaskedKD default, alpha = 0.5) |
| Loss, CE off | `0.5 * KL(p_teacher || p_student)`. The CE term is dropped; the KD weight and the LR are unchanged. This is a known limitation: the total loss scale changes |
| Masking | Student last-layer CLS attention, top-k. **Primary: keep 98 of 196 tokens (~50% teacher cost)** |
| Recipe | MaskedKD/DeiT defaults (AdamW, lr 5e-4 x 128/512, mixup 0.8, cutmix 1.0, label smoothing 0.1, drop-path 0.1), batch 128, `--no-repeated-aug`. The sampler advances its epoch on a single GPU and is seeded by the run seed |
| Epochs | Set from `probe/bench.py` **before** stage 1 starts; identical for every run |
| Seeds | 0, 1, 2. Runs are paired by seed across arms: same init, same data order |
| Metric | **Last-epoch** top-1 on the 5,000 val images, not the best epoch. Teacher agreement and ECE may be logged but are not used in any decision |

## Decision rule (primary, 12 runs)
Notation: `gap_c = A_full,c - A_mask,c`, paired by seed, where c = 1 means CE on and c = 0 means CE off. `ΔCE = gap_0 - gap_1`.

Both conditions must hold, each with the same sign in all 3 seeds:
1. `gap_0 >= 1.0 pt`: without labels, the cheap teacher costs accuracy.
2. `ΔCE >= 0.5 pt`: a substantial part of that loss is explained by removing labels.

| Outcome | Condition | Reading | Next |
|---|---|---|---|
| A | 1 and 2 | CE hides the masking loss | stage 2 |
| B | 1, not 2 | loss appears with or without labels. If `gap_1 >= 0.5`, MaskedKD is not reproduced: check that first | no automatic stage 2 |
| C | 2, not 1 | masking helps when labels are present. This is a different finding, not P1 | no stage 2 |
| D | otherwise | P1 rejected | stop |

Reported for each gap: per-seed values, paired mean ± sd, and an n = 3 t-interval (descriptive only).

## Secondary condition (registered now)
- **Runs:** keep 59 tokens (~30% cost), CE on/off, 3 seeds (+6 runs). The full-teacher runs are shared with the primary.
- **Whether to run it:** decided from the bench **before** stage-1 results. It is the first thing dropped if the budget is short.
- **Evaluation:** the same rule, applied independently. Prediction: `gap_0(30%) >= gap_0(50%)`.
- **If the primary fails and only the secondary passes:** this does **not** count as P1. It is recorded as the narrower claim "the loss appears only at <= 30% cost", which would need re-testing under new conditions.
- **No stronger masking** than 30% will be added.

## Shared-GPU note
The GPU is shared with another job. Throughput measured here is for **budgeting only**. Accuracy is unaffected because epochs are fixed. Any later wall-clock or cost claim requires a separate measurement on an idle GPU.

## Budget rule
- After the bench, the GPU-hours for the 12 (+6) runs are fixed.
- If the estimate is over budget, then **before** stage 1: shorten the epochs equally for all runs, or drop the secondary.
- After results exist, epochs, seeds, masking ratio and thresholds are not changed.

## Deviation log
(none)
