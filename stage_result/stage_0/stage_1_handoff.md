# Stage 1 handoff from the completed Stage 0 specification

**Date:** 10 October 2026. This file records later implementation needs discovered during Stage 0. It does not claim Stage 1 was executed. All Stage 0 products remain in this folder, per the user's requested output location.

## 1. Fixed starting point

Use the research question, scope, three-way split manifests, and initial effect margins in this package. Start from `DefaultStrategy` and the existing pose module. Preserve unrelated budget/split experiments. The source checkout revision observed in Stage 0 is `3a655ff4fa7673a2c98060198c45f2897bb39f3e`; inherited planning files were staged/untracked, so revision alone is not the complete working state. [Workspace provenance](evidence/workspace_provenance.json) records hashes and initial status.

## 2. Repository hooks and implications

| Hook | Current behavior checked | Required Stage 1 action |
|---|---|---|
| [COLMAP parser](../../examples/datasets/colmap.py) | Sorts filenames; loads world-to-camera then inverts; normalizes poses/points | Match stable manifest IDs and record normalization |
| `Dataset.indices` in that file | Default two-way every-eighth-view split | Add explicit train/development/test manifest loading before preloading/optimizers |
| [CameraOptModule](../../examples/utils.py) | Train-item embedding; right-composed camera-to-world update; 3 translation + 6 rotation coordinates | Preserve view-ID maps; measure physical angle/center displacement |
| [Trainer](../../examples/simple_trainer.py) | Pose optimization configuration exists; automatic eval is a single validation dataset | Direct development-time eval to development views; reserve final test invocation |
| Trainer `--ckpt` path | Evaluation-only; not full training-state resume | Add a distinct full-state snapshot/replay path |
| [DefaultStrategy](../../gsplat/strategy/default.py) | Accumulates normalized gradient norms/counts; clone/split by scale | Use identical units and preserve strategy accumulators |
| Existing [small-data launcher](../../scripts/test-small-dataset.ps1) | Runs old two-way split and does not expose all CCD-study controls | Do not present its current commands as manifest-aware experiments |

## 3. Full-state snapshot requirements before a branch study

Save Gaussian values/parameterization, optimizer state for every scene tensor, actual camera base/current state and pose optimizer, LR schedulers, density accumulators/reset schedule, step, stable Gaussian IDs/lineage, candidate IDs, training item-to-view IDs, Python/NumPy/Torch CPU/CUDA RNG states, and future view/sample sequence. Include appearance state if later enabled. Include any cache/history used to make structural decisions.

A scene checkpoint sufficient for rendering can still be insufficient for branching. Reloaded branches must retain unaffected optimizer moments and apply one common new-child state expansion rule. Explicitly handle random draws consumed by a split so the next sampled training views do not drift between branches.

Before interpreting split utility, run two no-intervention continuations from the same snapshot. Report parameter/render/loss discrepancies and observed nondeterminism. Establish a noise floor for the decision criteria, rather than assume CUDA rasterization is bitwise deterministic.

## 4. Stage 1 work order

1. Implement manifest-aware loading and verify sample sequences cannot include development/test views in optimization.
2. Record environment/GPU/version/configuration and initialization hashes at run time.
3. Run ordinary density control on both chosen scenes under the new split; inspect checkpoint reload and development renders.
4. Run the same setup with ordinary pose optimization, with the same initialization, seed, and density schedule.
5. Generate and save clean/mild/moderate training-only perturbations using the Stage 0 convention. Keep the point cloud fixed.
6. Implement and verify full-state snapshots and matched no-intervention replay.
7. Measure branch/probe feasibility and make any justified prospective horizon/cost revisions before looking at CCD-GS effects.

All produced Stage 1 outputs belong to its own stage folder when Stage 1 is requested. This package intentionally contains no newly trained baseline or speculative performance table.

## 5. External-baseline limits carried forward

FastGS is the first candidate for a runnable allocation reproduction. SalientGS provides runnable source but changes MCMC/front-end/pose supervision. Energy-GS and BSGS public default branches are placeholders at the audited commits. Revising/HarmoGS official runnable releases were not established. BlitzGS v2 survival-gate parity with the repository needs checking. Record these states openly; never turn an internal approximation into an “official reproduced” row.

## 6. Required next decision

Stage 1 is complete only when baseline runs are inspectable under the declared splits and matched replay is reliable. Proceed to the isolated probe after that. Existing room quality records establish a working pipeline; they establish neither new-split performance nor CCD-GS efficacy.
