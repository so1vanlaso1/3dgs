# Existing room: exploratory revisit repair and viewer

Date: 2026-11-03. Goal: a freely viewable reconstruction of the whole recorded room. The existing trained baseline is available at `http://localhost:8090` when the viewer is running. It is a genuine 3D Gaussian scene but has reconstruction gaps, haze and uncertain geometry; a complete accurate room is not established. The viewer returned HTTP 200 after launch. This launch alone does not newly prove browser interaction or novel-view rendering; the earlier free-camera audit is documented separately.

## Revisit repair attempt

Used the eight existing distributed-revisit match sets. Manually inspected their [match sheet](../results/room_improve_sdk_distributed_revisits_20261103/contact_sheet.jpg): most correspondences are concentrated on display graphics and the exhibit, some potentially repeated or reflective. This is a review, not manual confirmation of each physical correspondence. `scripts/audit-room-revisit-rigid-repair.py` splits each previously selected match set in half, fits a rigid 3D RGB-D transform with RANSAC to one half and evaluates the other half. Three pairs lacked enough training points; five produced exploratory transforms. 202/250 used 17 training matches and 17 check matches; its median check residual was 0.110 m under the SDK graph and 0.011 m with its fitted transform. This is local, correlated feature-level evidence on a mostly planar exhibit, not independent pose truth. Pair 202/234 had 8 inliers and its check residual worsened from 0.086 to 0.102 m under its own fitted transform. Full results and pair fitting caveats: [report](../results/room_improve_sdk_rigid_repair_20261103/report.json).

`scripts/try-room-revisit-pose-repair.py` created a separate experimental keyframe graph using existing pose-relative adjacent edges plus the 202/250 fitted transform. It did **not** optimize the complete original measurement graph. The fitted loop residual dropped from 0.264 m / 4.37 degrees to 0.017 m / 0.14 degrees. That decrease is expected from fitting the loop; it is not independent validation. The separate exploratory 202/234 transform residual increased from 0.865 m / 9.64 degrees to 1.005 m / 10.13 degrees. The 266/314 comparison changed little (0.0453 to 0.0452 m); all these check transforms are themselves uncertain fits, not ground truth. The trial moved camera centers by a median 0.105 m and maximum 1.073 m in the SDK world frame. This substantial drift and the mixed check do not support replacing the graph. [Graph report](../results/room_improve_sdk_rigid_repair_20261103/graph_trial_report.json) and [separate poses](../results/room_improve_sdk_rigid_repair_20261103/graph_trial_poses.npz).

Decision: **reject this repair for training or deployment**. There is no trustworthy evidence yet that it improves full-room geometry. No original graph, dataset, cloud or checkpoint was modified, and no candidate model trained. A valid repair needs manually confirmed distributed correspondences, nonplanar geometry and independent check regions. The existing recording may be insufficient, especially through missing-depth and reflective areas.

## Use the existing viewer

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/view-clean-realsense.ps1 -ResultDir results/realsense_20261030_134523_120k -Port 8090 -Background
```

Open `http://localhost:8090`. If it is already running, open that address directly. Use Capture replay to navigate recorded views and Pause / explore for free-camera movement. This is the preserved baseline, not the rejected pose trial. The viewer renders a fixed reconstructed scene, not newly acquired room geometry.

Reproduce the trial with fresh output directory: `./.venv/Scripts/python.exe -u scripts/audit-room-revisit-rigid-repair.py`, followed by `./.venv/Scripts/python.exe -u scripts/try-room-revisit-pose-repair.py`. Both use existing preserved diagnostic matches. The trial and evidence should not be mistaken for a full manually annotated match set.
