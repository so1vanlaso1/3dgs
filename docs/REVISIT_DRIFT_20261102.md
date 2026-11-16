# Courtyard revisit drift investigation — 2026-11-02

## Problem

The user observes the same area looking like separate floors or spaces when moving between the capture passes, one with a group of people and one after they left. The saved 120k checkpoint is preserved. A straight-line viewer transition can itself cross the floor, so that visual effect alone does not establish a duplicate reconstruction.

## Findings

The all-30 RGB-D pose graph contains 29,276 edges. None spans 300 or more frames; consequently it has no direct long-range constraints between the approximately 107s and 216s revisits. The graph can be connected while accumulating substantial drift.

A preliminary SIFT scan matched reference frame_003206.png against every 30th frame from 5000 through 6980. Frame_006470.png yielded 262 Lowe-ratio matches, 219 fundamental-matrix RANSAC inliers; see candidate_matches.json. The two source images show the same courtyard, with the people only in the earlier image. This is a candidate, not independent proof of correct geometry.

Using the stored aligned sampled sensor depths and nearest depth samples within two pixels, 55 matched pairs had depth in both views. A 3D rigid RANSAC using the pipeline's approximate depth-dependent threshold retained 29/55 pairs, with median residual 0.0384 m and p90 0.0549 m among inliers. That candidate relative transform differs from the saved pose-graph relative transform by 13.65 m in translation and 33.05 degrees in rotation. Frames/indices were checked against calibration.json. The depth-based estimate is preliminary: correspondences concentrate heavily on the floor, which is close to planar, and dynamic objects, depth calibration and sampling can introduce errors. Do not treat this one pair as ground truth or as a safe loop-closure edge without additional independent checks.

Evidence: results/revisit_diagnostic_20261102/candidate_matches.json, depth_revisit.json, depth_matches.jpg (shows all both-depth pairs, not just RANSAC inliers). This is strong evidence of a revisit inconsistency and a plausible explanation for duplicate or misplaced surfaces, not proof that people caused a second floor.

## Repair still required

Build and independently validate multiple static-region long-range loop closures with spatially distributed, non-coplanar feature/depth evidence; inspect transform consistency and held-out correspondences. Re-optimize a separate pose graph, without the existing tree-distance gate that rejects large but legitimate loop corrections, and validate trajectory and surface alignment. Rebuild the initialized scene and retrain a separate model using corrected poses; handle moving people with an explicit static-scene masking strategy if a clean empty scene is the goal. Compare matched held-out images and revisit surface alignment before accepting it. A static reconstruction cannot faithfully show the group both present and absent as a time-varying event. No corrected poses, trained model, or verified fix exists yet.
