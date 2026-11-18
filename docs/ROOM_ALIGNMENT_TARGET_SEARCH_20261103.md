# Search for additional RGB-depth alignment targets

Date: 2026-11-03. **Result: one new candidate target found; alignment remains inconclusive.** No global RGB-depth correction is supported.

## Search and sources

Reviewed an [overview of 40 prepared images](../results/room_improve_target_search_20261103/recording_overview.jpg) spaced approximately four seconds apart across the capture. Selected 22 views around promising or challenging areas and reprojected their original paired DB3 depth with `scripts/audit-room-rgbd-alignment.py`. Their RGB/depth comparisons and metadata are under [target projections](../results/room_improve_target_projections_20261103/). The DB3 was opened read-only; all selected saved cleaned-depth samples were reproduced exactly by the existing preprocessing implementation. That is a reproduction check, not independent calibration validation.

At approximately 8.144 s, the left edge of a black box near the doorway has RGB contrast and projected depth on both sides. In the prepared frame `frame_000024.png`, nine preselected rows were assessed with `scripts/measure-room-new-target.py` in a visually selected narrow strip. Eight had exactly one qualifying adjacent depth jump: signed depth x minus RGB x ranged from -1 to +2 pixels, median 0 px; one had no qualifying jump. Depth values at these jumps were approximately 2.2–2.3 m. [Markers](../results/room_improve_target_search_20261103/black_box_markers.png) and [scanline data](../results/room_improve_target_search_20261103/black_box_scanlines.csv). Cyan marks the strongest RGB gradient and magenta the depth-jump start. This is a provisional boundary correspondence on one object; adjacent rows are correlated and the depth jump may not be the exact same physical edge.

## Check additional original frames

`scripts/extract-room-target-raw.py` extracted **seven original RGB frames and their nearest-time paired depth messages** within roughly 0.3 s of that prepared reference, including the reference itself. This is a targeted sample of the DB3, not all recorded frames. Each pair has original message IDs, timestamps, nearest gyro sample and gap in [manifest.json](../results/room_improve_target_raw_20261103/manifest.json); the [contact sheet](../results/room_improve_target_raw_20261103/contact_sheet.jpg) shows RGB and projected depth jumps. Recreated reference RGB matched the prepared frame pixel-for-pixel, and its projected depth matched the earlier diagnostic map exactly. These checks confirm processing consistency, not sensor accuracy.

For the same black-box edge, `scripts/measure-room-raw-target.py` uses manually placed narrow strips and the same eight selected rows in each frame; it only records rows with one qualifying depth jump and a sufficiently strong RGB gradient. [All rows and rejected cases](../results/room_improve_target_raw_20261103/box_sequence_measurements.csv), [per-frame summary](../results/room_improve_target_raw_20261103/box_sequence_report.json), and individual annotated markers are saved. The strips were chosen after inspecting the images, so the other frames are nearby-view checks, not blind independent validation.

| RGB message ID | Seconds from reference | Candidate rows / 8 | Median signed depth-start minus RGB x | Interpretation |
|---|---:|---:|---:|---|
| 6253 | -0.301 | 4 | +5.5 px | Markers do not coincide; the selected depth edge may be another surface. |
| 6383 | -0.135 | 3 | +3 px | Limited candidates; correspondence uncertain. |
| 6433 | -0.068 | 1 | +1 px | Too few candidates. |
| 6487 | 0 | 7 | 0 px | Reference; seven-row result excludes one of the nine rows in the earlier single-frame test. |
| 6539 | +0.066 | 5 | 0 px | Nearby visual check only. |
| 6617 | +0.166 | 0 | — | No valid unique jump under this rule. |
| 6721 | +0.299 | 1 | -3 px | Too few candidates. |

Depth jumps use the left pixel of the adjacent pair; the actual boundary lies between pixels. Percentages and individual rows are not independent observations or statistical confidence intervals. Image content, viewpoint, exposure motion and selected surface change across frames. The gyro is one nearest sample, not integrated motion. The changing signs and marker positions do not by themselves show a timing error, an incorrect transform, or a true boundary mismatch.

## Other candidates and decision

- The [kiosk](../results/room_improve_target_search_20261103/000000_kiosk.png) contains depth gaps over its screen and scattered jumps near the border. It is not yet a reliable correspondence.
- At [32 s, the doorway edge](../results/room_improve_target_search_20261103/000096_door.png) has depth jumps offset from a prominent RGB boundary and values near the 6 m limit. These may be different physical surfaces; do not treat the observed separation as a calibration error.
- The [8 s box grid](../results/room_improve_target_search_20261103/000024_bin.png) is the best new candidate identified in this search, but nearby views demonstrate unstable or missing measured edges. The box and the previously examined statue offer two distinct objects at different indicated ranges, but they do not yet constitute sufficient validated alignment evidence.
- Bright outdoor doorways, reflective cases and missing depth limit several other candidates. The search was sampled, not exhaustive; it cannot establish that no better target exists in the DB3.

**Decision:** keep the alignment gate unresolved, do not fit or apply a correction, and preserve all current data. Next, inspect the box with explicitly annotated physical foreground/background edges and use additional distinct non-reflective regions and reserved frames to evaluate any candidate correction. If the data cannot support that, constrain depth use to regions with independent evidence and document the limitation. No graph changes, point deletion or training performed.
