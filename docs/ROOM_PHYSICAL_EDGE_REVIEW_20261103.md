# Physical RGB-depth edge review

Date: 2026-11-03. **Decision: the RGB-depth alignment gate remains inconclusive.** Interior depth supports identifying foreground and background on three objects, but their boundary measurements do not yet establish a reliable global offset. No correction was fit or applied.

## Method and limitations

`scripts/review-room-physical-edges.py` takes existing, calibrated, unfilled projected camera-Z maps and prepared RGB images. It marks manually selected interior rectangles and reports valid-pixel counts and median depth. The color panels use the same 0.3–6 m scale; black indicates a missing projection. These rectangles were selected after seeing the images and do not independently verify calibration, surface identity, or boundary position. They are not ground-truth range measurements. Outputs are in [physical-edge evidence](../results/room_improve_physical_edges_v02_20261103/), including `probes_and_reservations.json` and four annotated images. The magenta horizontal lines on the box images locate recorded depth profiles, not fitted edges.

An initial version placed the statue rectangles on mixed or incorrect surfaces. That output was retained in `results/room_improve_physical_edges_20261103/` for provenance; its statue labels and values must not be used. The corrected probe positions and results are in the **v02** directory. Both label assignment and depth interpretation remain subject to visual uncertainty.

## Results

| Object and source | Foreground interior median | Background interior median | Physical review |
|---|---:|---:|---|
| [Black box, RGB ID 6487, about 8.144 s](../results/room_improve_physical_edges_v02_20261103/box_reference_6487.png) | 1.902 m | 2.343 m | The foreground box and the wall/column behind it are distinguishable. The depth transition includes a narrow invalid strip and intermediate values. |
| [Same box, RGB ID 6253, about 0.30 s earlier](../results/room_improve_physical_edges_v02_20261103/box_early_6253.png) | 1.897 m | 2.330 m | Same qualitative ordering, but the foreground/background transition is broad and the exact physical silhouette is uncertain. |
| [Kiosk pedestal, frame 000000](../results/room_improve_physical_edges_v02_20261103/kiosk_000000.png) | 2.167 m | 2.519 m | A separate, visually identifiable pedestal is in front of the wall. Its left boundary is a promising candidate, although the edge has holes and a broad depth transition. This reviews the pedestal, not the reflective screen. |
| [Exhibit statue, frame 000276, about 92 s](../results/room_improve_physical_edges_v02_20261103/statue_000276.png) | 1.184 m | 2.080 m | The statue's head is nearer than the background exhibit props. A substantial missing-depth band along the head boundary prevents a precise adjacent-pixel edge comparison. |

The box [depth profiles](../results/room_improve_physical_edges_v02_20261103/box_depth_profiles.csv) contain 200 individual pixel positions across two rows in each of two RGB frames; six positions have no projected sample. For example, at RGB ID 6487, y=400, the depth changes from 2.339 m at x=488 to 2.314 m at x=498, 2.238 m at x=500, missing at x=502, and 1.965 m at x=504. At ID 6253, y=400, depth changes from 2.314 m at x=548 to 2.197 m at x=560, missing at x=562, and 1.977 m at x=564. These are samples, not a validated range edge or proof of why the transition is broad. In particular, the earlier method's single adjacent jump near 2.2–2.3 m is only part of the transition to the approximately 1.9 m box interior; the earlier pixel offsets cannot be treated as exact physical alignment errors.

## Reserved future checks

The following prepared RGB/depth pairs have been reserved by source identity for possible later checks, rather than used to fit a correction during this review. Their detailed content and suitability were not assessed in this review; neighboring views or the same locations have been examined before, so they are not independent scenes or a blind, globally representative test. None appears in the saved 60-image baseline held-out list. The existing 60-image evaluation split is unchanged.

| Prepared image | RGB message ID | Depth message ID | Replay time |
|---|---:|---:|---:|
| frame_000004.png | 1415 | 1406 | 1.637 s |
| frame_000145.png | 37829 | 37824 | 48.358 s |
| frame_000295.png | 76895 | 76890 | 98.484 s |

Exact timestamps are stored in `probes_and_reservations.json` and were checked against the capture manifest. Reserve status applies to this alignment investigation, not to an existing model training split. Before using any of these as validation, check that it shows the required surface and that it was excluded from the particular correction's selection and fitting; if not, choose a different frame and document it. More spatially distinct, independently selected structures would strengthen the evaluation.

## Decision and next step

The three objects provide different tentative ranges, but the surfaces and samples were selected after inspection and the boundaries include gradual transitions or holes. We cannot infer an accurate global pixel shift, time shift, calibration error, or error bound from them. Leave the alignment gate open and retain invalid depth as invalid. Next, test a boundary method that explicitly represents the full transition and rejects holes or ambiguous surfaces, using development objects first and reserved source frames only after the method and any correction are frozen. If usable independent evidence remains insufficient, document the limitation and consider a separately evaluated workflow with restricted or no depth rather than claiming a global depth-alignment pass.

Verification: script compiled, four output images were reviewed, all three reserved RGB/depth IDs matched the source manifest, 200 profile entries and their missing flags were checked, and all depth arrays were finite with matching RGB dimensions. No graph, baseline model, recording or training data was changed. An initial PowerShell `rg` invocation used an unsupported wildcard path; it was rerun with `-g` successfully (no matching prior report or held-out list entries for the reserved frame names).
