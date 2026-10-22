# Development data, camera conventions, and frozen view identities

**Declared:** 10 October 2026, Asia/Saigon. **Split version:** stage0-v1. Data preparation and CPU parser checks were executed; baseline training was not. The primary evidence is [data_audit.json](evidence/data_audit.json), the full [synthetic view table](manifests/analytic_spheres_v1_views.csv), and the full [room view table](manifests/room_small_views.csv).

## 1. Chosen scenes and access

| Scene | Purpose | Location relative to repository | Resolution | Views | Initial points | Availability |
|---|---|---|---|---|---|---|
| `analytic_spheres_v1` | Small known-camera fixture for replay/pose diagnostics | `stage_result/stage_0/data/analytic_spheres_v1` | 128×128 | 48 | 1,280 | Generated and parser-checked in Stage 0 |
| `room_small` | Low-resolution official real-scene development | `data/room_small` | 389×259 | 311 | 112,627 | Existing official download, all images audited |

The synthetic scene contains three differently colored/textured opaque spheres and a finite checker floor. Each image uses four deterministic subpixel ray samples, fixed world lighting, and an OpenCV pinhole camera. There are no learned rendering components. Camera ground truth is the exact transform used to cast rays, saved in [ground_truth_cameras.npz](data/analytic_spheres_v1/ground_truth_cameras.npz). The source generator and seed make this provenance inspectable, unlike relying on an undocumented previous smoke fixture.

This is a mechanism fixture, not a public synthetic benchmark or a substitute for realistic scenes. Its known surface-sampled initialization is accurate, its textures/illumination are simple, and its observations are noise-free. Such advantages deliberately isolate the initial camera intervention. Empty point tracks mean it is unsuitable for testing reprojection bundle-adjustment losses without additional correspondence construction. Later generalization requires public synthetic datasets with their original conventions and fresh real scenes.

The room source is the official [Mip-NeRF 360 data release](https://jonbarron.info/mipnerf360/), specifically `room/images_8` and `room/sparse/0` in [the source archive](https://storage.googleapis.com/gresearch/refraw360/360_v2.zip). The existing downloader preserves source files under `data/room_small/source`, rescales intrinsics to the actual image dimensions, and prepares `images` for `data_factor=1`. This audit checked the local prepared/source equivalence; it did not redownload the archive. The recorded local source manifest is embedded in the audit.

Do not use `data_factor=8` on the prepared room folder: it is already at factor eight and its prepared intrinsics already match 389×259. The full-resolution `room_factor2` fixture is outside the first low-resolution Stage 0 choice.

## 2. Exact three-way split

Sort filenames using the repository parser's lexicographic order. Let `i` be the zero-based parser index:

```text
test:        i % 8 == 0
development: i % 8 == 4
train:       i % 8 in {1, 2, 3, 5, 6, 7}
```

This rule is independent of rendered quality, perturbation severity, candidate utility, and method results. It retains every-eighth-view test identities while drawing development views from the former training pool. The three sets are disjoint and exhaust the selected views.

| Scene | Training | Development | Reserved test | Stable identity manifest |
|---|---:|---:|---:|---|
| Analytic fixture | 36 | 6 | 6 | [JSON IDs and parser indices](manifests/analytic_spheres_v1_splits.json) |
| Room | 233 | 39 | 39 | [JSON IDs and parser indices](manifests/room_small_splits.json) |

Synthetic reserved test filenames: `view_000.png`, `view_008.png`, `view_016.png`, `view_024.png`, `view_032.png`, `view_040.png`. Development filenames: `view_004.png`, `view_012.png`, `view_020.png`, `view_028.png`, `view_036.png`, `view_044.png`. All other `view_000`–`view_047` filenames belong to training.

The manifests list **every** room filename, original COLMAP image ID, parser index, role, pixel dimensions, and SHA-256. Stable view identity is `scene:filename`. Do not confuse it with COLMAP's integer ID or the dataset item's `image_id`, which is the position within a split-specific dataset.

For pose embeddings, preserve a map from training dataset item index to stable view ID. The trainer's camera optimizer is sized to the training dataset; after removing development views, previous embedding row numbers do not retain their former meanings. Checkpoint reuse needs explicit remapping. Existing scene-only checkpoints do not establish the required matched-state initialization for this new split.

## 3. Allowed uses and leakage rules

| Role | Allowed uses | Disallowed uses |
|---|---|---|
| Training | Gaussian/camera optimization, pose-probe fit, score measurements, eligibility statistics | Relabeling selected failures into development/test |
| Development | Score selection, branch utility, tuning probe bounds/schedules, quality/count pilot, failure inspection | Camera fitting from development images in the primary evaluation |
| Reserved test | Fixed evaluation after method/protocol freeze | Score/parameter selection, continuation/horizon selection, test-image pose optimization |

Stage 0 decoded files for access/integrity checks and used camera metadata to validate transforms. No model quality was measured on test views. The synthetic preview contains six training views only. Pose metadata may be used for dataset preparation and camera consistency checks; test RGB may not supervise scene/probe optimization.

The real room has already been trained on and its every-eighth-view images have already been evaluated in older runs. No retrospective split can erase this exposure. Therefore, **room is a development fixture for paper claims**, even though its prospectively reserved test role is useful for checking a later frozen protocol. Label such an evaluation “previously exposed development scene” and do not describe it as untouched generalization evidence. Reserve new scenes and their final views for Stage 6.

The COLMAP point cloud and original poses come from the supplied scene reconstruction, which may have used the entire capture. This is conventional provided-initialization provenance, not a strictly training-only SfM reconstruction. All compared methods must share it in the first isolation study. If a future claim requires strict no-test-image preprocessing, reconstruct initialization from training views only and evaluate it as a different setting.

## 4. Camera conventions checked

| Quantity | Convention |
|---|---|
| COLMAP extrinsics | World-to-camera: `x_camera = R_wc @ x_world + t_wc` |
| Parser output | Camera-to-world: `T_cw = inverse(T_wc)` |
| Camera axes | x right, y down, z forward |
| Renderer input | `viewmats = inverse(camtoworlds)` |
| Local pose update | Right composition: `T_cw_updated = T_cw @ Delta` |
| Intrinsics | Pinhole `fx, fy, cx, cy`, pixel coordinates at prepared resolution |
| Scene normalization | Parser similarity/principal-axis transformation applied consistently to poses/points |
| Scene scale | Maximum camera-center distance from the mean center, as recorded by this parser |

Both chosen datasets have `PINHOLE` cameras. Distortion estimation is outside scope. The analytic world has z up before normalization; normalized coordinates need not preserve that intuitive axis direction. Store raw camera truth and the parser normalization transform; compare pose errors in one common coordinate system.

Camera-space translation in `Delta` becomes a world camera-center displacement through the original rotation. Explicitly measure camera-center movement for normalized translation bounds. The 6D rotation embedding's numeric norm is not a rotation-angle bound. Rotation error uses the relative rotation's geodesic angle; translation error uses camera centers divided by the recorded scene scale.

For the initial fixed scene/prior, use its recorded frame directly to compare synthetic camera truth. If later joint methods change the global gauge, estimate alignment from training reference cameras or an explicitly declared fixed anchor and apply the same alignment to every evaluated camera. Never fit a gauge or evaluation pose using final test RGB.

## 5. Executed integrity checks

The preparation script loaded both datasets through the actual COLMAP parser, with `factor=1`, `normalize=True`, and `test_every=8`. It verified all images are decodable, all camera/point values finite, inverse transforms consistent, rotations orthonormal, rotation determinants near +1, and split identities disjoint. It instantiated restricted train/development datasets and loaded a representative sample from each.

Prepared room poses and all points match the preserved source reconstruction; its prepared camera intrinsics match a copy of each source camera rescaled to the prepared dimensions. Numeric errors and source-model hashes are in [the audit](evidence/data_audit.json). Pixel hashes are per-view CSV columns. These checks establish inspectable access and conventions; they do not establish baseline quality or pose truth for room.

Saved normalized camera transforms, intrinsics, normalization transform, and scene scale are in [synthetic cameras](manifests/analytic_spheres_v1_cameras.json) and [room cameras](manifests/room_small_cameras.json). These are metadata, not learned camera corrections.

## 6. Trainer integration required in Stage 1

The stock dataset implements two roles: train excludes every eighth view, and every other split string selects the held-out set. Thus `split='dev'` does **not** create the three-way split. Stage 0 verified manifest-restricted dataset loading, but did not modify the trainer.

Stage 1 must load the explicit manifest before creating optimizer embeddings or preloading images:

```python
# Sketch of the required integration, not a patched training entry point.
trainset = Dataset(parser, split="train")
trainset.indices = np.asarray(manifest["parser_indices"]["train"])
devset = Dataset(parser, split="val")
devset.indices = np.asarray(manifest["parser_indices"]["development"])
testset = Dataset(parser, split="val")
testset.indices = np.asarray(manifest["parser_indices"]["test"])
# Automatic eval during development must consume devset, never testset.
# Build the training item-to-view-ID map before initializing CameraOptModule.
```

Verify that all Stage 1 Gaussian training, camera updates, probe sampling, and continuation sampling exclude development/test indices. Existing launcher commands run the old two-way split and cannot silently be reused for the new study.

## 7. Preparation reproduction

From `D:\gsplat\gsplat`:

```powershell
.\.venv\Scripts\python.exe stage_result/stage_0/scripts/prepare_stage0_data.py
```

The script regenerates its analytic fixture and Stage 0 manifests deterministically. It reads the existing room/source data and writes only under `stage_result/stage_0`. It does not alter room images, COLMAP models, production trainer code, or existing results. Preserve a copy/version before deliberately changing the scene generator or split policy; changed hashes require a new manifest version.
