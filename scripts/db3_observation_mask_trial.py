"""Fit DB3 camera poses with visually reviewed observation exclusions.

Prepare freezes source-space masks and all track assignments before any fit.
Run reuses the direct measured-depth solver and untouched historical checks.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy.spatial import cKDTree

import db3_direct_depth_refine as direct


tracked, recovery, geometry = direct.t, direct.r, direct.m
ROOT = geometry.ROOT
REFERENCE = ROOT / "results/db3_direct_depth_20261111"
REGIONS = Path(__file__).with_name("db3_observation_mask_regions.json")
CALIBRATION = geometry.INPUT / "calibration_clean.json"
ARMS = ("unmasked", "people_candidate", "people_and_reflections", "matched_track_count")
REGION_NAMES = ("people_candidate", "reflection_glare", "static_candidate")
PATCH_RADIUS = 11
RESERVE_RADIUS = 12


def load_inputs(case, intrinsics):
    parent = direct.INPUTS / case["label"]
    tracked_parent = direct.PARENT / case["label"]
    rows = geometry.read(parent / "inputs.json")
    if [row["source_id"] for row in rows] != case["source_ids"]:
        raise ValueError("Input roster does not match the archived window")

    original = []
    image_paths = []
    for row in rows:
        path = parent / "inputs" / f'{row["source_id"]}.npz'
        image_paths.append(path.with_suffix(".png"))
        with np.load(path) as data:
            original.append({"uv": data["uv"], "depth": data["depth"]})
    with np.load(tracked_parent / "fit_features.npz") as data:
        fitting = [
            {"uv": data[f"uv_{c}"], "depth": frame["depth"],
             "check": np.zeros(len(data[f"uv_{c}"]), dtype=bool)}
            for c, frame in enumerate(original)
        ]
    with np.load(parent / "moving_checks.npz") as data:
        moving = [
            {"uv": xy, "depth": frame["depth"]}
            for xy, frame in zip(data["uv"], original)
        ]
        reserved = [data[f"reserved_{c}"] for c in range(len(rows))]
    for feature_set in (original, fitting, moving):
        recovery.observations(feature_set, intrinsics)
    tracks = geometry.read(tracked_parent / "tracks.json")["tracks"]
    for c, features in enumerate(fitting):
        uv = features["uv"]
        if geometry.check_tile(uv).any() or near_points(uv, reserved[c]).any():
            raise ValueError("Archived fitting observations overlap a historical reserve")
    return original, fitting, moving, tracks, image_paths


def near_points(uv, reserved):
    if not len(reserved):
        return np.zeros(len(uv), dtype=bool)
    return cKDTree(reserved).query(uv)[0] <= RESERVE_RADIUS


def rasterize(case_regions, width, height, frame_count):
    masks = {}
    for name in REGION_NAMES:
        stack = np.zeros((frame_count, height, width), dtype=np.uint8)
        for polygon in case_regions[name]:
            keys = sorted(map(int, polygon))
            vertices = np.array([polygon[str(k)] for k in keys])
            if keys[0] != 0 or keys[-1] != frame_count - 1:
                raise ValueError("Each polygon needs first and last frame keyframes")
            for c, mask in enumerate(stack):
                points = [[np.interp(c, keys, values) for values in xy]
                          for xy in vertices.transpose(1, 2, 0)]
                cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        masks[name] = stack
    return masks


def patch_membership(masks, features):
    inside = []
    for mask, frame in zip(masks, features):
        xy = np.rint(frame["uv"]).astype(int)
        if np.any(xy < 0) or np.any(xy >= [mask.shape[1], mask.shape[0]]):
            raise ValueError("Observation outside prepared image coordinates")
        inside.append(mask[xy[:, 1], xy[:, 0]].astype(bool))
    return inside


def choose_static_holdout(tracks, fitting, static, pairs):
    """Reserve complete source tracks using only annotation and depth support."""
    rng = np.random.default_rng(20261117)
    tile_counts = Counter()
    selected = []
    for tid in rng.permutation(len(tracks)):
        track = tracks[tid]
        if not all(static[c][i] for c, i in track):
            continue
        depth_views = {c for c, i in track if fitting[c]["valid"][i]}
        if not any(a in depth_views and b in depth_views for a, b in pairs):
            continue
        c, i = track[0]
        tile = tuple((fitting[c]["uv"][i] // 80).astype(int))
        if tile_counts[tile] >= 6:
            continue
        tile_counts[tile] += 1
        selected.append(int(tid))
    return sorted(selected)


def freeze_assignments(tracks, fitting, masks, pairs, fixed_static_ids=None):
    kernel = np.ones((2 * PATCH_RADIUS + 1,) * 2, dtype=np.uint8)
    people, glare = (
        patch_membership([cv2.dilate(mask, kernel) for mask in masks[name]], fitting)
        for name in REGION_NAMES[:2]
    )
    static = patch_membership(
        [cv2.erode(mask, kernel) for mask in masks["static_candidate"]], fitting
    )
    static = [s & ~p & ~g for s, p, g in zip(static, people, glare)]
    static_ids = fixed_static_ids
    if static_ids is None:
        static_ids = choose_static_holdout(tracks, fitting, static, pairs)
    static_tracks = [tracks[tid] for tid in static_ids]
    check_rows = recovery.fixed_checks(fitting, static_tracks, pairs)

    reserved = [[] for _ in fitting]
    for track in static_tracks:
        for c, i in track:
            reserved[c].append(fitting[c]["uv"][i])
    blocked = [near_points(frame["uv"], xy) for frame, xy in zip(fitting, reserved)]
    base = [tid for tid, track in enumerate(tracks) if not any(blocked[c][i] for c, i in track)]
    people_free = [tid for tid in base if not any(people[c][i] for c, i in tracks[tid])]
    glare_free = [tid for tid in people_free if not any(glare[c][i] for c, i in tracks[tid])]

    # Match the masked arm's track count within each seed-frame / lifespan group.
    # Spatial distribution, depth validity and residual count can still differ.
    def strata(tid):
        return tracks[tid][0][0], len(tracks[tid]) // 3

    counts = Counter(map(strata, glare_free))
    eligible = defaultdict(list)
    for tid in base:
        eligible[strata(tid)].append(tid)
    rng = np.random.default_rng(20261118)
    count_matched = sorted(
        int(tid) for group, count in sorted(counts.items())
        for tid in rng.choice(eligible[group], count, replace=False)
    )
    return {
        "static_check_tracks": static_ids,
        "static_checks": check_rows,
        "arm_track_ids": dict(zip(ARMS, (base, people_free, glare_free, count_matched))),
    }


def save_preview(image_paths, masks, dest):
    panels = []
    colors = ((184, 40, 188), (30, 81, 240), (81, 182, 71))
    for c in (0, len(image_paths) // 2, len(image_paths) - 1):
        img = cv2.imread(str(image_paths[c]))
        shade = img.copy()
        for name, color in zip(REGION_NAMES[::-1], colors[::-1]):
            shade[masks[name][c] > 0] = color
        panel = cv2.addWeighted(img, .72, shade, .28, 0)
        cv2.rectangle(panel, (0, 0), (610, 33), (24, 24, 24), -1)
        cv2.putText(panel, f'{image_paths[c].stem}  Pink: possible person  Orange: glare  Green: static',
                    (14, 24), cv2.FONT_HERSHEY_SIMPLEX, .53, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    if not cv2.imwrite(str(dest), np.vstack(panels)):
        raise OSError(f"Could not write {dest}")


def prepare(out, reflection_override=None, reuse_holdout_from=None):
    out.mkdir(parents=True, exist_ok=False)
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    dependencies = [
        Path(__file__), REGIONS, Path(direct.__file__), Path(tracked.__file__),
        Path(tracked.p.__file__), Path(recovery.__file__), Path(recovery.route.__file__),
        Path(recovery.t.__file__), Path(geometry.__file__),
    ]
    if reflection_override is not None:
        dependencies.append(reflection_override.resolve())
        override = geometry.read(reflection_override)
        parent_protocol = geometry.read(reuse_holdout_from / "protocol.json")
        for name, digest in parent_protocol["frozen_assignments"].items():
            if geometry.sha(reuse_holdout_from / name) != digest:
                raise ValueError(f"Changed parent assignment or mask: {name}")
    for path in dependencies:
        shutil.copyfile(path, source_dir / path.name)
    calibration = geometry.read(CALIBRATION)
    K = np.array(calibration["K"])
    historical = geometry.read(REFERENCE / "protocol.json")
    annotations = geometry.read(REGIONS)
    image_size = [calibration["width"], calibration["height"]]
    if image_size != annotations["image_size"]:
        raise ValueError("Mask coordinates do not match the prepared camera")

    files = list(dependencies) + [CALIBRATION, REFERENCE / "protocol.json"]
    frozen = []
    for case in historical["cases"]:
        label = case["label"]
        dest = out / label
        dest.mkdir()
        _, fitting, _, tracks, image_paths = load_inputs(case, K)
        regions = annotations["cases"][label]
        fixed_static_ids = None
        if reflection_override is not None:
            regions = {**regions, "reflection_glare": override[label]}
            old_path = (reuse_holdout_from / label / "assignments.json").resolve()
            files.append(old_path)
            old = geometry.read(old_path)
            fixed_static_ids = old["static_check_tracks"]
        masks = rasterize(regions, *image_size, len(fitting))
        assignments = freeze_assignments(tracks, fitting, masks, case["check_pairs"], fixed_static_ids)
        if fixed_static_ids is not None:
            for key in ("unmasked", "people_candidate"):
                if assignments["arm_track_ids"][key] != old["arm_track_ids"][key]:
                    raise ValueError(f"Comparison support changed: {label} {key}")
            if assignments["static_checks"] != old["static_checks"]:
                raise ValueError(f"Static checks changed: {label}")
        geometry.write(dest / "assignments.json", assignments)
        np.savez_compressed(dest / "masks.npz", **masks)
        save_preview(image_paths, masks, dest / "mask_preview.jpg")
        frozen.extend([dest / "assignments.json", dest / "masks.npz"])
        files.extend(image_paths)
        files.extend(p.with_suffix(".npz") for p in image_paths)
        files.extend(direct.INPUTS / label / name for name in (
            "inputs.json", "moving_checks.npz", "frozen_checks.json",
            "original_checks.json", "old_chain_poses.json",
        ))
        files.extend(direct.PARENT / label / name for name in ("fit_features.npz", "tracks.json"))
        files.extend(REFERENCE / label / name for name in ("poses.json", "result.json"))
        print("FROZEN", label, "static checks", [len(r["refs"]) for r in assignments["static_checks"]],
              "fitting tracks", {k: len(v) for k, v in assignments["arm_track_ids"].items()}, flush=True)
    geometry.write(out / "protocol.json", {
        "date": annotations["date"], "timezone": "Asia/Saigon",
        "reflection_override": None if reflection_override is None else override["notes"],
        "fixed_static_ids": reuse_holdout_from is not None,
        "cases": historical["cases"], "geometry_limits": historical["gates"], "arms": ARMS,
        "question": "Does fit-only exclusion of moving foreground and reflected/glare pixels improve held-out pose consistency?",
        "treatment": "Same archived LK tracks. Remove entire fit tracks touching a visual mask anywhere, with an 11px native patch guard. Archived optical flow is reused, including its pyramid footprint. Sensor depth samples and original image pixels are unchanged.",
        "initializer_and_solver": "Rebuild the same adjacent robust metric chain in each arm. Use the existing measured-depth camera-only objective with fixed first camera and intrinsics. Match count by seed-frame and lifespan group in the count control. Use no validation in the optimizer.",
        "evaluation": "Report all original moving and descriptor checks in full. Additional visually static candidates come from exposed archived fit tracks, withheld from all new fits with whole-track and 12px observation reservations. Diagnostic subgroup labels use masks only, never residuals.",
        "decision": "Use the same local-consistency thresholds for complete moving checks. Report static candidate coverage, both half-pairs, metric error and the count-matched comparison. Regional development only. No whole-scene inference from this interval.",
        "versions": {"opencv": cv2.__version__, "numpy": np.__version__},
        "input_hashes": {p.relative_to(ROOT).as_posix(): geometry.sha(p) for p in files},
        "frozen_assignments": {p.relative_to(out).as_posix(): geometry.sha(p) for p in frozen},
    })


def region_statistics(moving, masks, measured):
    regions = {name: patch_membership(masks[name], moving) for name in REGION_NAMES}
    summary = []
    for row in measured:
        a, b = row["cameras"]
        groups = defaultdict(list)
        for tid, error in zip(row["track_ids"], row["errors_px"]):
            if any(regions["people_candidate"][c][tid] for c in range(a, b + 1)):
                group = "people_candidate"
            elif any(regions["reflection_glare"][c][tid] for c in range(a, b + 1)):
                group = "reflection_glare"
            elif all(regions["static_candidate"][c][tid] for c in range(a, b + 1)):
                group = "static_candidate"
            else:
                group = "other"
            groups[group].append(error)
        summary.append({
            "cameras": [a, b],
            "regions": {k: {"matches": len(v), "median_p90_px": np.percentile(v, [50, 90]).tolist()}
                        for k, v in sorted(groups.items())},
        })
    return summary


def score(case, poses, K, original, fitting, moving, assignments, masks):
    parent = direct.INPUTS / case["label"]
    moving_rows = recovery.evaluate(moving, geometry.read(parent / "frozen_checks.json"), poses, K)
    return {
        "moving_checks": moving_rows,
        "descriptor_checks": recovery.evaluate(
            original, geometry.read(parent / "original_checks.json"), poses, K
        ),
        "static_checks": recovery.evaluate(fitting, assignments["static_checks"], poses, K),
        "moving_checks_by_region": region_statistics(moving, masks, moving_rows),
    }


def fit_arm(tracks, fitting, K, dest):
    dest.mkdir(exist_ok=False)
    initial, chain = tracked.initializer(fitting, tracks)
    geometry.write(dest / "initialization.json", chain)
    if initial is None:
        return None, {"complete": False, "reason": "Insufficient fit-only initializer support"}
    try:
        poses, optimization = direct.refine(fitting, tracks, initial, K)
    except ValueError as error:
        return None, {"complete": False, "reason": str(error)}
    geometry.write(dest / "poses.json", poses.tolist())
    geometry.write(dest / "optimization.json", optimization)
    return poses, {"complete": True, **{k: v for k, v in optimization.items() if k != "fitting_pair_refs"}}


def run(out):
    protocol = geometry.read(out / "protocol.json")
    for root, group in ((ROOT, "input_hashes"), (out, "frozen_assignments")):
        for name, digest in protocol[group].items():
            if geometry.sha(root / name) != digest:
                raise ValueError(f"Changed frozen artifact: {name}")
    K = np.array(geometry.read(CALIBRATION)["K"])
    cases = []
    for case in protocol["cases"]:
        label = case["label"]
        dest = out / label
        original, fitting, moving, tracks, _ = load_inputs(case, K)
        assignments = geometry.read(dest / "assignments.json")
        with np.load(dest / "masks.npz") as data:
            masks = {name: data[name] for name in REGION_NAMES}
        fixed = assignments["static_check_tracks"] if protocol.get("fixed_static_ids") else None
        if assignments != freeze_assignments(tracks, fitting, masks, case["check_pairs"], fixed):
            raise ValueError(f"Frozen track split did not reproduce: {label}")
        result = {
            "label": label,
            "mask_pixel_fraction": {name: region.mean(axis=(1, 2)).tolist() for name, region in masks.items()},
            "checks": {},
        }
        sources = {
            "prior_direct_depth": REFERENCE / label / "poses.json",
            "old_chain": direct.INPUTS / label / "old_chain_poses.json",
        }
        for name, path in sources.items():
            poses = np.array(geometry.read(path))
            # The archived poses may have seen the new static checks as fitting tracks.
            result["checks"][name] = score(
                case, poses, K, original, fitting, moving, assignments, masks
            )
        for arm in protocol["arms"]:
            track_ids = assignments["arm_track_ids"][arm]
            poses, fit = fit_arm([tracks[tid] for tid in track_ids], fitting, K, dest / arm)
            entry = {"tracks_available": len(track_ids), "fit": fit}
            if poses is not None:
                entry.update(score(case, poses, K, original, fitting, moving, assignments, masks))
                optimizer_ok = all(fit[k] for k in ("success", "finite", "all_fit_points_in_front"))
                entry["local_consistency_pass"] = optimizer_ok and recovery.support_pass(entry["moving_checks"])
                entry["static_diagnostic_pass"] = optimizer_ok and recovery.support_pass(entry["static_checks"])
                print(label, arm, "moving", entry["moving_checks"][0]["symmetric_reprojection_px"],
                      "static", entry["static_checks"][0]["symmetric_reprojection_px"],
                      "local pass", entry["local_consistency_pass"], flush=True)
            else:
                print(label, arm, fit["reason"], flush=True)
            result["checks"][arm] = entry
        geometry.write(dest / "result.json", result)
        cases.append(result)
    geometry.write(out / "result.json", {"cases": cases, "whole_scene_accepted": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reflection-override", type=Path)
    parser.add_argument("--reuse-holdout-from", type=Path)
    args = parser.parse_args()
    if bool(args.reflection_override) != bool(args.reuse_holdout_from):
        parser.error("Reflection override and parent holdout arguments are required together")
    cv2.setNumThreads(6)
    if args.action == "prepare":
        prepare(args.output_dir, args.reflection_override, args.reuse_holdout_from)
    else:
        run(args.output_dir)
