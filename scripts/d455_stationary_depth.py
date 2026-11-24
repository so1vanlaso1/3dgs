"""Capture or analyze one stationary D455 depth trial at a measured distance."""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np

from realsense_projection import deproject


def analyze(raw, config, reference_m):
    if raw.ndim != 3 or raw.dtype != np.uint16 or raw.shape[0] < 2:
        raise ValueError("Expected at least two uint16 Z16 frames")
    if not np.isfinite(reference_m) or reference_m <= 0:
        raise ValueError("Reference distance must be positive and finite")
    scale = config["depth_scale_m"]
    if not 0 < scale < .1:
        raise ValueError("Invalid depth scale")
    depth = np.where(raw > 0, raw * scale, np.nan)
    _, h, w = depth.shape
    yy, xx = np.indices((h, w))
    dx, dy = abs((xx+.5)/w-.5), abs((yy+.5)/h-.5)
    roi = (dx < .4) & (dy < .4)
    center = (dx < .1) & (dy < .1)
    rim = roi & ((dx >= .3) | (dy >= .3))
    persistent = (raw > 0).mean(0) >= .9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Empty pixels excluded by the support mask.
        median_z = np.nanmedian(depth, axis=0)
        std = np.nanstd(depth, axis=0, ddof=1)
        mad = np.nanmedian(abs(depth-median_z), axis=0)
    selected = roi & persistent
    if selected.sum() < 3 or not (center & persistent).any() or not (rim & persistent).any():
        raise ValueError("Insufficient support in the central region or outer band")
    points = deproject(np.c_[xx[selected], yy[selected]], median_z[selected], config["intrinsics"])
    centroid = points.mean(0)
    covariance = (points-centroid).T @ (points-centroid)
    normal = np.linalg.eigh(covariance)[1][:, 0]
    if normal[2] < 0:
        normal = -normal
    plane_errors = (points-centroid) @ normal
    residual_map = np.full((h, w), np.nan)
    residual_map[selected] = plane_errors
    center_z = float(np.median(median_z[center & persistent]))
    edge_z = float(np.median(median_z[rim & persistent]))
    counts = (raw[:, roi] > 0).mean(1)
    tilt = float(np.rad2deg(np.arccos(np.clip(normal[2], -1, 1))))
    return dict(
        frames=len(raw), reference_axial_m=reference_m, configuration=config,
        valid_fraction_whole_image=float((raw > 0).mean()),
        valid_fraction_central_80_percent=float(counts.mean()),
        frame_valid_fraction_p10_p50_p90=np.percentile(counts, [10, 50, 90]).tolist(),
        persistent_roi_pixels=int(selected.sum()), roi_pixels=int(roi.sum()),
        median_center_depth_m=center_z, median_roi_depth_m=float(np.median(median_z[selected])),
        median_center_depth_bias_m=center_z-reference_m,
        fitted_axis_intersection_bias_m=float(normal @ centroid / normal[2]-reference_m) if normal[2] > .1 else None,
        temporal_pixel_std_m_p50_p90=np.percentile(std[selected], [50, 90]).tolist(),
        temporal_pixel_mad_m_p50_p90=np.percentile(mad[selected], [50, 90]).tolist(),
        plane_normal_camera=normal.tolist(), plane_tilt_degrees=tilt,
        orthogonal_plane_rms_m=float(np.sqrt(np.mean(plane_errors**2))),
        orthogonal_plane_p90_abs_m=float(np.percentile(abs(plane_errors), 90)),
        edge_minus_center_depth_m=edge_z-center_z,
        edge_minus_center_plane_residual_m=float(np.nanmedian(residual_map[rim])-np.nanmedian(residual_map[center])),
        note="Depth bias assumes the reference is measured to the depth camera optical origin along its center ray. "
             "Use a flat matte target filling the field, perpendicular to that ray. Record reference uncertainty separately.")


def capture(width, height, fps, count):
    import pyrealsense2 as rs

    devices = [d for d in rs.context().query_devices() if "D455" in d.get_info(rs.camera_info.name)]
    if len(devices) != 1:
        raise RuntimeError(f"Expected exactly one connected D455, found {len(devices)}")
    device = devices[0]
    serial = device.get_info(rs.camera_info.serial_number)
    pipeline, requested = rs.pipeline(), rs.config()
    requested.enable_device(serial)
    requested.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(requested)
    try:
        sensor = profile.get_device().first_depth_sensor()
        active = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        k = active.get_intrinsics()
        model_names = {rs.distortion.none: "None", rs.distortion.brown_conrady: "Brown Conrady",
                       rs.distortion.inverse_brown_conrady: "Inverse Brown Conrady"}
        model = model_names[k.model]
        config = dict(device=device.get_info(rs.camera_info.name), serial=serial,
                      firmware=device.get_info(rs.camera_info.firmware_version), width=k.width, height=k.height,
                      fps=active.fps(), format="Z16", depth_scale_m=sensor.get_depth_scale(),
                      filter_chain="None applied to recorded device depth",
                      intrinsics={key: getattr(k, key) for key in ("fx", "fy", "ppx", "ppy")} |
                      dict(model=model, coeffs=",".join(map(str, k.coeffs))))
        options = ("visual_preset", "emitter_enabled", "laser_power", "exposure", "enable_auto_exposure", "gain")
        frames, ids, stamps, exposure = [], [], [], []
        for _ in range(2 * fps):
            pipeline.wait_for_frames()
        config["sensor_options"] = {name: sensor.get_option(getattr(rs.option, name))
                                    for name in options if sensor.supports(getattr(rs.option, name))}
        for _ in range(count):
            frame = pipeline.wait_for_frames().get_depth_frame()
            frames.append(np.asanyarray(frame.get_data()).copy())
            ids.append(frame.get_frame_number())
            stamps.append(frame.get_timestamp())
            key = rs.frame_metadata_value.actual_exposure
            exposure.append(frame.get_frame_metadata(key) if frame.supports_frame_metadata(key) else None)
        config.update(frame_numbers=ids, timestamps_ms=stamps, actual_exposure_us=exposure,
                      frame_number_discontinuities=int((np.diff(ids) != 1).sum()))
        return np.stack(frames), config
    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--distance-m", required=True, type=float)
    parser.add_argument("--uncertainty-m", required=True, type=float)
    parser.add_argument("--from-capture", type=Path, help="Saved raw_depth.npz from a prior stationary trial")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if not np.isfinite(args.uncertainty_m) or args.uncertainty_m <= 0 or args.frames < 2:
        parser.error("Use a positive reference uncertainty and at least two frames")
    if args.output.exists():
        parser.error("Output must be a fresh directory")
    if args.from_capture is None:
        raw, config = capture(args.width, args.height, args.fps, args.frames)
    else:
        with np.load(args.from_capture) as data:
            raw, config = data["raw"], json.loads(str(data["configuration"]))
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "raw_depth.npz", raw=raw, configuration=json.dumps(config))
    result = analyze(raw, config, args.distance_m) | {"reference_uncertainty_m": args.uncertainty_m}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
