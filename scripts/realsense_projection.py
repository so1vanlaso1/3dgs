"""Versioned DB3 projection, shared by preparation, bridging and diagnostics.

SDK mode follows librealsense v2.59.1 src/rs.cpp (Apache-2.0), including
the inverse-Brown tangential evaluation after radial scaling. Historical
unversioned calibration is always interpreted as legacy, never upgraded silently.
Coordinates use camera Z in metres, R @ X + t, column-major recorded rotations.
"""
import numpy as np

LEGACY = 'legacy_inverse_brown_v1'
SDK = 'librealsense_2_59_1_v1'
MODES = (LEGACY, SDK)
MODEL_IDS = {'None': 0, 'Inverse Brown Conrady': 2, 'Brown Conrady': 4}


def intrinsic(info):
    k = np.array([[float(info['fx']), 0, float(info['ppx'])],
                  [0, float(info['fy']), float(info['ppy'])], [0, 0, 1]])
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Invalid intrinsics')
    return k


def coefficients(info):
    c = np.array([float(v) for v in info['coeffs'].split(',')])
    if c.shape != (5,) or not np.isfinite(c).all():
        raise ValueError('Expected finite k1,k2,p1,p2,k3')
    if info['model'] not in MODEL_IDS:
        raise ValueError(f"Unsupported distortion model: {info['model']}")
    return c


def brown(xy, coeff, modified=False):
    x, y = np.asarray(xy, dtype=float).T
    k1, k2, p1, p2, k3 = coeff
    r2 = x*x + y*y
    f = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    xf, yf = x*f, y*f
    if modified:
        x, y = xf, yf
    return np.column_stack((xf + 2*p1*x*y + p2*(r2+2*x*x),
                            yf + p1*(r2+2*y*y) + 2*p2*x*y))


def invert_brown(ideal, coeff):
    raw = ideal.copy()
    for _ in range(30):
        raw += ideal - brown(raw, coeff)
    error = np.max(np.abs(brown(raw, coeff)-ideal))
    if not np.isfinite(raw).all() or error > 1e-7:
        raise ValueError(f'Inverse distortion did not converge: {error}')
    return raw


def project(points, info, mode=SDK):
    if mode not in MODES:
        raise ValueError(f'Unknown projection version: {mode}')
    xyz = np.asarray(points, dtype=float)
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0):
        raise ValueError('Projection requires finite points with positive camera Z')
    xy = xyz[:, :2]/xyz[:, 2:3]
    c = coefficients(info)
    if info['model'] == 'Inverse Brown Conrady':
        xy = invert_brown(xy, c) if mode == LEGACY else brown(xy, c, modified=True)
    elif info['model'] == 'Brown Conrady':
        xy = brown(xy, c)
    k = intrinsic(info)
    return xy*[k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]


def deproject(pixels, depth_m, info):
    """Pinned SDK deprojection for supported model IDs (10 iterations)."""
    k = intrinsic(info)
    c = coefficients(info)
    xy = (np.asarray(pixels, dtype=float)-[k[0, 2], k[1, 2]])/[k[0, 0], k[1, 1]]
    x, y = xy.T.copy()
    xo, yo = x.copy(), y.copy()
    if info['model'] != 'None' and np.any(np.abs(c) >= np.finfo('f4').eps):
        k1, k2, p1, p2, k3 = c
        for _ in range(10):
            r2 = x*x+y*y
            ic = 1/(1+((k3*r2+k2)*r2+k1)*r2)
            xq, yq = (x/ic, y/ic) if info['model'] == 'Inverse Brown Conrady' else (x, y)
            dx = 2*p1*xq*yq+p2*(r2+2*xq*xq)
            dy = 2*p2*xq*yq+p1*(r2+2*yq*yq)
            x, y = (xo-dx)*ic, (yo-dy)*ic
    return np.column_stack((x, y, np.ones(len(x))))*np.asarray(depth_m)[..., None]


def remap(info, K, width, height, mode=SDK):
    yy, xx = np.mgrid[:height, :width]
    rays = np.column_stack(((xx.ravel()-K[0, 2])/K[0, 0],
                            (yy.ravel()-K[1, 2])/K[1, 1], np.ones(width*height)))
    maps = project(rays, info, mode).reshape(height, width, 2).astype('f4')
    valid = np.isfinite(maps).all(2) & (maps[:, :, 0] >= 0) & (maps[:, :, 1] >= 0)
    valid &= (maps[:, :, 0] < int(info['width'])-1) & (maps[:, :, 1] < int(info['height'])-1)
    return maps, valid


def calibration_mode(cal):
    mode = cal.get('projection_version', LEGACY)
    if mode not in MODES:
        raise ValueError(f'Unknown calibration projection version: {mode}')
    return mode


def relative_extrinsics(depth, color):
    def parse(s):
        f = dict(v.split('=', 1) for v in s.split(';'))
        r = np.array([float(v) for v in f['rotation'].split(',')]).reshape(3, 3, order='F')
        t = np.array([float(v) for v in f['translation'].split(',')])
        if not np.isfinite(t).all() or not np.allclose(r.T@r, np.eye(3), atol=1e-5) or np.linalg.det(r) < .999:
            raise ValueError('Invalid recorded rigid calibration')
        return r, t
    rd, td = parse(depth)
    rc, tc = parse(color)
    r = rc@rd.T
    return r, tc-r@td


def zbuffer(pixels, z, width, height, margin=0):
    """Nearest surface in serialized coordinates; preserve holes and collision evidence."""
    # Gate in source precision, then quantize before grouping. Gating after f32
    # conversion changes historical border membership (and downstream sampling).
    p, z = np.asarray(pixels), np.asarray(z)
    valid = np.isfinite(p).all(1) & np.isfinite(z) & (z > 0)
    valid &= (p[:, 0] >= margin) & (p[:, 0] < width-max(margin, .5))
    valid &= (p[:, 1] >= margin) & (p[:, 1] < height-max(margin, .5))
    p, z = p[valid].astype('f4'), z[valid].astype('f4')
    xy = np.rint(p).astype(int)
    # A float64 value just below w-.5 can become exactly w-.5 in float32.
    # Reject a serialized out-of-bounds pixel rather than wrapping to another row.
    in_bounds = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    p, z, xy = p[in_bounds], z[in_bounds], xy[in_bounds]
    key = xy[:, 1]*width+xy[:, 0]
    order = np.lexsort((z, key))
    unique, first, count = np.unique(key[order], return_index=True, return_counts=True)
    keep = order[first]
    dense = np.zeros((height, width), dtype='f4')
    dense.ravel()[unique] = z[keep]
    observed = dense > 0
    collision = np.zeros((height, width), bool)
    collision.ravel()[unique] = count > 1
    return p[keep], z[keep], dict(depth_m=dense, observed=observed,
        missing=~observed, multiple_projected_samples=collision,
        discarded_collisions=len(z)-len(keep))


def align_depth(depth, units, info, R, t, K, width, height, near=.3, far=6., filtered=True):
    import cv2
    if not 0 < units < .1:
        raise ValueError('Invalid depth scale')
    # The historical cleaner only accepted zero-distortion depth. Keep its arithmetic
    # for byte reproducibility; support other declared models via pinned deprojection.
    coeff = coefficients(info) if 'coeffs' in info else np.zeros(5)
    z = depth*units
    valid = np.isfinite(z) & (z > near) & (z < far)
    if filtered:
        low = cv2.erode(z, np.ones((3, 3), np.uint8))
        high = cv2.dilate(z, np.ones((3, 3), np.uint8))
        valid &= (low > near) & (high < far) & ((high-low) < np.maximum(.03, z*.02))
        z = cv2.medianBlur(z.astype('f4'), 3)
    y, x = np.nonzero(valid)
    v = z[valid]
    kd = intrinsic(info)
    if not np.any(coeff):
        xyz = np.column_stack(((x-kd[0, 2])/kd[0, 0]*v, (y-kd[1, 2])/kd[1, 1]*v, v))
    else:
        xyz = deproject(np.column_stack((x, y)), v, info)
    xyz = xyz@np.asarray(R).T+np.asarray(t)
    xyz = xyz[np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)]
    uv = xyz[:, :2]/xyz[:, 2:3]*[K[0, 0], K[1, 1]]+[K[0, 2], K[1, 2]]
    p, v, evidence = zbuffer(uv, xyz[:, 2], width, height, margin=1 if filtered else 0)
    return p, v, float(valid.mean()), evidence
