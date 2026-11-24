"""Sample full-resolution fused surface tiles without merging their face arrays."""

import json

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def paths(geometry):
    manifest = geometry / "surface_tiles.json"
    if manifest.is_file():
        return [geometry / name for name in json.loads(manifest.read_text())]
    return [geometry / "scene.glb"]


def read_tile(path):
    tile = o3d.io.read_triangle_mesh(str(path))
    if not tile.has_triangles() or not tile.has_vertex_colors():
        raise ValueError(f"Incomplete mesh tile {path}")
    return tile


def weights(tile):
    xyz, faces, rgb = map(np.asarray, (tile.vertices, tile.triangles, tile.vertex_colors))
    areas = np.zeros(len(faces), dtype=np.float32)
    detail = np.zeros_like(areas)
    for first in range(0, len(faces), 200_000):
        block = slice(first, first + 200_000)
        edge = faces[block]
        a, b, c = (xyz[edge[:, i]] for i in range(3))
        area = .5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        color_a, color_b, color_c = (rgb[edge[:, i]] for i in range(3))
        contrast = np.maximum.reduce((np.linalg.norm(color_a - color_b, axis=1),
                                      np.linalg.norm(color_a - color_c, axis=1),
                                      np.linalg.norm(color_b - color_c, axis=1)))
        areas[block] = area
        detail[block] = area * np.minimum(255 / 20 * contrast, 3.)
    return areas, detail


def seed_tiles(geometry, count, rng):
    tiles = paths(geometry)
    arrays = [weights(read_tile(path)) for path in tiles]
    areas = np.array([a.sum(dtype=np.float64) for a, _ in arrays])
    details = np.array([d.sum(dtype=np.float64) for _, d in arrays])
    total_area, total_detail = areas.sum(), details.sum()
    probability = areas / total_area
    if total_detail > 0:
        probability = .72 * probability + .28 * details / total_detail
    allocations = rng.multinomial(count, probability / probability.sum())
    chunks, offset = [], 0
    for path, (area, detail), allocation in zip(tiles, arrays, allocations):
        if allocation == 0:
            offset += len(area)
            continue
        tile = read_tile(path)
        p = area.astype(np.float64) / total_area
        if total_detail > 0:
            p *= .72
            p += .28 * detail.astype(np.float64) / total_detail
        face = rng.choice(len(area), allocation, p=p / p.sum())
        vertex = np.asarray(tile.triangles)[face]
        tri = np.asarray(tile.vertices)[vertex]
        uv = rng.random((allocation, 2))
        uv = np.where(uv.sum(1, keepdims=True) > 1, 1 - uv, uv)
        bary = np.column_stack((1 - uv.sum(1), uv))
        points = np.einsum("ni,nij->nj", bary, tri)
        rgb = np.einsum("ni,nij->nj", bary, np.asarray(tile.vertex_colors)[vertex])
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        reference = np.eye(3)[np.abs(normals).argmin(axis=-1)]
        tangent = np.cross(reference, normals)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        rotation = np.stack((tangent, np.cross(normals, tangent), normals), axis=-1)
        quat = Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]]
        spacing = np.sqrt((area[face] / total_area) / p[face])
        chunks.append({"points": points.astype(np.float32), "quats": quat.astype(np.float32),
                       "rgb": rgb.astype(np.float32), "faces": face + offset, "barycentric": bary.astype(np.float32),
                       "relative_spacing": spacing.astype(np.float32)})
        offset += len(area)
    result = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    order = rng.permutation(count)
    return total_area, {key: data[order] for key, data in result.items()}
