"""Voxel connected components for Gaussian centers, preserving original splats."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def support_component_filter(means, scales, scene_scale, sigma, minimum_splats=100):
    """Conservative support approximation: spheres enclosing Gaussian ellipsoids.

    Links are tested only among 12 nearest centers, so this is a sparse graph,
    not an exhaustive ellipsoid intersection test. A scene-relative link cap
    prevents enormous splats from connecting arbitrarily distant regions.
    """
    distance, neighbors = cKDTree(means).query(means, k=13, workers=8)
    distance, neighbors = distance[:, 1:], neighbors[:, 1:]
    radii = scales.max(1)
    support = sigma*(radii[:, None]+radii[neighbors])
    connected = (distance <= support) & (distance <= .04*scene_scale)
    a, slot = np.nonzero(connected)
    b = neighbors[a, slot]
    graph = coo_matrix((np.ones(len(a), dtype=np.uint8), (a, b)), shape=(len(means), len(means))).tocsr()
    n, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    keep = sizes[labels] >= minimum_splats
    info = dict(method='12-nearest-center graph with enclosing-sphere support overlap',
                sigma=sigma, maximum_link_distance_scene_fraction=.04,
                minimum_splats=minimum_splats, components=n,
                kept_components=int((sizes >= minimum_splats).sum()),
                largest_components=sorted(sizes.tolist(), reverse=True)[:20],
                removed=int((~keep).sum()), retained=int(keep.sum()))
    return keep, info


def component_filter(means, voxel_size, minimum_splats):
    cells = np.floor((means-means.min(0))/voxel_size).astype(np.int64)
    cells, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    dims = cells.max(0)+3
    # Padding permits neighbor lookup without wrapping at the bounding box.
    cells += 1
    keys = np.ravel_multi_index(cells.T, tuple(dims))
    assert np.all(keys[1:] > keys[:-1])
    edge_a, edge_b = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                delta = dx*dims[1]*dims[2]+dy*dims[2]+dz
                index = np.searchsorted(keys, keys+delta)
                valid = index < len(keys)
                source = np.flatnonzero(valid)
                source = source[keys[index[source]] == keys[source]+delta]
                edge_a.append(source)
                edge_b.append(index[source])
    a, b = np.concatenate(edge_a), np.concatenate(edge_b)
    graph = coo_matrix((np.ones(len(a), dtype=np.uint8), (a, b)), shape=(len(keys), len(keys))).tocsr()
    n, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, weights=counts).astype(np.int64)
    keep = sizes[labels[inverse]] >= minimum_splats
    info = dict(voxel_size=float(voxel_size), adjacency='26 neighboring occupied voxels',
                minimum_splats=minimum_splats, occupied_voxels=len(keys), components=n,
                kept_components=int((sizes >= minimum_splats).sum()),
                largest_components=sorted(sizes.tolist(), reverse=True)[:20],
                removed=int((~keep).sum()), retained=int(keep.sum()))
    return keep, info


if __name__ == '__main__':
    import json
    import torch
    s = torch.load('results/realsense_15fps_experiment/train30k/ckpts/final_post_update.pt', map_location='cpu', weights_only=True)['splats']
    for fraction in (.004, .008, .016):
        _, info = component_filter(s['means'].numpy(), fraction*1.3304576673302062, 500)
        print(json.dumps(info), flush=True)
