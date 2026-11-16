# Annotated literature and extraction comparison

Consolidated 9 November 2026 for schedule Weeks 2–3. Primary publication pages/abstracts and method descriptions were checked; these are focused method notes, not a claim to have reproduced every paper. Local prior source audits remain under `stage_result/stage_0/`.

| Source | Mechanism / useful insight | Limitation and implication here |
|---|---|---|
| [Kerbl et al., 3DGS, 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | Sparse calibrated points initialize anisotropic Gaussians; interleaved optimization/density control and visibility-aware splatting optimize view synthesis. | Opacity/covariance primitives are volumetric appearance elements, not a polygon adjacency structure or signed surface; high PSNR does not certify a mesh. Local B00 is gsplat, not an independent official-code reproduction. |
| [Revising Densification, 2024](https://arxiv.org/abs/2404.06109) | Reconsiders adaptive density control and image-error-driven densification. | Motivates attributed image-error ranking; does not make this repo's ranking new or prove high-error parents are useful splits. Local residual A05 warm-start evidence was negative. |
| [AbsGS, 2024](https://arxiv.org/abs/2404.10484) | Identifies gradient collision and uses a homodirectional positional gradient to improve splitting in detailed regions. | Local A05 combines existing renderer absolute gradients with its budget/ranking/split operator. Threshold changes with gradient policy; separate effects are not isolated. |
| [Improving Densification, 2025](https://arxiv.org/abs/2508.12313) | A densification-focused method used by the local long-axis split line of work. | Local A04 changes placement, scale and opacity together; factorial ablation isolates the bundle from clone replacement, not each internal component. |
| [SuGaR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Guedon_SuGaR_Surface-Aligned_Gaussian_Splatting_for_Efficient_3D_Mesh_Reconstruction_and_CVPR_2024_paper.pdf) | Regularizes Gaussians toward surfaces and extracts an oriented point-based mesh with Poisson reconstruction; can refine mesh-bound Gaussians. | Directly addresses Gaussian-to-surface alignment. More integration work than a plain depth-fusion baseline; the local curvature prior is not a reproduction. |
| [2DGS, 2024](https://arxiv.org/abs/2403.17888) | Uses 2D Gaussian surface primitives and geometry-oriented regularization for reconstruction. | Changes the representation and rasterization assumptions. Appropriate geometry comparator, but replacing classic 3DGS would change this project's frozen allocation experiment. |
| [Gaussian Opacity Fields, 2024](https://arxiv.org/abs/2404.10772) | Formulates an opacity field for adaptive surface reconstruction from Gaussians. | More principled field extraction than treating Gaussian centers as a surface; requires its field/geometry assumptions and implementation. |

## Surface reconstruction choices

| Approach | Required input | Strength | Main failure / local decision |
|---|---|---|---|
| Poisson | Oriented surface samples/normals | Produces a globally reconstructed surface; SuGaR uses this route after alignment. | Inconsistent normals, outliers and missing observations can create unsupported closed surfaces. Do not feed arbitrary Gaussian centers as certified surface samples. |
| Marching Cubes | A scalar grid with a defined isovalue | Explicit polygon extraction from an implicit field. | It extracts whatever field it is given; density isovalue and voxel resolution can radically alter geometry. It is an extraction algorithm, not an image-to-field method. |
| Rendered-depth TSDF + Marching Cubes | Gaussian checkpoint, training cameras, expected depths, camera intrinsics | Reuses working renderer; enables a small explicit, reproducible optimized-Gaussian-to-mesh baseline. | Expected depths can mix surfaces; opacity holes and pose errors remain. Chosen initial extraction route, not a claim of state-of-the-art geometry. |
| Ball pivoting | Point cloud, oriented normals and radius | Existing reversible diagnostic mesh on RGB-D source points. | Radius sensitivity, holes and bridging. Useful source-depth baseline; does not extract the optimized Gaussian surface. |

[Open3D RGB-D integration documentation](https://open3d.org/html/tutorial/pipelines/rgbd_integration.html) documents the scalable TSDF interface used locally. [Open3D TSDF integration](https://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/integration.html) explains its mesh extraction through Marching Cubes. The local choice is an engineering baseline inferred from available renderer outputs and library support; no cited paper validates the local outcome in advance.
