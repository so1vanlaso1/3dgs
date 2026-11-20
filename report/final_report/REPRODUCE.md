# Reproduce the specialized project results

The editable LaTeX entry point is [final_report.tex](final_report.tex). The pre-revision manuscript is documented in [report_draft.md](report_draft.md). Run the commands below from the repository root.

## Environment and inputs

Use the extracted supplement or the project checkout accompanying the report. The supplement includes gsplat, the GLM CUDA headers, the Python runners, the report and the evidence index. The workstation used Python 3.12.10 x64, PyTorch 2.9.1 with CUDA 12.8, torchvision 0.24.1 with CUDA 12.8, NVIDIA GTX 1660 SUPER (compute capability 7.5), MSVC v143 with Windows SDK 10.0.22621, and Blender 4.5.4 LTS. Open3D 0.20.0 and pycolmap 4.2.1 match the tested extraction and dataset preparation. The package inventory records the measured environment. [Package inventory](evidence/python_packages.txt)

On a new Windows checkout, install a CUDA 12.8 Toolkit, the MSVC x64 C++ workload and Windows SDK, Git, Python 3.12 x64 and Blender. For a Git checkout, initialize submodules. The supplement already includes the required GLM headers. In a PowerShell session inherited from the x64 Visual Studio developer shell, set the CUDA toolkit directory and run the installation commands. Use the GPU's actual compute capability if different from 7.5.

~~~powershell
if (Test-Path .git) { git submodule update --init --recursive }
py -3.12 -m venv .venv
$python = '.\.venv\Scripts\python.exe'
$env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:CUDA_PATH = $env:CUDA_HOME
$env:PATH = "$env:CUDA_HOME\bin;$env:PATH"
$env:TORCH_CUDA_ARCH_LIST = '7.5'
$env:NUM_CHANNELS = '3,4'
$env:MAX_JOBS = '2'
$env:PYTHONPATH = "$(Get-Location);$(Get-Location)\examples;$(Get-Location)\scripts"
& $python -m pip install --upgrade pip setuptools wheel ninja numpy rich
& $python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install --no-build-isolation -e .
& $python -m pip install --no-build-isolation -r examples/requirements.txt
& $python -m pip install open3d==0.20.0 pycolmap==4.2.1 requests matplotlib
& $python -m pip check
& $python -c "import torch; from gsplat.cuda._backend import _C; assert torch.cuda.is_available() and _C is not None; print(torch.__version__, torch.cuda.get_device_name())"
~~~

The repository workstation can use `.\scripts\run-school-completion.ps1` in place of `& $python` for GPU commands. It sets up the local portable MSVC and CUDA directories under `.tools`, a CUDA extension cache and the project import paths. On a new system use the configured system tools above. The numerical rehearsal used the existing local toolchain and an empty results directory.

Prepare the official Mip-NeRF 360 factor-two photographs and COLMAP model. The downloader uses ZIP byte ranges and converts the supplied intrinsics to the actual image dimensions. The downloader's hard links require its source and prepared files on the same filesystem. A usable Room directory has `images/` and `sparse/0/`, 311 images at 1557×1038, and 112,627 initializer points. If the ZIP server cannot honor ranges, obtain the official archive and use the downloaded `room/images_2` and `room/sparse/0`. Keep the poses and points and rescale the camera to the actual image dimensions as in the downloader. [Mip-NeRF 360 data](https://jonbarron.info/mipnerf360/) · [Preparation code](../../scripts/download-room-small.py)

~~~powershell
& $python scripts/download-room-small.py --scene room --image-factor 2
~~~

## Fresh Room training through the edited asset

From the configured PowerShell window, run the same command as in Appendix A:

~~~powershell
$run = '.\report\final_report\reproduce_room.ps1'
$blender = 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe'
& $run -Blender $blender -Work 'results/report_room_replay'
~~~

The script's default Python interpreter is the new checkout's virtual environment. To run each stage individually, add one of "Prepare", "Train", "Extract", "Clean", "Edit" or "Verify" via the Stage parameter, in that order with the same Work directory. Use the full path to your installed Blender executable if it differs from the example. The script contains the exact commands and full flags. The native commands below are equivalent and include a baseline invocation for a comparison.

The commands use the released sequence of fitting-set indices. Its JSON file has SHA-256 `a1fe2a83fc0457ee88e76174e6e732d45cf5159cba10aab1a828e549afa82ab5`. The trained checkpoint, raw polygons, cleaned PLY and reopened Blender project are generated locally. Room's scripted edit expects the prepared dataset under `data/room_factor2` so its fixed camera and patch coordinates are meaningful.

~~~powershell
$work = 'results/report_room_replay'
$fitOrder = 'report/final_report/evidence/room_camera_order.json'
$blender = 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe'
# On the project workstation use .tools/blender-4.5.4-windows-x64/blender.exe.
$training = @(
    '--seed', '42', '--independent-sampling', '--record-experiment',
    '--data-dir', 'data/room_factor2', '--data-factor', '1',
    '--max-steps', '7000', '--eval-steps', '1', '7000',
    '--save-steps', '7000', '--save-ply', '--ply-steps', '7000',
    '--disable-viewer', '--disable-video', '--batch-size', '1', '--packed',
    '--sh-degree', '3', '--sh-degree-interval', '1000', '--cache-images',
    '--data-workers', '0', '--camera-sequence-path', $fitOrder,
    '--strategy.refine-start-iter', '500', '--strategy.refine-stop-iter', '6000',
    '--strategy.refine-every', '100', '--strategy.reset-every', '3000'
)
$candidate = @(
    '--strategy.absgrad', '--strategy.grow-grad2d', '0.0008',
    '--strategy.cap-max', '700000', '--strategy.error-weight', '1',
    '--strategy.edge-weight', '0.25', '--strategy.split-mode', 'long_axis',
    '--strategy.clone-policy', 'split', '--strategy.split-distance', '0.45',
    '--strategy.opacity-reduction', '0.6'
)
if (Test-Path $work) { throw "Choose a fresh results directory" }
& $python -u examples/simple_trainer.py budget @training @candidate --result-dir "$work/candidate"
if ($LASTEXITCODE) { throw 'Training failed' }
& $python -u scripts/gaussians_to_mesh.py --checkpoint "$work/candidate/ckpts/final_post_update.pt" `
    --data data/room_factor2 --output "$work/raw" --views 64 --downsample 4 --voxel 0.01 --max-resolution 192
if ($LASTEXITCODE) { throw 'Extraction failed' }
& $python -u scripts/week11_room_cleanup.py --source "$work/raw" --output "$work/design"
if ($LASTEXITCODE) { throw 'Cleanup failed' }
& $blender -b --factory-startup --python-exit-code 1 --python scripts/week11_room_blender.py -- create "$work/design"
if ($LASTEXITCODE) { throw 'Blender creation failed' }
& $blender -b --disable-autoexec "$work/design/room_tabletop_edited.blend" --python-exit-code 1 `
    --python scripts/week11_room_blender.py -- verify "$work/design"
if ($LASTEXITCODE) { throw 'Blender reopen verification failed' }
~~~

To generate a paired B00 result, run `& $python -u examples/simple_trainer.py default @training --result-dir "$work/baseline"` using the training arguments above. Use a second fresh extraction output for its checkpoint. To add the timed measurements for Room, replace the evaluation list with `--eval-steps 1 1000 2000 3000 4000 5000 6000 7000` and append `--timed-eval-seconds 90 180 270` in both arms. On a faster machine a deadline can fall beyond the 7,000-update horizon. The available samples then end with the last update. Choose common deadlines within both runs for a new timed experiment.

| Stage | Verify |
| --- | --- |
| Training | `candidate/experiment.json` records 7,000 updates, the complete camera order and a finite checkpoint at `candidate/ckpts/final_post_update.pt`. `stats/val_step6999.json` has the final RGB scores and count. The candidate must stay under its recorded scheduled targets. |
| Extraction | `raw/report.json` records the checkpoint and PLY hashes, 64 fitting renders, a nonempty surface, and the grid. The two export formats are `raw_mesh.ply` and `raw_mesh.obj`. |
| Cleanup | `design/cleanup.json` and `design/room_cleaned.ply` record the exact source mesh, topology and output digest. `silhouette_cameras.json` identifies matched views and crops. |
| Editing | `design/room_tabletop_edited.blend` opens in Blender. `design/design_reopen.json` records a connected patch and measured +Z displacement near 0.040 normalized units. The saved file's digest must agree. |

Compare generated files with the evidence records using their digests to identify compatible inputs. Treat changes in last digits, exact primitive counts or checkpoint bytes as rerun differences to inspect. The report uses the verified fresh replay numbers separately from the original five-scene pairs. Reject incomplete runs, nonfinite splats, unexpected camera scheduling, empty TSDF volumes and failed reopen checks. For an empty grid, inspect the fitted camera convention, scale, alpha/depth support and actual grid before adjusting extraction settings. Use the logged run directory for diagnostics and choose a fresh destination for a rerun.

## Audited studies and document build

The Room study uses the archived October B00 Room directory to audit the released sequence, initializer and 39 target RGB hashes. It saves commands, source snapshots, checkpoints, per-run verification and a summary. The matte grid study uses the prepared local matte data, both original Week 12 checkpoints and the three reserved depth messages. These private project captures are referenced by their existing paths and hashes. The public Room pipeline above requires public benchmark data and the released camera sequence.

~~~powershell
.\scripts\run-school-completion.ps1 -u scripts/final_report_study.py --root results/room_study_replay
.\scripts\run-school-completion.ps1 -u scripts/final_report_study.py --tight-ranking --root results/room_ranking_replay
.\scripts\run-school-completion.ps1 -u scripts/final_report_geometry.py --root results/matte_grid_replay
& $python report/final_report/build_evidence.py
Push-Location report/final_report
tectonic final_report.tex
Pop-Location
~~~

The outdoor evaluation and Room seed replications use the system CUDA/Python environment configured above. Prepare Room at factor two and Garden at factor four, then run the frozen follow-up:

~~~powershell
& $python scripts/download-room-small.py --scene room --image-factor 2
& $python scripts/download-room-small.py --scene garden --image-factor 4
& $python -u scripts/final_report_followup.py --root results/report_followup_replay
~~~

The follow-up uses the 700,000 ceiling and the selected absolute-gradient policy as fixed in the report. The archived summary includes all completed paired runs and camera/target hashes. The evidence index at the supplement root maps the report's evidence IDs to bundled files.

Generated evidence tables and figures in this folder correspond to the verified project runs dated 19 and 20 November 2026. See the [experiment log](../../docs/EXPERIMENT_LOG.md) for archive locations, checks and measured limitations.
