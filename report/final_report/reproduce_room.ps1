param(
    [ValidateSet('All', 'Prepare', 'Train', 'Extract', 'Clean', 'Edit', 'Verify')]
    [string]$Stage = 'All',
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$Blender = 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe',
    [string]$Work = 'results/report_room_replay'
)

$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '../..')
try {
    $dataset = 'data/room_factor2'
    $fitOrder = 'report/final_report/evidence/room_camera_order.json'
    $candidate = Join-Path $Work 'candidate'
    $raw = Join-Path $Work 'raw'
    $design = Join-Path $Work 'design'
    $checkpoint = Join-Path $candidate 'ckpts/final_post_update.pt'
    $project = Join-Path $design 'room_tabletop_edited.blend'

    if ($Stage -eq 'All' -and (Test-Path -LiteralPath $Work)) {
        throw "Choose a fresh results directory. Already exists: $Work"
    }
    if ($Stage -in 'All', 'Prepare') {
        & $Python scripts/download-room-small.py --scene room --image-factor 2
        if ($LASTEXITCODE -ne 0) { throw 'Preparation failed' }
    }
    if ($Stage -in 'All', 'Train') {
        if (Test-Path -LiteralPath $candidate) { throw "Training output exists: $candidate" }
        & $Python -u examples/simple_trainer.py budget --seed 42 --independent-sampling --record-experiment `
            --data-dir $dataset --data-factor 1 --max-steps 7000 --eval-steps 1 7000 --save-steps 7000 `
            --save-ply --ply-steps 7000 --disable-viewer --disable-video --batch-size 1 --packed `
            --sh-degree 3 --sh-degree-interval 1000 --cache-images --data-workers 0 --camera-sequence-path $fitOrder `
            --strategy.refine-start-iter 500 --strategy.refine-stop-iter 6000 --strategy.refine-every 100 `
            --strategy.reset-every 3000 --strategy.absgrad --strategy.grow-grad2d 0.0008 --strategy.cap-max 700000 `
            --strategy.error-weight 1 --strategy.edge-weight 0.25 --strategy.split-mode long_axis `
            --strategy.clone-policy split --strategy.split-distance 0.45 --strategy.opacity-reduction 0.6 `
            --result-dir $candidate
        if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
    }
    if ($Stage -in 'All', 'Extract') {
        if (Test-Path -LiteralPath $raw) { throw "Extraction output exists: $raw" }
        & $Python -u scripts/gaussians_to_mesh.py --checkpoint $checkpoint --data $dataset --output $raw `
            --views 64 --downsample 4 --voxel 0.01 --max-resolution 192
        if ($LASTEXITCODE -ne 0) { throw 'Extraction failed' }
    }
    if ($Stage -in 'All', 'Clean') {
        if (Test-Path -LiteralPath $design) { throw "Cleanup output exists: $design" }
        & $Python -u scripts/week11_room_cleanup.py --source $raw --output $design
        if ($LASTEXITCODE -ne 0) { throw 'Cleanup failed' }
    }
    if ($Stage -in 'All', 'Edit') {
        if (Test-Path -LiteralPath $project) { throw "Blender output exists: $project" }
        & $Blender -b --factory-startup --python-exit-code 1 --python scripts/week11_room_blender.py -- create $design
        if ($LASTEXITCODE -ne 0) { throw 'Blender creation failed' }
    }
    if ($Stage -in 'All', 'Verify') {
        & $Blender -b --disable-autoexec $project --python-exit-code 1 --python scripts/week11_room_blender.py -- verify $design
        if ($LASTEXITCODE -ne 0) { throw 'Blender reopen verification failed' }
    }
}
finally {
    Pop-Location
}
