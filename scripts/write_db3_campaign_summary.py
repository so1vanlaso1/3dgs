"""Publish actual campaign decisions after verification, preserving failed gates."""
import importlib.util
from pathlib import Path

s=importlib.util.spec_from_file_location('local',Path(__file__).with_name('db3_local_geometry.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def main():
    integrity=m.read(m.RUN/'integrity_audit.json');delivery=m.read(m.RUN/'local_delivery.json')
    results={r:m.read(m.RUN/r/'pilot/paired_evaluation.json') for r in ('matte','exhibit')}
    extra=m.read(m.RUN/'exhibit/pilot/paired_evaluation_supported_15k.json')
    extra_check=m.read(m.RUN/'exhibit/pilot/selected_policy_confirmation_result_supported_15k.json')
    accepted={r:v['declared_local_followup_gate_pass'] for r,v in delivery['submaps'].items()}
    decision=dict(R0=dict(complete=True,baseline_preserved=True),
        R1=dict(engineering_complete=True,numerical_gate=True,precise_physical_calibration_complete=False),
        R2=dict(bounded_local_geometry_and_D8_experiment_complete=True,local_submaps_checked=True,whole_recording_poses_accepted=False),
        R3=dict(common_7000_matrix_complete=True,matte_local_appearance_consistency_followup=accepted['matte'],
                exhibit_supported_15000_followup=accepted['exhibit'],original_universal_depth_gain_claim=False,
                confidence_policy_superiority_established=False),
        source_baseline_R1_integrity=True,selected_valid_7000_updates=42000,additional_supported_updates=15000,
        preserved_invalid_job_updates=7000,tests='246 passed; one existing unknown pytest env warning',
        all_R0_R3_acceptance_gates_finished_cleanly=False,
        remaining=['Precise independent physical alignment/calibration','Distributed physically verified whole-recording constraints or uncertain separate submaps',
                   *([] if accepted['exhibit'] else ['Difficult exhibit structure/coverage/appearance acceptance'])])
    m.write(m.RUN/'phase_decision.json',decision)
    lines=['The final reload/integrity audit passed. The full raw SHA256 remains',
        f'`{integrity["raw_sha256_unchanged"]}`. Baseline originals/copies,',
        'all 477 baseline RGB/depth identities, all R1 input hashes and every fresh local',
        'input hash match. Native/historical reservations and spatial feature exclusions',
        'are verified. Six valid 7k checkpoints plus the additional 15k checkpoint are',
        'finite and within the cap. Original 7k initializations and full replays match;',
        'the 15k candidate matches initialization and the exact first 7k replay.',
        'The rejected unsafe-depth job remains explicitly preserved and excluded.',
        '', '### Difficult-region 7k result', '', '| Arm | PSNR | SSIM | LPIPS |','|---|---:|---:|---:|']
    for arm,label in [('baseline','Preserved contextual baseline'),('off','Depth off'),('current','Repaired current'),('confidence','Confidence')]:
        rec=results['exhibit']['results'][arm]['mean_appearance'];lines.append(f'| {label} | {rec["psnr"]:.3f} | {rec["ssim"]:.5f} | {rec["lpips"]:.5f} |')
    rec=extra['results']['confidence']['mean_appearance']
    lines += ['', 'The 7k exhibit confirmation fails: improved SSIM/LPIPS and several visible',
        'details do not offset failed PSNR, coverage and structure checks. No promotion.',
        '', '### Additional bounded 15k support/coverage experiment', '',
        f'Final appearance: **{rec["psnr"]:.3f} dB / {rec["ssim"]:.5f} SSIM / {rec["lpips"]:.5f} LPIPS**.',
        'Same fixed geometry/cloud/IDs/seed and 7k replay prefix; depth lambda 0.05,',
        'supported-sensor alpha lambda 0.02; 180k cap, refinement still stops at 6k.',
        'Missing/zero-confidence observations get no alpha prior. This is an exposed',
        'follow-up after a failed confirmation and jointly changes objective/compute;',
        'it cannot establish policy superiority over 7k controls.', '',
        f'Coarse quantitative gate: **{extra_check["coarse_quantitative_pass"]}**. '
        f'Actual statue/post/wall/floor gate: **{extra_check["manual_structure_quantitative_pass"]}**. '
        f'Overall visual-and-quantitative local acceptance: **{accepted["exhibit"]}**.', '',
        'Per-view source coverage, RGB gains, median/p90 depth residuals and every',
        'rejection are retained in the paired/confirmation reports. Thresholds were',
        'not loosened. The original 7k failures remain. The 15k exhibit candidate passes',
        'the original coarse pipeline depth/appearance gate, but fails the stronger',
        'source-specific structures and first-view target appearance checks. Matte',
        'still fails the original all-view depth-gain requirement. No universal gain.',
        '', '### Actual phase status', '', '| Phase | Verified outcome | Remaining limit |', '|---|---|---|',
        '| R0 | Complete; frozen baseline/source preserved | Baseline is context, not geometric truth |',
        '| R1 | Numerical projection/input/timing deliverables complete | Precise physical calibration remains partial |',
        '| R2 | Checked local submaps and concrete D8 separate-submap experiment complete | No accepted whole-recording merge/trajectory |',
        f'| R3 | Six valid 7k controls + bounded 15k follow-up; matte appearance/consistency accepted; exhibit acceptance {accepted["exhibit"]} | No universal depth improvement or confidence-policy superiority; external geometry accuracy unestablished |',
        '', '**Do not mark every R0–R3 scientific acceptance gate closed.** Execution,',
        'failure handling and artifact integrity are complete; the table distinguishes',
        'the demonstrated local result from unresolved reconstruction requirements.',
        '', '### Local artifacts and reproduction', '',
        '- [Machine phase decision](../results/db3_r2_r3_20261110/phase_decision.json).',
        '- [Final integrity/source-code audit](../results/db3_r2_r3_20261110/integrity_audit.json).',
        '- [Local checkpoint/dataset/camera/viewer index](../results/db3_r2_r3_20261110/local_delivery.json).',
        '- [Matte captured/baseline/selected comparison](../results/db3_r2_r3_20261110/matte/pilot/selected_comparison.jpg).',
        '- [Exhibit 7k control matrix](../results/db3_r2_r3_20261110/exhibit/pilot/paired_evaluation.json).',
        '- [Exhibit 15k paired result](../results/db3_r2_r3_20261110/exhibit/pilot/paired_evaluation_supported_15k.json).',
        '- [Exhibit 15k fixed-structure decision](../results/db3_r2_r3_20261110/exhibit/pilot/selected_policy_confirmation_result_supported_15k.json).',
        '', 'Every pilot folder includes exact commands, configuration, experiment/replay',
        'hashes, final checkpoint, metrics and logs. Source snapshots are in the campaign',
        '`source_snapshot/`. Local camera provenance records normalization, metric units,',
        'K, every source ID and actual timestamp replay. No global relative placement is',
        'invented. Replay was checked offline at start/mid/end; browser interaction and',
        'collision-free walking are not newly certified.', '',
        'Inspect the supported matte candidate using the existing viewer launcher:', '',
        '```powershell',
        '.\\scripts\\view-clean-realsense.ps1 -ResultDir results/db3_r2_r3_20261110/matte/pilot -Port 8092 -Background',
        '```', '',
        'Fresh stage order (use a new versioned namespace; existing evidence is refused):',
        '`db3_local_geometry.py prepare`, four `map` runs, four `db3_validate_local.py`',
        'checks, `db3_transition_revisit.py`, two `db3_local_pilot.py prepare` runs,',
        'the three-arm `run-db3-local-pilot.ps1` matrices, paired evaluation and explicit',
        'confirmation. The optional 15k candidate uses `-Arms confidence_supported_15k',
        '-Steps 15000`; its evaluation uses `-Action evaluate -CandidateDir',
        'confidence_supported_15k -Steps 15000`. Keep failed evidence and gates.', '',
        'No package upgrade, full-recording 120k retraining, mesh production, commit or',
        'push occurred. Unrelated workspace modifications remain preserved.']
    doc=m.ROOT/'docs/DB3_R2_R3_RESULTS_20261110.md';text=doc.read_text(encoding='utf-8')
    old='Execution is in progress. See campaign status and the experiment log for current\nfacts; this section is completed only after the final evaluation and integrity\naudit, without changing the original failed gates.'
    assert old in text;doc.write_text(text.replace(old,'\n'.join(lines)),encoding='utf-8')
    print('PHASE DECISION',decision,flush=True)


if __name__=='__main__':main()
