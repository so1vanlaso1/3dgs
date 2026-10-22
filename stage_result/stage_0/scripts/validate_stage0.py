"""Validate the Stage 0 artifact package without training or network access."""
import csv
import hashlib
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []
    docs = list(OUT.glob("*.md"))
    missing = []
    for doc in docs:
        content = doc.read_text(encoding="utf-8")
        assert "\ufffd" not in content, f"Replacement characters in {doc.name}"
        for target in re.findall(r"\]\(([^)]+)\)", content):
            target = target.split("#", 1)[0]
            if not target or target.startswith(("https://", "http://", "mailto:")):
                continue
            generated_here = target in {"evidence/validation_report.json", "manifests/artifact_manifest.json"}
            if not (doc.parent / target).exists() and not generated_here:
                missing.append({"document": doc.name, "target": target})
    assert not missing, missing
    checks.append({"check": "markdown_local_links_and_utf8", "passed": True, "documents": len(docs)})
    for scene, expected in [("analytic_spheres_v1", {"train": 36, "development": 6, "test": 6}),
                            ("room_small", {"train": 233, "development": 39, "test": 39})]:
        manifest = json.loads((OUT / "manifests" / f"{scene}_splits.json").read_text())
        with (OUT / "manifests" / f"{scene}_views.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert manifest["counts"] == expected
        assert len(rows) == sum(expected.values())
        assert len({r["view_id"] for r in rows}) == len(rows)
        indices = [i for role in expected for i in manifest["parser_indices"][role]]
        assert sorted(indices) == list(range(len(rows)))
        data = ROOT / manifest["data_path_relative_to_repo"]
        for row in rows:
            i = int(row["sorted_index"])
            role = "test" if i % 8 == 0 else "development" if i % 8 == 4 else "train"
            assert row["split"] == role
            assert row["view_id"] in manifest["view_ids"][role]
            assert i in manifest["parser_indices"][role]
            assert sha(data / "images" / row["image_name"]) == row["image_sha256"]
        checks.append({"check": f"{scene}_identity_partition_and_pixel_hashes", "passed": True, "views": len(rows)})
    audit = json.loads((OUT / "evidence" / "data_audit.json").read_text())
    for scene in audit["scenes"]:
        assert scene["all_images_decodable"] and scene["all_camera_values_finite"] and scene["all_point_values_finite"]
        assert scene["max_rotation_orthogonality_error"] < 1e-10
        assert scene["max_rotation_determinant_error"] < 1e-10
        assert scene["max_inverse_identity_error"] < 1e-10
        assert scene["train_dev_loading_verified"]
        if scene["scene"] == "analytic_spheres_v1":
            assert scene["camera_truth_export_max_error"] < 1e-10
    for key in ["prepared_poses_max_difference_from_source", "prepared_points_max_difference_from_source", "rescaled_intrinsics_max_difference"]:
        assert audit["room_source_check"][key] < 1e-10
    checks.append({"check": "recorded_camera_truth_and_room_source_equivalence", "passed": True})
    sources = json.loads((OUT / "evidence" / "source_audit.json").read_text())
    required = {"revising", "fastgs", "salientgs", "energygs", "bsgs", "harmogs", "blitzgs"}
    assert required.issubset({p["id"] for p in sources["papers"] if p.get("status") == 200})
    repos = {r["repository"]: r for r in sources["repositories"]}
    for name in ["SkylerGao/ENGS", "wsxujm/bsgs"]:
        assert repos[name]["file_count"] == 2 and repos[name]["code_file_count"] == 0
    assert all(not r.get("tree_truncated") for r in repos.values())
    checks.append({"check": "named_source_coverage_and_recorded_release_inventory", "passed": True})
    decisions = json.loads((OUT / "manifests" / "study_decisions.json").read_text())
    assert decisions["status"] == "prospective_no_ccd_results"
    assert decisions["scope"]["pose_parameters"] == ["rotation", "translation"]
    assert decisions["primary_comparator"] == "ordinary_density_control_with_common_pose_refinement"
    assert decisions["quality_margins"]["psnr_loss_db_max"] == 0.20
    checks.append({"check": "machine_readable_primary_comparison_and_claim_status", "passed": True})
    required_files = ["README.md", "research_brief.md", "study_specification.md", "related_work_matrix.md",
        "source_verification_notes.md", "data_and_splits.md", "decision_criteria.md", "introduction_outline.md",
        "stage_1_handoff.md", "EXPERIMENT_LOG.md", "references.bib"]
    assert all((OUT / f).exists() for f in required_files)
    report = {"date_local": "2026-10-10", "timezone": "Asia/Saigon", "status": "passed",
        "checks": checks, "markdown_word_count": sum(len(p.read_text(encoding="utf-8").split()) for p in docs),
        "scope": "artifact integrity and recorded data checks; no model training or external reproduction",
        "fixture_reproducibility": json.loads((OUT / "evidence" / "fixture_reproducibility.json").read_text())}
    assert report["fixture_reproducibility"]["all_fixture_files_identical"]
    dest = OUT / "evidence" / "validation_report.json"
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Avoid self-hash recursion; include every other deliverable.
    artifacts = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json" and "__pycache__" not in p.parts]
    (OUT / "manifests" / "artifact_manifest.json").write_text(json.dumps({"date_local": "2026-10-10",
        "files": artifacts, "self_excluded": "manifests/artifact_manifest.json"}, indent=2) + "\n", encoding="utf-8")
    assert (OUT / "evidence" / "validation_report.json").exists()
    assert (OUT / "manifests" / "artifact_manifest.json").exists()
    print(json.dumps({"status": "passed", "checks": len(checks), "documents": len(docs),
        "words": report["markdown_word_count"], "artifacts": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
