"""Independently recompute the reported G4 depth statistics from saved arrays."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from db3_g4 import BASE, G3, MODES, REFERENCES, STRONG
from week12_comparison import read, sha, write


def validate(row, errors, covered, stable):
    assert row["error"]["pixels"] == errors.size
    assert np.isclose(row["coverage"], covered / stable, atol=1e-12, rtol=0)
    actual = (np.median(errors), np.median(abs(errors)), np.percentile(abs(errors), 90))
    expected = [row["error"][k] for k in ("median_signed_m","median_abs_m","p90_abs_m")]
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-7)


def figure(out, scores, views):
    arms = list(scores["counts"])
    focus = "rgb_0010545.png"
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4), layout="constrained")
    colors = ["#535d66","#277497","#e19e49","#4f9267","#865f91"]
    for i,arm in enumerate(arms):
        for j,kind in enumerate(("render","mesh")):
            view = scores["all_arm"][kind]["per_view"][focus]["arms"][arm]
            axes[0].bar(i+(j-.5)*.35, 100*view["error"]["p90_abs_m"], .34,
                        color=["#517995","#ba804e"][j],label=["Render","Mesh"][j] if i == 0 else None)
        if arm != BASE:
            for j,view in enumerate(views):
                rows = scores["versus_A"]["render"][arm]["per_view"][view]["arms"]
                delta = rows[arm]["rgb"]["psnr_db"] - rows[BASE]["rgb"]["psnr_db"]
                axes[1].scatter(i,delta,marker=["o","s","^"][j],color=["#517995","#ba804e","#55525f"][j],
                                label=Path(view).stem[4:] if i == 1 else None)
        if arm not in REFERENCES:
            snapshot = read(out/f"{arm}_recovery.json")["snapshots"]
            steps = (3000,3001,4500,7000)
            cover = [100*snapshot[str(s)]["development"]["pooled"][arm]["coverage"] for s in steps]
            axes[2].plot(range(4),cover,marker=["o","s","^"][i-2],mfc="none",color=colors[i],label=arm)
    axes[0].set(title="Tail error on view 0010545",ylabel="Absolute p90 (cm)",xticks=range(5),xticklabels=arms)
    axes[1].axhline(-.5,linestyle="--",color=".35",linewidth=1)
    axes[1].set(title="RGB change on each view",ylabel="PSNR difference from G4-A (dB)",
                xticks=range(1,5),xticklabels=arms[1:])
    axes[2].set(title="Visible depth after opacity reset",ylabel="Coverage of stable sensor pixels (%)",
                xlabel="Completed updates (sampled checkpoints)",xticks=range(4),
                xticklabels=["3000","3001","4500","7000"],ylim=(-4,105))
    for ax in axes:
        ax.legend(frameon=False,fontsize=8)
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(out/"g4_comparison.png",dpi=160)
    plt.close(fig)


def readback(out):
    scores = read(out/"final_scores.json")
    arms = list(scores["counts"])
    views = list(scores["all_arm"]["render"]["per_view"])
    maps = {}
    for name in views:
        with np.load(out/f"{Path(name).stem}_final.npz") as layers:
            maps[name] = dict(layers)
        with np.load(G3/f"evaluation/{Path(name).stem}_final.npz") as original:
            np.testing.assert_array_equal(maps[name]["sensor"],original["sensor_m"])
            np.testing.assert_array_equal(maps[name]["stable"],original["stable"])
            for g4,g3 in REFERENCES.items():
                for kind in ("render","mesh"):
                    np.testing.assert_array_equal(maps[name][f"{g4}_{kind}_valid"],original[f"{g3}_{kind}_valid"])
                    np.testing.assert_allclose(maps[name][f"{g4}_{kind}"],original[f"{g3}_{kind}_depth"],
                                               rtol=1e-5,atol=1e-6)
    checks = 0
    for kind in ("render","mesh"):
        pairs = [(scores["all_arm"][kind],arms)]
        pairs.append((scores["dose_pair"][kind],[read(out/"selection.json")["selected"],"G4-E"]))
        for table,anchor in (("versus_A",BASE),("versus_B",STRONG)):
            pairs.extend((comparison,[anchor,arm]) for arm,comparison in scores[table][kind].items())
        for summary, subset in pairs:
            residuals = {arm:[] for arm in subset}
            coverage = {arm:0 for arm in subset}
            total = 0
            for name,arr in maps.items():
                stable = arr["stable"]
                masks = {arm:stable & arr[f"{arm}_{kind}_valid"] for arm in subset}
                joint = np.logical_and.reduce(list(masks.values()))
                assert int(joint.sum()) == summary["per_view"][name]["shared_pixels"]
                total += int(stable.sum())
                for arm in subset:
                    z = arr[f"{arm}_{kind}"]
                    if kind == "render":
                        np.testing.assert_array_equal(arr[f"{arm}_render_valid"],
                                                      (z > 0) & (arr[f"{arm}_alpha"] > .7))
                    else:
                        np.testing.assert_array_equal(arr[f"{arm}_mesh_valid"],np.isfinite(z) & (z > 0))
                    error = z[joint] - arr["sensor"][joint]
                    covered = int(masks[arm].sum())
                    validate(summary["per_view"][name]["arms"][arm],error,covered,int(stable.sum()))
                    residuals[arm].append(error)
                    coverage[arm] += covered
                    checks += 1
            for arm in subset:
                validate(summary["pooled"][arm],np.concatenate(residuals[arm]),coverage[arm],total)
                checks += 1
    for arm in arms:
        run = G3/REFERENCES[arm] if arm in REFERENCES else out/arm
        stats = read(run/"stats/val_step6999.json")
        rgb = scores["all_arm"]["render"]["pooled"][arm]["rgb"]
        np.testing.assert_allclose([stats["psnr"],stats["ssim"]],[rgb["psnr_db"],rgb["ssim"]],atol=1e-4)
        assert stats["num_GS"] == scores["counts"][arm]
    figure(out,scores,views)
    artifacts = sorted(p for p in out.iterdir() if p.is_file() and p.suffix in {".json",".npz",".png"}
                       and p.name != "readback.json")
    write(out/"readback.json",dict(verified_depth_rows=checks,matching_control_maps=True,
          all_arm_rgb_and_counts=True,artifacts_sha256={p.name:sha(p) for p in artifacts}))
    print(f"Verified {checks} depth and coverage rows, reference maps, final RGB and counts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    readback(parser.parse_args().output.resolve())
