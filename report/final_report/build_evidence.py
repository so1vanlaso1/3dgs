"""Typeset the verified evidence snapshots and render their report plots."""

import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
FIGURES = ROOT / "figures"
CANDIDATE_BLUE = "#166582"
BASELINE_GRAY = "#48505a"
AMBER = "#a65a1d"


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def rows(name, cells):
    lines = [" & ".join(row) + r" \\" for row in cells]
    (EVIDENCE / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def metrics(row):
    return [f"{row['psnr']:.3f}", f"{row['ssim']:.5f}", f"{row['lpips']:.5f}"]


def format_axis(axis, x, y):
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    axis.grid(axis="y", color="#d9e0e6", linewidth=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)


def save_plot(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"Creator": "Final report evidence"})
    slides = FIGURES / "slides"
    slides.mkdir(exist_ok=True)
    fig.savefig(slides / f"{name}.png", dpi=240, facecolor="white")
    plt.close(fig)


def room_time(room):
    baseline = room["B00"]["probes"]
    candidate = room["A05_absgrad"]["probes"]
    timed = []
    for left, right in zip(baseline["timed"], candidate["timed"], strict=True):
        assert left["target_seconds"] == right["target_seconds"]
        for index, (arm, row) in enumerate((("B00", left), ("A05-Abs", right))):
            timed.append([
                f"{row['target_seconds']:.0f}" if index == 0 else "", arm,
                f"{row['train_seconds']:.2f}", f"{row['completed_updates']:,}",
                *metrics(row), f"{row['num_GS']:,}",
            ])
    rows("room_time_rows.tex", timed)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), layout="constrained", sharey=True)
    for arm, probe, color, marker in (
        ("B00", baseline, BASELINE_GRAY, "s"), ("A05-Abs", candidate, CANDIDATE_BLUE, "o"),
    ):
        updates = probe["updates"][1:]
        axes[0].plot([row["completed_updates"] for row in updates],
                     [row["psnr"] for row in updates], marker=marker, color=color, label=arm,
                     linewidth=2.25, markersize=6)
        deadlines = probe["timed"]
        axes[1].plot([row["train_seconds"] for row in deadlines],
                     [row["psnr"] for row in deadlines], marker=marker, color=color,
                     linewidth=2.25, markersize=6)
    format_axis(axes[0], "Completed updates", "Mean PSNR (dB)")
    format_axis(axes[1], "Training time (s)", "")
    axes[0].set_xticks([1000, 3000, 5000, 7000], ["1,000", "3,000", "5,000", "7,000"])
    axes[1].set_xticks([90, 180, 270])
    axes[0].legend(loc="lower right", frameon=False)
    save_plot(fig, "room_progress")


def room_components(room):
    settings = (
        ("A04", "Both cues", "Signed", r"$2\times10^{-4}$"),
        ("signed_high", "Both cues", "Signed", r"$8\times10^{-4}$"),
        ("abs_low", "Both cues", "Absolute", r"$2\times10^{-4}$"),
        ("A05_absgrad", "Both cues (A05-Abs)", "Absolute", r"$8\times10^{-4}$"),
        ("no_error", "No RGB-error score", "Absolute", r"$8\times10^{-4}$"),
        ("no_edge", "No edge score", "Absolute", r"$8\times10^{-4}$"),
    )
    table = []
    for key, policy, gradient, threshold in settings:
        outcome = room[key]["metrics"]
        table.append([policy, gradient, threshold, *metrics(outcome), f"{outcome['num_GS']:,}"])
    rows("room_component_rows.tex", table)
    bands = (("cap400k", "Both scoring cues"), ("no_error_400k", "No RGB-error score"),
             ("no_edge_400k", "No edge score"))
    table = []
    for key, policy in bands:
        arm = room[key]
        row = arm["metrics"]
        table.append([policy, str(arm["binding_events"]), *metrics(row), f"{row['num_GS']:,}"])
    rows("room_ranking_rows.tex", table)


def room_caps(room):
    settings = (("cap400k", "400,000"), ("A05_absgrad", "700,000"), ("cap1000k", "1,000,000"))
    outcomes = [room[key]["metrics"] for key, _ in settings]
    table = [["A05-Abs", cap, f"{row['num_GS']:,}", *metrics(row)]
             for (_, cap), row in zip(settings, outcomes, strict=True)]
    baseline = room["B00"]["metrics"]
    rows("room_cap_rows.tex", [*table, ["B00", "Unrestricted", f"{baseline['num_GS']:,}", *metrics(baseline)]])

    fig, axis = plt.subplots(figsize=(7.8, 3.5), layout="constrained")
    points = (
        (outcomes[0], "400k cap", CANDIDATE_BLUE, "o", (0, -11), "center", "top"),
        (outcomes[1], "Selected A05-Abs", CANDIDATE_BLUE, "o", (0, 10), "center", "bottom"),
        (room["abs_low"]["metrics"], "Lower gradient threshold", AMBER, "D", (0, 10), "center", "bottom"),
        (baseline, "B00", BASELINE_GRAY, "s", (-10, 0), "right", "center"),
    )
    for row, label, color, marker, offset, ha, va in points:
        x, y = row["num_GS"], row["psnr"]
        axis.scatter(x, y, color=color, marker=marker, s=65, zorder=4)
        axis.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                      ha=ha, va=va, color=color, fontweight="bold")
    format_axis(axis, "Realized Gaussian count", "Mean PSNR (dB)")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    axis.set_xlim(340_000, 865_000)
    axis.set_ylim(30, 31.31)
    save_plot(fig, "room_cap_tradeoff")


def followup(room, pairs):
    arms = (("B00", "B00"), ("A05_absgrad", "A05-Abs"))
    room_42 = {
        "scene": "room", "seed": 42, "arms": {key: room[key] for key, _ in arms},
        "delta": {"psnr": room["A05_absgrad"]["metrics"]["psnr"] - room["B00"]["metrics"]["psnr"]},
        "gaussian_reduction_fraction": 1 - room["A05_absgrad"]["metrics"]["num_GS"] / room["B00"]["metrics"]["num_GS"],
    }
    replications = [room_42, *(pair for pair in pairs if pair["scene"] == "room")]
    garden = next(pair for pair in pairs if pair["scene"] == "garden")
    ordered = [*replications, garden]
    values_by_seed = [pair["delta"]["psnr"] for pair in replications]
    reductions = [100 * pair["gaussian_reduction_fraction"] for pair in replications]
    cells = []
    for pair in ordered:
        for index, (key, name) in enumerate(arms):
            entry = pair["arms"][key]
            result = entry["metrics"]
            cells.append([
                pair["scene"].title() if index == 0 else "",
                str(pair["seed"]) if index == 0 else "", name, *metrics(result),
                f"{result['num_GS']:,}", str(entry["binding_events"]) if index else "---",
            ])
    rows("followup_rows.tex", cells)
    (EVIDENCE / "room_seed_summary.tex").write_text(
        rf"The mean paired Room PSNR difference was \({mean(values_by_seed):+.3f}\pm{stdev(values_by_seed):.3f}\) dB "
        rf"(mean $\pm$ sample SD across three seeds). Gaussian count fell by {mean(reductions):.1f}\% on average, "
        rf"with a sample SD of {stdev(reductions):.1f} percentage points. "
        "These summaries describe repeatability on Room under this protocol.\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.45), layout="constrained", sharey=True)
    names = [f"{pair['scene'].title()}, s{pair['seed']}" for pair in ordered]
    data = (
        ([100 * pair["gaussian_reduction_fraction"] for pair in ordered], "Fewer Gaussians (%)", "{:.1f}"),
        ([pair["delta"]["psnr"] for pair in ordered], "PSNR change (dB)", "{:+.2f}"),
    )
    for axis, (values, title, number_format) in zip(axes, data, strict=True):
        bars = axis.barh(names, values, height=0.58, color=[CANDIDATE_BLUE] * len(replications) + [AMBER])
        axis.set_title(title, loc="left", fontweight="bold")
        axis.bar_label(bars, labels=[number_format.format(value) for value in values], padding=4)
        axis.axvline(0, color=BASELINE_GRAY, linewidth=0.8)
        axis.grid(axis="x", color="#d9e0e6", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.margins(x=0.24)
    axes[0].invert_yaxis()
    save_plot(fig, "paired_followup")


def geometry(data):
    depths, topology = [], []
    for variant, label in (("gaussian", "Gaussian ED"), (128, "Mesh 128"), (192, "Mesh 192"), (256, "Mesh 256")):
        for arm, name in (("B00", "B00"), ("A05_absgrad", "A05-Abs")):
            key = f"{arm}_{variant}"
            row = data["metrics"][key]
            signed = "$%.2f$" % (100 * row["median_signed_m"])
            depths.append([label, name, f"{row['coverage']:.2%}".replace("%", r"\%"),
                           signed, f"{100 * row['median_abs_m']:.2f}", f"{100 * row['p90_abs_m']:.2f}"])
            if key in data["variants"]:
                mesh = data["variants"][key]
                topology.append([str(variant), name, f"{mesh['triangles']:,}", f"{mesh['components']:,}",
                                 f"{mesh['boundary_edges']:,}",
                                 f"{mesh['largest_component_area_fraction']:.1%}".replace("%", r"\%")])
    rows("matte_grid_depth_rows.tex", depths)
    rows("matte_grid_topology_rows.tex", topology)


def main():
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    room = read("room_study.json")
    room_time(room)
    room_components(room)
    room_caps(room)
    geometry(read("matte_grid.json"))
    followup(room, read("followup.json"))


if __name__ == "__main__":
    main()
