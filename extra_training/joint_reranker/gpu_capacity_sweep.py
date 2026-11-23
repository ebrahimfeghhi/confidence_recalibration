"""Back-to-back GPU capacity sweep inside ONE existing GPU allocation.

The sweep launches trials sequentially and never releases the outer scheduler
allocation; each child subprocess exits cleanly after its trial, so CUDA OOM
doesn't poison the following crop-size test. No standalone/synthetic PAE
tensor benchmarks are accepted as measured safe training limits.

Usage is documented in README.md. The command template must launch the REAL
training step through gpu_probe_step.py and use {crop}, {poses}, {case}, {steps},
{prefix}; it can use torchrun with all allocated GPUs.
"""

import argparse
import csv
import glob
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

DEFAULT_GRIDS = {
    256: (4, 8, 12, 16, 20, 24),
    384: (2, 4, 6, 8, 10, 12),
    512: (2, 3, 4, 5, 6, 8),
}
OOM_PATTERN = re.compile(r"(?:out of memory|cublas_status_alloc_failed|cudnn_status_alloc_failed|cuda error.*memory)",
                         re.IGNORECASE)


def parse_grids(spec: str) -> dict[int, tuple[int, ...]]:
    """'256:4,8,12;384:2,4,6;512:2,4,6'."""
    grids = {}
    for group in spec.split(";"):
        if not group.strip():
            continue
        cs, sep, ks = group.partition(":")
        if not sep:
            raise ValueError("grid must look like '256:4,8;384:2,4'")
        crop = int(cs)
        poses = tuple(sorted(set(int(x) for x in ks.split(",") if x.strip())))
        if crop <= 0 or not poses or poses[0] < 2:
            raise ValueError("crop must be positive and each group must have >=2 poses")
        grids[crop] = poses
    if not grids:
        raise ValueError("grid cannot be empty")
    return grids


def load_rank_reports(prefix: Path, ranks: int, crop: int, k: int,
                      case: str, steps: int) -> list[dict]:
    files = sorted(glob.glob(f"{prefix}.rank*.json"))
    reports = [json.loads(Path(p).read_text()) for p in files]
    if len(reports) != ranks or sorted(r["rank"] for r in reports) != list(range(ranks)):
        raise ValueError(f"expected {ranks} rank reports, found {len(reports)}: {files}")
    for report in reports:
        if (report.get("status") != "ok" or report.get("world") != ranks
                or report.get("crop_size") != crop or report.get("poses") != k
                or report.get("case") != case or report.get("steps", 0) < steps
                or not report.get("head_grad_live") or not report.get("reranker_grad_live")):
            raise ValueError(f"invalid/incomplete REAL training report for rank {report.get('rank')}")
        for field in ("initial_free_bytes", "total_memory_bytes",
                      "peak_allocated_bytes", "peak_reserved_bytes", "max_step_seconds"):
            if field not in report or report[field] <= 0:
                raise ValueError(f"rank {report['rank']} lacks valid {field}")
        if report["peak_allocated_bytes"] > report["peak_reserved_bytes"]:
            raise ValueError("allocated memory cannot exceed reserved peak")
        if report["peak_reserved_bytes"] > report["total_memory_bytes"]:
            raise ValueError("reserved memory cannot exceed physical device memory")
    return reports


def save_results(path: Path, rows: list[dict], limits: dict, margin: float,
                 dry_run: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "report.json").write_text(json.dumps({
        "dry_run": dry_run,
        "safety_margin": margin,
        "trials": rows,
        "safe_pose_limits": limits,
    }, indent=2) + "\n")
    cols = ("crop", "poses", "case", "status", "min_free_before_gib",
            "max_peak_reserved_gib", "max_step_seconds", "detail")
    with (path / "trials.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})
    md = ["# Real-training GPU capacity sweep", "",
          f"Safety margin: {margin:.0%} of *initial available* GPU memory per rank.",
          "A passing trial requires all DDP ranks, live gradients into the head and",
          "reranker, and >=2 real forward/backward/optimizer steps. No synthetic results.",
          "", "| crop | pose group | case | result | peak reserved (worst rank) |",
          "|---|---:|---|---|---:|"]
    for row in rows:
        mem = f"{row['max_peak_reserved_gib']:.1f} GiB" if row.get("max_peak_reserved_gib") else "—"
        md.append(f"| {row['crop']} | {row['poses']} | {row['case']} | {row['status']} | {mem} |")
    md.extend(["", "## Largest validated group per crop", "",
               *(f"- {crop}: {k} poses" for crop, k in limits.items()),
               "", "These are measured training-memory bounds, NOT a validation of",
               "the ranking objective, crop correctness, or launch-ready integration."])
    (path / "report.md").write_text("\n".join(md) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--command-template", required=True,
                    help="torchrun command with {crop}, {poses}, {case}, {steps}, {prefix}")
    ap.add_argument("--out-dir", default="training/joint_reranker_capacity")
    ap.add_argument("--grids", default="256:4,8,12,16,20,24;384:2,4,6,8,10,12;512:2,3,4,5,6,8")
    ap.add_argument("--cases", default="worst",
                    help="comma-separated input cases supplied by the REAL probe adapter")
    ap.add_argument("--ranks", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--safety-margin", type=float, default=0.15)
    ap.add_argument("--dry-run", action="store_true",
                    help="print planned commands; do NOT claim a measured capacity")
    a = ap.parse_args()
    if a.ranks <= 0 or a.steps < 2 or not 0 < a.safety_margin < 0.5:
        ap.error("ranks>0, steps>=2 and safety margin in (0,0.5) are required")
    template_fields = ("{crop}", "{poses}", "{case}", "{steps}", "{prefix}")
    if not all(f in a.command_template for f in template_fields):
        ap.error("command template must contain ALL placeholders " + ", ".join(template_fields))
    out_dir = Path(a.out_dir)
    grids = parse_grids(a.grids)
    cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    if not cases:
        ap.error("--cases cannot be empty")

    rows = []
    limits = {}
    for crop, poses in grids.items():
        safe = 0
        for k in poses:
            # Every case for this K must pass. One OOM/insufficient margin stops
            # higher K for this crop, then CONTINUES with the next crop.
            stage_ok = True
            for case in cases:
                prefix = out_dir / "trials" / f"crop{crop}_k{k}_{case}"
                prefix.parent.mkdir(parents=True, exist_ok=True)
                for old in glob.glob(f"{prefix}.rank*.json"):
                    Path(old).unlink()  # forbid stale reports masquerading as new runs
                command = a.command_template.format(
                    crop=crop, poses=k, case=case, steps=a.steps, prefix=str(prefix)
                )
                argv = shlex.split(command)
                print(f"PROBE crop={crop} poses={k} case={case}: {' '.join(argv)}", flush=True)
                if a.dry_run:
                    rows.append({"crop": crop, "poses": k, "case": case, "status": "DRY_RUN",
                                 "detail": command})
                    continue

                with Path(f"{prefix}.log").open("w") as log:
                    proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                          check=False)
                log_text = Path(f"{prefix}.log").read_text(errors="replace")
                error_reports = []
                for p in glob.glob(f"{prefix}.rank*.json"):
                    try:
                        error_reports.append(json.loads(Path(p).read_text()))
                    except (OSError, json.JSONDecodeError):
                        pass
                oom = OOM_PATTERN.search(log_text) or any(
                    v.get("status") == "oom" for v in error_reports
                )
                row = {"crop": crop, "poses": k, "case": case}
                if proc.returncode != 0:
                    row.update(status="OOM" if oom else "ERROR",
                               detail=f"exit={proc.returncode}; log={prefix}.log")
                    stage_ok = False
                    rows.append(row)
                    save_results(out_dir, rows, limits, a.safety_margin, False)
                    if not oom:
                        print(f"Non-OOM integration failure: {prefix}.log", file=sys.stderr)
                        return 2
                    break

                try:
                    reports = load_rank_reports(prefix, a.ranks, crop, k, case, a.steps)
                except (KeyError, TypeError, ValueError, OSError) as e:
                    row.update(status="ERROR", detail=f"invalid report: {e}")
                    rows.append(row)
                    save_results(out_dir, rows, limits, a.safety_margin, False)
                    print(f"Invalid trial, cannot claim capacity: {e}", file=sys.stderr)
                    return 2

                peak = max(v["peak_reserved_bytes"] for v in reports)
                free = min(v["initial_free_bytes"] for v in reports)
                step_secs = max(v["max_step_seconds"] for v in reports)
                has_margin = all(
                    v["peak_reserved_bytes"] <= (1.0 - a.safety_margin) * v["initial_free_bytes"]
                    for v in reports
                )
                row.update(status="PASS" if has_margin else "LOW_HEADROOM",
                           min_free_before_gib=round(free / (1024**3), 2),
                           max_peak_reserved_gib=round(peak / (1024**3), 2),
                           max_step_seconds=round(step_secs, 3),
                           detail=f"worst rank memory ratio={max(v['peak_reserved_bytes'] / v['initial_free_bytes'] for v in reports):.3f}")
                rows.append(row)
                print(f"  {row['status']} peak={row['max_peak_reserved_gib']} GiB, "
                      f"worst step={row['max_step_seconds']} s", flush=True)
                save_results(out_dir, rows, limits, a.safety_margin, False)
                if not has_margin:
                    stage_ok = False
                    break
            if not a.dry_run and not stage_ok:
                break
            if not a.dry_run:
                safe = k
                limits[str(crop)] = safe
        if not a.dry_run and safe == 0:
            limits[str(crop)] = None
        save_results(out_dir, rows, limits, a.safety_margin, a.dry_run)

    print("SAFE GROUP LIMITS (None means even smallest tested group not validated):", limits, flush=True)
    if a.dry_run:
        print("DRY RUN ONLY; no CUDA memory was measured.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
