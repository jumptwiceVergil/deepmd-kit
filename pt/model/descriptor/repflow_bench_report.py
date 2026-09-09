"""Three-table reporting and sequence-number based autograd attribution."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(values):
    values = sorted(values)
    if not values:
        return {"samples": 0, "median_ms": None, "p90_ms": None, "min_ms": None}
    return {"samples": len(values), "median_ms": statistics.median(values),
            "p90_ms": values[min(len(values)-1, math.ceil(.9*len(values))-1)],
            "min_ms": values[0]}


def ancestry(event):
    while event is not None:
        yield event
        event = event.cpu_parent


def marker(event):
    for parent in ancestry(event):
        if parent.name.startswith("rfbench|"):
            _, stage, op, label = parent.name.split("|", 3)
            return stage, op, label
    return None


def attributed_kernels(events):
    """Return OWN correlated kernels only (never inclusive parent CUDA totals).

    First-backward-created forward ops carry sequence numbers. Their matching
    backward events during loss.backward inherit double_backward attribution.
    Thread is part of the key because sequence counters are thread-local.
    """
    sources = {}
    resolved = {}
    events = sorted(events, key=lambda e: e.time_range.start)

    def resolve(event):
        for parent in ancestry(event):
            label = marker(parent)
            if label is not None:
                return label
            if parent.name.startswith("autograd::engine::evaluate_function"):
                key = (getattr(parent, "fwd_thread", 0), parent.sequence_nr)
                source = sources.get(key)
                if source:
                    stage, op, tag = source
                    return ("double_backward" if stage == "backward" else "ordinary_backward", op, tag)
        return None

    for event in events:
        label = resolve(event)
        resolved[id(event)] = label
        # fwd_thread == 0 distinguishes forward ops from their backward events.
        if (label and event.sequence_nr >= 0 and getattr(event, "fwd_thread", 0) == 0
                and not event.name.startswith("autograd::")):
            sources[(event.thread, event.sequence_nr)] = label

    records = defaultdict(lambda: {"gpu_ms": 0.0, "launches": 0})
    total = 0.0
    for event in events:
        if not getattr(event, "kernels", None):
            continue
        label = resolved[id(event)] or ("unattributed", "other", "other")
        factory = next((p.name.split("|", 1)[1] for p in ancestry(event)
                        if p.name.startswith("rfbench_kernel|")), "torch_or_runtime")
        for kernel in event.kernels:
            key = (*label, factory, kernel.name)
            records[key]["gpu_ms"] += kernel.duration / 1000.0
            records[key]["launches"] += 1
            total += kernel.duration / 1000.0
    result = [{"stage": key[0], "op": key[1], "layer_call": key[2],
               "factory": key[3], "kernel": key[4],
               "activity": ("memcpy" if "memcpy" in key[4].lower()
                            else "memset" if "memset" in key[4].lower() else "kernel"), **value}
              for key, value in records.items()]
    return result, total


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)


def render_reports(directory, runs):
    directory = Path(directory)
    stages, attribution, resources = [], [], []
    for run in runs:
        version, frame = run["version"], run["frame"]
        valid = run.get("correctness", {}).get("pass", False) and not run.get("error")
        for stage, values in run.get("timings", {}).items():
            stages.append({"version": version, "frame": frame, "op": "training",
                           "layer_call": "all", "stage": stage,
                           "metric": "cuda_event_interval", "correctness_pass": valid,
                           **summarize(values)})
        for stage, values in run.get("wall_timings", {}).items():
            stages.append({"version": version, "frame": frame, "op": "training",
                           "layer_call": "all", "stage": stage,
                           "metric": "host_wall_including_phase_sync", "correctness_pass": valid,
                           **summarize(values)})
        grouped = defaultdict(list)
        for sample_id, sample in enumerate(run.get("profiles", [])):
            sample_groups = defaultdict(float)
            for row in sample:
                attribution.append({"version": version, "frame": frame,
                                    "profile_sample": sample_id, **row})
                sample_groups[(row["op"], row["layer_call"], row["stage"])] += row["gpu_ms"]
            for key, value in sample_groups.items():
                grouped[key].append(value)
        for (op, layer, stage), values in grouped.items():
            stages.append({"version": version, "frame": frame, "op": op,
                           "layer_call": layer, "stage": stage,
                           "metric": "profile_attributed_kernel_sum", "correctness_pass": valid,
                           **summarize(values)})
        # Sum measured calls within each profiling sample, not median * nlayers.
        totals = defaultdict(list)
        for sample in run.get("profiles", []):
            groups = defaultdict(float)
            for row in sample:
                groups[(row["op"], row["stage"])] += row["gpu_ms"]
            for key, value in groups.items():
                totals[key].append(value)
        for (op, stage), values in totals.items():
            stages.append({"version": version, "frame": frame, "op": op,
                           "layer_call": "all", "stage": stage,
                           "metric": "profile_attributed_kernel_sum", "correctness_pass": valid,
                           **summarize(values)})
        counts = defaultdict(lambda: defaultdict(list))
        for sample in run.get("profiles", []):
            sample_counts = defaultdict(lambda: defaultdict(int))
            for row in sample:
                if row["op"] == "other":
                    continue
                sample_counts[(row["op"], row["stage"])][row.get("activity", "kernel")] += row["launches"]
            for key, value in sample_counts.items():
                for activity in ("kernel", "memset", "memcpy"):
                    counts[key][activity].append(value[activity])
        for (op, stage), values in counts.items():
            resources.append({"version": version, "frame": frame, "op": op, "stage": stage,
                              "correctness_pass": valid,
                              **{k + "_activities_mean": statistics.mean(v) for k, v in values.items()},
                              "registers_per_thread": "N/A", "shared_bytes_per_block": "N/A"})
        for stage, values in run.get("memory", {}).items():
            resources.append({
                "version": version, "frame": frame, "stage": stage,
                "baseline_allocated_MiB": values[0] / 2**20,
                "peak_allocated_MiB": values[1] / 2**20,
                "peak_increment_MiB": (values[1]-values[0]) / 2**20,
                "correctness_pass": valid,
                "max_abs": run.get("correctness", {}).get("max_abs"),
                "relative_l2": run.get("correctness", {}).get("relative_l2"),
                "warmup_seconds": run.get("warmup_seconds"),
                "error": run.get("error", ""),
                "registers_per_thread": "N/A: collect separately with Nsight Compute",
                "shared_bytes_per_block": "N/A: collect separately with Nsight Compute",
            })
        if run.get("error"):
            resources.append({"version": version, "frame": frame, "stage": "error",
                              "correctness_pass": False, "error": run["error"]})
    def key(row):
        return tuple(row.get(k) for k in ("frame", "op", "layer_call", "stage", "metric"))
    lookup = {(r["version"], key(r)): r for r in stages}
    for row in stages:
        for baseline in ("original", "legacy_rebuilt"):
            ref = lookup.get((baseline, key(row)))
            duration = row["median_ms"]
            row["speedup_vs_" + baseline] = (
                ref["median_ms"] / duration
                if ref and ref["correctness_pass"] and row["correctness_pass"]
                and duration and ref["median_ms"] is not None else None)
    write_csv(directory / "01_stage_times.csv", stages)
    write_csv(directory / "02_attribution.csv", attribution)
    write_csv(directory / "03_correctness_resources.csv", resources)
    (directory / "results.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# RepFlow benchmark",
        "",
        "Three tables: [stage times](01_stage_times.csv), [attribution](02_attribution.csv), "
        "[correctness/resources](03_correctness_resources.csv).",
        "",
        "CUDA-event training intervals and profiler-attributed kernel sums are DIFFERENT metrics. "
        "Only rows of the same metric and workload are compared. Per-op backward/double-backward "
        "rows are GPU kernel sums, not Python-inclusive autograd latency.",
        "",
        "model_loss_forward_including_force contains energy evaluation, force autograd and loss construction. "
        "loss_backward includes ordinary energy backward and force/virial double backward. "
        "Do not add force_autograd to model_loss_forward_including_force.",
        "",
        "Failed correctness rows have no speedup. Missing profiler resource metrics are N/A, not zero.",
        "",
        "## 1. Stage times (all measured layer calls summed for per-op rows)",
        "",
        "| Version | Frame | Op | Stage | Metric | Median ms | vs original | Correct |",
        "|---|---:|---|---|---|---:|---:|---|",
    ]
    for row in stages:
        if row["layer_call"] == "all" and row["op"] != "other" and row["metric"] != "host_wall_including_phase_sync":
            speed = row.get("speedup_vs_original")
            speed = f"{speed:.3f}x" if speed is not None else "N/A"
            lines.append(f'| {row["version"]} | {row["frame"]} | {row["op"]} | {row["stage"]} | '
                         f'{row["metric"]} | {row["median_ms"]:.6f} | {speed} | {row["correctness_pass"]} |')
    lines += ["", "## 2. Attribution preview (full per-kernel/per-layer data in CSV)", "",
              "| Version | Frame | Op | Stage | Factory | GPU ms | Activities |",
              "|---|---:|---|---|---|---:|---:|"]
    grouped = defaultdict(lambda: [0., 0])
    for row in attribution:
        if row["op"] == "other" or row["profile_sample"] != 0:
            continue
        key = tuple(row[k] for k in ("version", "frame", "op", "stage", "factory"))
        grouped[key][0] += row["gpu_ms"]
        grouped[key][1] += row["launches"]
    for key, value in grouped.items():
        lines.append("| " + " | ".join(map(str, key)) + f" | {value[0]:.6f} | {value[1]} |")
    lines += ["", "## 3. Correctness and memory (separate memory passes)", "",
              "| Version | Frame | Stage | Baseline MiB | Peak MiB | Increment MiB | Correct |",
              "|---|---:|---|---:|---:|---:|---|"]
    for row in resources:
        if "peak_allocated_MiB" in row:
            lines.append(f'| {row["version"]} | {row["frame"]} | {row["stage"]} | '
                         f'{row["baseline_allocated_MiB"]:.3f} | {row["peak_allocated_MiB"]:.3f} | '
                         f'{row["peak_increment_MiB"]:.3f} | {row["correctness_pass"]} |')
        if row.get("error"):
            lines += ["", f'Failure: {row["version"]}, frame {row["frame"]}. See results.json for traceback.']
    (directory / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
