"""Unmodified training autograd graph; only outer-phase synchronized timings.

Correctness is SKIPPED. Per-op data is profiler attribution, not isolated latency.
"""
import contextlib
import copy
import inspect
import json
from pathlib import Path
import random
import statistics
import time
from collections import defaultdict


@contextlib.contextmanager
def force_calls(rec, measured, rows, memory_rows=None):
    import torch
    original = torch.autograd.grad
    def grad(*args, **kwargs):
        frame = inspect.currentframe().f_back
        matched = (frame.f_code.co_name == "task_deriv_one" and
                   frame.f_code.co_filename.replace("\\", "/").endswith("/transform_output.py"))
        del frame
        if not matched:
            return original(*args, **kwargs)
        if not kwargs.get("create_graph") or not kwargs.get("retain_graph"):
            raise RuntimeError("Expected create_graph=True, retain_graph=True")
        xs = kwargs.get("inputs", args[1] if len(args) > 1 else ())
        rec.derivative_requests.append([dict(shape=list(x.shape), dtype=str(x.dtype)) for x in xs])
        old = rec.phase
        rec.phase = "force"
        try:
            if memory_rows is not None:
                return measure_memory(lambda: original(*args, **kwargs), memory_rows, "force_autograd")
            if measured:
                return timed(lambda: original(*args, **kwargs), rows, "force_autograd")
            return original(*args, **kwargs)
        finally:
            rec.phase = old
    torch.autograd.grad = grad
    try:
        yield
    finally:
        torch.autograd.grad = original


def timed(fn, rows, stage):
    import torch
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    end.record()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000
    rows.append(dict(stage=stage, wall_ms=wall, cuda_ms=start.elapsed_time(end)))
    return result


def measure_memory(fn, rows, stage):
    """One independent allocator-peak sample, never nested in a timed pass."""
    import torch
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    rows.append(dict(stage=stage, baseline_allocated_bytes=baseline,
                     peak_allocated_bytes=peak, increment_allocated_bytes=peak - baseline,
                     baseline_reserved_bytes=reserved,
                     peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    return result


@contextlib.contextmanager
def metadata_diagnostics(version, rec):
    """Cache hit/miss annotations only in the separate profiler pass."""
    if version == "original":
        yield
        return
    import torch
    if version == "legacy_rebuilt":
        import repflow_bench_legacy as selected
    else:
        from deepmd.pt.model.descriptor import utils_tilelang as selected
    original = selected._sym_owner_metadata
    def metadata(owner, num_owner):
        if not rec.enabled:
            return original(owner, num_owner)
        base = owner
        while base._base is not None:
            base = base._base
        key = (id(base), owner.data_ptr(), owner.numel(), owner.stride(),
               owner._version, num_owner, owner.device)
        cached = selected._sym_owner_cache.get(key)
        hit = cached is not None and cached[0]() is base
        with torch.profiler.record_function("rfbench_metadata|" + ("hit" if hit else "miss")):
            return original(owner, num_owner)
    selected._sym_owner_metadata = metadata
    try:
        yield
    finally:
        selected._sym_owner_metadata = original


def write_report(directory, runs):
    summary = []
    for run in runs:
        values = defaultdict(list)
        for sample in run.get("samples", []):
            summed = defaultdict(float)
            for row in sample:
                for metric in ("wall_ms", "cuda_ms"):
                    summed[row["stage"], metric] += row[metric]
            for key, value in summed.items():
                values[key].append(value)
        for (stage, metric), xs in values.items():
            summary.append(dict(version=run["version"], frame=run["frame"], stage=stage,
                                metric=metric, median_ms=statistics.median(xs), min_ms=min(xs),
                                max_ms=max(xs), samples=len(xs), correctness="SKIPPED"))
    lookup = {(r["version"], r["frame"], r["stage"], r["metric"]): r for r in summary}
    for r in summary:
        ref = lookup.get(("original", r["frame"], r["stage"], r["metric"]))
        r["unvalidated_ratio_vs_original"] = ref["median_ms"] / r["median_ms"] if ref and r["median_ms"] else None
    lines = ["# RepFlow v2：真实训练图计时", "", "正确性：SKIPPED。所有比值未经正确性验证。",
             "不使用 _Boundary/_VJP，不修改算子输入、owner 对象或内部 autograd 图。",
             "", "## 1. 完整训练阶段（独立采样，ms）", "",
             "| 版本 | frame | 阶段 | 墙钟中位数 | CUDA 区间中位数 | 原始/当前墙钟 |",
             "|---|---:|---|---:|---:|---:|"]
    for r in summary:
        if r["metric"] != "wall_ms":
            continue
        cuda = lookup[r["version"], r["frame"], r["stage"], "cuda_ms"]["median_ms"]
        ratio = r["unvalidated_ratio_vs_original"]
        lines.append(f'| {r["version"]} | {r["frame"]} | {r["stage"]} | {r["median_ms"]:.4f} | '
                     f'{cuda:.4f} | {ratio:.3f} |' if ratio is not None else
                     f'| {r["version"]} | {r["frame"]} | {r["stage"]} | {r["median_ms"]:.4f} | {cuda:.4f} | — |')
    lines.extend(["", "train_forward_including_force = 整个 wrapper（包含坐标求导和 loss 构建）；",
                  "force_autograd = task_deriv_one 的完整坐标求导调用；",
                  "loss_backward = 完整 loss.backward，包含一阶及二阶路径；",
                  "full_step = wrapper + loss.backward，无中间同步。不同阶段来自不同新图采样，不能相减或相加。",
                  "", "## 2. 算子 GPU 活动耗时之和（独立 profiler 采样，ms）", "",
                  "| 版本 | frame | 算子 | forward | backward | double backward |",
                  "|---|---:|---|---:|---:|---:|"])
    for run in runs:
        for op in ("sym", "edge", "angle"):
            cells = []
            for stage in ("forward", "backward", "double_backward"):
                xs = []
                for p in run.get("profiles", []):
                    rs = [r for r in p["kernels"] if r["op"] == op and r["stage"] == stage]
                    if rs:
                        xs.append(sum(r["kernel_ms"] for r in rs))
                cells.append(f"{statistics.median(xs):.4f}" if xs else "缺失")
            lines.append(f'| {run["version"]} | {run["frame"]} | {op} | ' + " | ".join(cells) + " |")
    lines.extend(["", "## 3. 采样诊断", "", "| 版本 | frame | profile | 未归属活动数 | 歧义序列数 |",
                  "|---|---:|---:|---:|---:|"])
    for run in runs:
        for i, p in enumerate(run.get("profiles", [])):
            lines.append(f'| {run["version"]} | {run["frame"]} | {i} | {p["unattributed_activities"]} | {p["ambiguous_sequences"]} |')
    lines.extend(["", "未归属活动包含模型其他部分；缺失或歧义不能当作零耗时。",
                  "CPU 标记时长、GPU 首末时间戳及 factory 分解保存在 results.json。",
                  "CPU 标记时长不是 GPU 完成时间；首末 GPU 跨度可能夹杂其他算子，不是独立算子延迟。",
                  "profiler 中的 hooks/标记有开销，但不会用于表 1 的计时。"])
    from repflow_resource_report_v2 import tables
    resource_lines, resource_data = tables(directory, runs)
    lines.extend([""] + resource_lines)
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (directory / "results.json").write_text(json.dumps(dict(summary=summary, runs=runs,
                                                           resources=resource_data), indent=2), encoding="utf-8")


def benchmark(trainer, args, directory):
    import torch
    import numpy as np
    from deepmd.pt.model.descriptor import repflow_layer, utils_tilelang
    from repflow_bench_adapters import Recorder, select_version
    from repflow_trace_v2 import analyze
    if trainer.multi_task or isinstance(trainer.wrapper, torch.jit.ScriptModule):
        raise RuntimeError("Requires single-task eager training")
    if args.step < trainer.warmup_steps:
        raise RuntimeError("Expected warmup_steps=0")
    wrapper = trainer.wrapper
    wrapper.train()
    state = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
             for k, v in wrapper.state_dict().items()}
    batches = [trainer.get_data(is_train=True, task_key="Default") for _ in range(args.frames)]
    rec = Recorder()
    rec.layer_names = {id(m): n for n, m in wrapper.named_modules() if isinstance(m, repflow_layer.RepFlowLayer)}
    if not rec.layer_names:
        raise RuntimeError("No eager RepFlowLayer found")
    metadata = dict(correctness="SKIPPED", torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), tf32=torch.backends.cuda.matmul.allow_tf32,
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    layers=list(rec.layer_names.values()),
                    source_paths=[repflow_layer.__file__, utils_tilelang.__file__],
                    measurement="real training graph; independent outer-phase timing passes")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    runs = []
    for version in dict.fromkeys(["original"] + args.versions):
        wrapper.load_state_dict(state)
        with select_version(version, rec):
            for frame, batch in enumerate(batches):
                run = dict(version=version, frame=frame, correctness="SKIPPED", samples=[], profiles=[], memory=[])
                runs.append(run)
                def execute(mode):
                    rec.clear_graph_hooks()
                    rec.derivative_requests.clear()
                    rec.phase = "forward"
                    wrapper.zero_grad(set_to_none=True)
                    torch.manual_seed(args.seed)
                    random.seed(args.seed)
                    np.random.seed(args.seed)
                    inputs, labels, _ = batch
                    inputs = {k: v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                              for k, v in inputs.items()}
                    rows = []
                    def forward():
                        return wrapper(**inputs, label=labels, cur_lr=trainer.lr_exp.value(args.step), task_key="Default")
                    def step():
                        _, loss, _ = forward()
                        rec.phase = "loss"
                        loss.backward()
                    try:
                        with force_calls(rec, mode == "force_autograd", rows,
                                         run["memory"] if mode == "memory_force_autograd" else None):
                            if mode == "memory_full_step":
                                measure_memory(step, run["memory"], "full_step")
                            elif mode == "full_step":
                                timed(step, rows, mode)
                            else:
                                if mode == "memory_train_forward_including_force":
                                    _, loss, _ = measure_memory(forward, run["memory"], "train_forward_including_force")
                                else:
                                    _, loss, _ = (timed(forward, rows, mode) if mode == "train_forward_including_force" else forward())
                                rec.phase = "loss"
                                if mode == "memory_loss_backward":
                                    measure_memory(loss.backward, run["memory"], "loss_backward")
                                elif mode == "loss_backward":
                                    timed(loss.backward, rows, mode)
                                else:
                                    loss.backward()
                        if not rec.derivative_requests:
                            raise RuntimeError("Actual coordinate grad was not intercepted")
                        torch.cuda.synchronize()
                        run["coordinate_grad_requests"] = list(rec.derivative_requests)
                        return rows
                    finally:
                        rec.clear_graph_hooks()
                try:
                    rec.enabled = rec.capture = False
                    print(f"[{version} frame={frame}] warmup/JIT", flush=True)
                    for _ in range(args.warmup):
                        execute("warmup")
                    for sample in range(args.repeat):
                        rows = []
                        for mode in ("full_step", "train_forward_including_force", "force_autograd", "loss_backward"):
                            rows.extend(execute(mode))
                        run["samples"].append(rows)
                        print(f"  true-graph sample {sample + 1}/{args.repeat}", flush=True)
                    print("  independent allocator-peak samples", flush=True)
                    for stage in ("full_step", "train_forward_including_force", "force_autograd", "loss_backward"):
                        execute("memory_" + stage)
                    for sample in range(args.profile_repeat):
                        rec.enabled = True
                        # capture stays False: no index copies or GPU->CPU reads.
                        with metadata_diagnostics(version, rec), torch.profiler.profile(
                                activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
                            execute("profile")
                        rec.enabled = False
                        path = directory / f"trace_v2_{version}_{frame}_{sample}.json"
                        prof.export_chrome_trace(str(path))
                        run["profiles"].append(analyze(json.loads(path.read_text(encoding="utf-8"))))
                except Exception as exc:
                    run["error"] = repr(exc)
                    raise
                finally:
                    rec.enabled = rec.capture = False
                    write_report(directory, runs)
    print(f"Reports: {directory}\nCorrectness: SKIPPED", flush=True)


def main():
    # Reuse only configuration isolation and the real finetune entrypoint.
    # No call into the old benchmark body or graph-boundary timing code.
    import benchmark_repflow_training as entry
    import sys
    if "--out" not in sys.argv:
        sys.argv.extend(["--out", "repflow_training_v2"])
    old = entry.benchmark
    entry.benchmark = benchmark
    try:
        entry.main()
    finally:
        entry.benchmark = old


if __name__ == "__main__":
    main()
