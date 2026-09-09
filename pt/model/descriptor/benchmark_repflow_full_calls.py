"""Real finetune path, synchronized complete operator calls, correctness SKIPPED.

Run on the training GH100 host. Requires the existing benchmark helper files.
"""
import argparse
import copy
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import statistics
import time

from benchmark_repflow_training import prepare_config
from repflow_full_timing import Timer, boundaries, force_path, kernel_sums


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--finetune", required=True, type=Path)
    p.add_argument("--model-branch", default="OMat24")
    p.add_argument("--out", type=Path, default=Path("repflow_full_calls"))
    p.add_argument("--versions", nargs="+", default=["original", "legacy_rebuilt", "v2_materialized", "current"],
                   choices=["original", "legacy_rebuilt", "v2_materialized", "current"])
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--profile-repeat", type=int, default=1, help="Separate kernel-sum passes; 0 disables")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--skip-neighbor-stat", action="store_true")
    p.add_argument("--disable-tf32", action="store_true")
    return p.parse_args()


def report(directory, runs):
    from collections import defaultdict
    rows = []
    for run in runs:
        grouped = defaultdict(list)
        for sample in run.get("samples", []):
            sums = defaultdict(lambda: dict(wall_ms=0., cuda_ms=0., calls=0))
            for row in sample:
                key = row["op"], row["stage"]
                for metric in ("wall_ms", "cuda_ms"):
                    sums[key][metric] += row[metric]
                sums[key]["calls"] += 1
            for key, value in sums.items():
                grouped[key].append(value)
        for (op, stage), values in grouped.items():
            row = dict(version=run["version"], frame=run["frame"], op=op, stage=stage,
                       correctness="SKIPPED", samples=len(values),
                       calls=[v["calls"] for v in values])
            for metric in ("wall_ms", "cuda_ms"):
                xs = sorted(v[metric] for v in values)
                row[metric + "_median"] = statistics.median(xs)
                row[metric + "_min"] = xs[0]
                row[metric + "_max"] = xs[-1]
            ks = [sum(r["kernel_ms"] for r in s if r["op"] == op and r["stage"] == stage)
                  for s in run.get("profiles", [])]
            row["kernel_ms_median"] = statistics.median(ks) if ks else None
            rows.append(row)
    lookup = {(r["version"], r["frame"], r["op"], r["stage"]): r for r in rows}
    for r in rows:
        ref = lookup.get(("original", r["frame"], r["op"], r["stage"]))
        r["unvalidated_wall_time_ratio_vs_original"] = (
            ref["wall_ms_median"] / r["wall_ms_median"] if ref and r["wall_ms_median"] else None)
    (directory / "results.json").write_text(json.dumps(dict(summary=rows, runs=runs), indent=2), encoding="utf-8")
    text = ["# RepFlow 完整调用计时（正确性未验证）", "",
            "主指标：每次算子调用前同步、调用后等待 GPU 完成的墙钟时间。",
            "每行是一个训练样本内该类算子全部调用的累计值，再对重复采样取中位数；单位 ms。",
            "计时使用 benchmark-only autograd 封装边界，含局部 autograd 调度和计时开销；不是无插桩训练延迟。",
            "CUDA Event 区间不是 kernel 求和；kernel 求和来自独立 profiler 采样，不从墙钟中相减。",
            "原始求导公式来自 repflow_layer 原函数的计算图；外层坐标 grad 和真实 loss.backward 不变。",
            "", "所有耗时比均未经正确性验证。", ""]
    def fmt(x):
        return "—" if x is None else f"{x:.4f}"
    for stage in ("forward", "backward", "double_backward"):
        text.extend([f"## {stage}", "",
                     "| 算子 | 版本 | frame | 调用数范围 | 完整墙钟 ms | CUDA 区间 ms | kernel 和 ms | 原始/当前墙钟 |",
                     "|---|---|---:|---:|---:|---:|---:|---:|"])
        for r in rows:
            if r["stage"] != stage:
                continue
            text.append(f'| {r["op"]} | {r["version"]} | {r["frame"]} | '
                        f'{min(r["calls"])}–{max(r["calls"])} | {fmt(r["wall_ms_median"])} | '
                        f'{fmt(r["cuda_ms_median"])} | {fmt(r["kernel_ms_median"])} | '
                        f'{fmt(r["unvalidated_wall_time_ratio_vs_original"])} |')
        text.append("")
    text.append("ordinary_backward（loss 中对原始前向图的一阶反传）单独保存在 results.json，不混入 double backward。")
    (directory / "REPORT.md").write_text("\n".join(text), encoding="utf-8")


def benchmark(trainer, args, directory):
    import numpy as np
    import torch
    from deepmd.pt.model.descriptor.repflow_layer import RepFlowLayer
    from repflow_bench_adapters import OPS, Recorder, select_version
    if trainer.multi_task or isinstance(trainer.wrapper, torch.jit.ScriptModule):
        raise RuntimeError("Requires single-task eager training")
    if args.step < trainer.warmup_steps:
        raise RuntimeError("Warmup schedule unsupported; supplied config has warmup_steps=0")
    wrapper = trainer.wrapper
    wrapper.train()
    state = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
             for k, v in wrapper.state_dict().items()}
    batches = [trainer.get_data(is_train=True, task_key="Default") for _ in range(args.frames)]
    names = dict(layers={id(m): n for n, m in wrapper.named_modules() if isinstance(m, RepFlowLayer)},
                 signatures={op: inspect.signature(getattr(RepFlowLayer, names[0])) for op, names in OPS.items()})
    if not names["layers"]:
        raise RuntimeError("No eager RepFlow layers found")
    timer = Timer()
    recorder = Recorder()  # Disabled: no old profiler graph hooks or correctness snapshots.
    runs = []
    metadata = dict(correctness="SKIPPED", torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), tf32=torch.backends.cuda.matmul.allow_tf32,
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    layers=list(names["layers"].values()), measurement="synchronized graph-encapsulated calls",
                    caveat="Benchmark-only local VJP engine boundaries; not uninstrumented training latency.")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with force_path(timer):
        for version in dict.fromkeys(["original"] + args.versions):
            wrapper.load_state_dict(state)
            with select_version(version, recorder), boundaries(timer, RepFlowLayer, names) as counts:
                for frame, batch in enumerate(batches):
                    run = dict(version=version, frame=frame, correctness="SKIPPED", samples=[], profiles=[])
                    runs.append(run)
                    def execute():
                        wrapper.zero_grad(set_to_none=True)
                        counts.clear()
                        timer.rows.clear()
                        timer.requests.clear()
                        timer.phase = "forward"
                        torch.manual_seed(args.seed)
                        random.seed(args.seed)
                        np.random.seed(args.seed)
                        inputs, labels, _ = batch
                        inputs = {k: v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                                  for k, v in inputs.items()}
                        _, loss, _ = wrapper(**inputs, label=labels,
                                             cur_lr=trainer.lr_exp.value(args.step), task_key="Default")
                        if not timer.requests:
                            raise RuntimeError("Real task_deriv_one coordinate grad was not intercepted")
                        timer.phase = "loss"
                        loss.backward()
                        torch.cuda.synchronize()
                    try:
                        print(f"[{version} frame={frame}] warmup/JIT", flush=True)
                        timer.enabled = timer.profile = False
                        for _ in range(args.warmup):
                            execute()
                        timer.enabled = True
                        for sample in range(args.repeat):
                            execute()
                            run["samples"].append(list(timer.rows))
                            run["coordinate_grad_requests"] = list(timer.requests)
                            print(f"  complete-call sample {sample + 1}/{args.repeat}", flush=True)
                        timer.enabled = False
                        timer.profile = True
                        for sample in range(args.profile_repeat):
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                  torch.profiler.ProfilerActivity.CUDA]) as prof:
                                execute()
                            path = directory / f"trace_full_{version}_{frame}_{sample}.json"
                            prof.export_chrome_trace(str(path))
                            run["profiles"].append(kernel_sums(json.loads(path.read_text(encoding="utf-8"))))
                        covered = {(r["op"], r["stage"]) for s in run["samples"] for r in s}
                        missing = [f"{op}/{stage}" for op in OPS for stage in
                                   ("forward", "backward", "double_backward") if (op, stage) not in covered]
                        if missing:
                            raise RuntimeError("Missing full-call ranges: " + ", ".join(missing))
                    except Exception as exc:
                        run["error"] = repr(exc)
                        raise
                    finally:
                        timer.enabled = timer.profile = False
                        report(directory, runs)
    print(f"Reports: {directory}\nCorrectness: SKIPPED (not PASS)", flush=True)


def main():
    args = arguments()
    if min(args.frames, args.warmup, args.repeat) < 1 or args.profile_repeat < 0:
        raise ValueError("frames/warmup/repeat >= 1; profile-repeat >= 0")
    args.input, args.finetune = args.input.resolve(), args.finetune.resolve()
    if not args.input.is_file() or not args.finetune.is_file():
        raise FileNotFoundError("Provide the existing training JSON and checkpoint")
    if os.environ.get("LOCAL_RANK") is not None:
        raise RuntimeError("Use one GPU without torchrun/DDP")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Run on the GH100 training server with CUDA and TileLang")
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    directory = args.out.resolve() / time.strftime("run_%Y%m%d_%H%M%S")
    directory.mkdir(parents=True, exist_ok=False)
    config = prepare_config(args, directory)
    entry = importlib.import_module("deepmd.pt.entrypoints.main")
    if entry.training.JIT:
        raise RuntimeError("Disable DeePMD JIT for benchmark-only graph boundaries")
    previous, original_run = Path.cwd(), entry.training.Trainer.run
    try:
        os.chdir(directory)
        entry.training.Trainer.run = lambda trainer: benchmark(trainer, args, directory)
        entry.train(input_file=str(config), init_model=None, restart=None, finetune=str(args.finetune),
                    init_frz_model=None, model_branch=args.model_branch,
                    skip_neighbor_stat=args.skip_neighbor_stat, use_pretrain_script=False,
                    force_load=False, output=str(directory / "effective_input.json"))
    finally:
        entry.training.Trainer.run = original_run
        os.chdir(previous)


if __name__ == "__main__":
    main()
