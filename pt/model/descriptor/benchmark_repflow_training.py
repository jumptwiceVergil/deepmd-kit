"""Benchmark the real finetune energy -> coordinate grad -> loss.backward graph.

Only added files are used. Existing source/config/checkpoint/data are not edited.
Initialization follows the training entrypoint; Trainer.run is replaced in this
process with fixed-weight repeated measurements, never optimizer.step.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

from repflow_bench_report import attributed_kernels, render_reports


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="Existing input_finetune_1.json")
    p.add_argument("--finetune", required=True, type=Path, help="Existing DPA-3.2-5M.pt")
    p.add_argument("--model-branch", default="OMat24")
    p.add_argument("--out", type=Path, default=Path("repflow_benchmark"))
    p.add_argument("--versions", nargs="+",
                   choices=("original", "legacy_rebuilt", "v2_materialized", "current"),
                   default=["original", "legacy_rebuilt", "v2_materialized", "current"])
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--profile-repeat", type=int, default=2)
    p.add_argument("--step", type=int, default=0, help="Loss/LR schedule step; no optimizer update")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=2e-3)
    p.add_argument("--skip-neighbor-stat", action="store_true")
    p.add_argument("--disable-tf32", action="store_true",
                   help="Disable PyTorch TF32; does NOT change TileLang operand precision")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Continue other versions after compile/runtime error; correctness failure always recorded")
    return p.parse_args()


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(v) for v in value]
    return value


def tensor_leaves(value, prefix=""):
    import torch
    if isinstance(value, torch.Tensor) or value is None:
        yield prefix, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from tensor_leaves(v, prefix + "/" + str(k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from tensor_leaves(v, prefix + "/" + str(i))


def compare(reference, actual, atol, rtol):
    import torch
    refs, acts = dict(tensor_leaves(reference)), dict(tensor_leaves(actual))
    errors, details = [], {}
    delta2 = ref2 = max_abs = 0.0
    for name in sorted(set(refs) | set(acts)):
        if name not in refs or name not in acts:
            errors.append(name + ": missing output")
            continue
        a, b = acts[name], refs[name]
        if a is None and b is None:
            continue
        if a is None:
            a = torch.zeros_like(b)
        if b is None:
            b = torch.zeros_like(a)
        if a.shape != b.shape:
            errors.append(name + ": shape mismatch")
            continue
        if not a.numel():
            continue
        a, b = a.double(), b.double()
        d = a - b
        err = d.abs().max().item()
        good = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
        details[name] = {"pass": good, "max_abs": err,
                         "relative_l2": d.norm().item() / max(b.norm().item(), 1e-30)}
        max_abs = max(max_abs, err) if torch.isfinite(d).all() else float("inf")
        delta2 += d.square().sum().item()
        ref2 += b.square().sum().item()
        if not good:
            errors.append(name)
    return {"pass": not errors, "max_abs": max_abs,
            "relative_l2": (delta2 / max(ref2, 1e-60))**.5,
            "failed": errors, "per_tensor": details}


def prepare_config(args, directory):
    """Resolve input paths before changing cwd; copy writable statistics."""
    config = json.loads(args.input.read_text(encoding="utf-8"))
    if "model_dict" in config["model"]:
        raise ValueError("This harness targets the supplied single-task finetune config")
    train = config["training"]
    for name in ("training_data", "validation_data"):
        if name in train:
            systems = train[name]["systems"]
            train[name]["systems"] = (
                [os.path.abspath(s) for s in systems] if isinstance(systems, list)
                else os.path.abspath(systems))
    stat = train.get("stat_file")
    if stat:
        original = Path(stat).absolute()
        target = directory / ("statistics" + original.suffix)
        if original.is_file():
            shutil.copy2(original, target)
        elif original.is_dir():
            shutil.copytree(original, target)
        train["stat_file"] = str(target)
    train["disp_file"] = str(directory / "unused_lcurve.out")
    train["save_ckpt"] = str(directory / "unused_checkpoint")
    if "tensorboard_log_dir" in train:
        train["tensorboard_log_dir"] = str(directory / "tensorboard")
    # These flags only control training-side outputs, not the model/loss.
    for flag in ("enable_tensorboard", "enable_profiler", "profiling"):
        if flag in train:
            train[flag] = False
    target = directory / "benchmark_input.json"
    target.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return target


def benchmark(trainer, args, directory):
    import torch
    from deepmd.pt.model.descriptor import repflow_layer, utils_tilelang
    from repflow_bench_adapters import Recorder, intercept_force_grad, select_version

    if trainer.multi_task:
        raise RuntimeError("Expected the supplied single-task training configuration")
    wrapper = trainer.wrapper
    if isinstance(wrapper, torch.jit.ScriptModule):
        raise RuntimeError("TorchScript bypasses Python instrumentation; run eager training")
    wrapper.train()
    state = copy.deepcopy(wrapper.state_dict())
    # Do not inflate GPU memory measurements with a duplicate GPU checkpoint.
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            state[name] = value.detach().cpu()
    batches = [trainer.get_data(is_train=True, task_key="Default") for _ in range(args.frames)]
    rec = Recorder()
    rec.layer_names = {id(m): name for name, m in wrapper.named_modules()
                       if isinstance(m, repflow_layer.RepFlowLayer)}
    if not rec.layer_names:
        raise RuntimeError("No eager RepFlowLayer found; cannot silently benchmark an unpatched model")
    lr = trainer.lr_exp.value(args.step)
    pref_lr = lr
    # Match scheduler warmup branch without depending on the LR object's fields.
    if args.step < trainer.warmup_steps:
        pref_lr = lr
        raise RuntimeError("Warmup loss scheduling needs explicit support; supplied config uses warmup_steps=0")
    metadata = {
        "command": f"dp --pt train {args.input} --finetune {args.finetune} --model-branch {args.model_branch}",
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "device_properties": str(torch.cuda.get_device_properties(torch.cuda.current_device())),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "correctness_tolerances": {"atol": args.atol, "rtol": args.rtol},
        "loss_schedule_step": args.step, "loss_pref_lr": pref_lr,
        "actual_repflow_layers": list(rec.layer_names.values()),
        "actual_model": repr(wrapper),
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                          for p in (repflow_layer.__file__, utils_tilelang.__file__)},
        "batch_log": [cpu_tree(log) for _, _, log in batches],
        "notes": [
            "No optimizer updates; identical loaded weights and batches for every version.",
            "Original methods use their real matmul-before-index_select order.",
            "legacy_rebuilt: commit 6ff0569 reachable definitions, edge bias correctness fix.",
            "Per-op GPU sums are profiler attribution, not full autograd latency.",
            "Warmup seconds are NOT guaranteed cold JIT time; caches may already be populated.",
        ],
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    torch.save(cpu_tree(batches), directory / "sampled_batches.pt")
    references = {}
    runs = []

    def execute(batch, measure=False, memory_target=None, snapshot=False):
        rec.clear_graph_hooks()
        rec.measure = measure
        rec.memory_target = memory_target
        rec.phase = "forward"
        wrapper.zero_grad(set_to_none=True)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        inputs, labels, _ = batch
        # Fresh coordinates/graph every iteration; no cached force graph replay.
        inputs = {k: v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                  for k, v in inputs.items()}
        torch.cuda.synchronize()
        memory = {}
        if memory_target == "wrapper":
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        start, mid, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        wall_start = time.perf_counter()
        start.record()
        prediction, loss, more_loss = wrapper(**inputs, label=labels, cur_lr=pref_lr, task_key="Default")
        mid.record()
        mid.synchronize()
        wall_mid = time.perf_counter()
        if not rec.force_events and measure:
            raise RuntimeError("task_deriv_one coordinate autograd was not intercepted")
        if memory_target == "wrapper":
            memory["model_loss_forward_including_force"] = (baseline, torch.cuda.max_memory_allocated())
        if memory_target == "force":
            if not rec.force_memory:
                raise RuntimeError("No force memory sample")
            memory["force_autograd"] = max(rec.force_memory, key=lambda x: x[1]-x[0])
        if memory_target == "loss":
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        rec.phase = "loss"
        # A separate start excludes the intentional phase-boundary synchronization.
        loss_start = torch.cuda.Event(enable_timing=True)
        loss_start.record()
        loss_wall_start = time.perf_counter()
        loss.backward()
        end.record()
        end.synchronize()
        wall_end = time.perf_counter()
        if memory_target == "loss":
            memory["loss_backward"] = (baseline, torch.cuda.max_memory_allocated())
        timings = {
            "model_loss_forward_including_force": start.elapsed_time(mid),
            "force_autograd": sum(a.elapsed_time(b) for a, b in rec.force_events),
            "loss_backward": loss_start.elapsed_time(end),
        }
        walls = {"model_loss_forward_including_force": (wall_mid-wall_start)*1000,
                 "loss_backward": (wall_end-loss_wall_start)*1000}
        result = None
        if snapshot:
            result = cpu_tree({
                "prediction": prediction, "loss": loss,
                "parameter_gradients": {n: p.grad for n, p in wrapper.named_parameters()},
            })
        rec.clear_graph_hooks()
        return timings, walls, memory, result

    versions = ["original"] + [v for v in args.versions if v != "original"]
    with intercept_force_grad(rec):
        for version in versions:
            wrapper.load_state_dict(state)
            with select_version(version, rec):
                for frame, batch in enumerate(batches):
                    run = {"version": version, "frame": frame, "timings": {}, "wall_timings": {},
                           "memory": {}, "profiles": []}
                    runs.append(run)
                    try:
                        print(f"[{version} frame={frame}] warmup / JIT", flush=True)
                        rec.enabled = rec.capture = False
                        t0 = time.perf_counter()
                        for _ in range(args.warmup):
                            execute(batch)
                        run["warmup_seconds"] = time.perf_counter()-t0
                        # End-to-end predictions, force/virial and parameter grads.
                        _, _, _, actual = execute(batch, snapshot=True)
                        if version == "original":
                            references[frame] = actual
                        run["correctness"] = compare(references[frame], actual, args.atol, args.rtol)
                        del actual
                        for _ in range(args.repeat):
                            timings, walls, _, _ = execute(batch, measure=True)
                            for key, value in timings.items():
                                run["timings"].setdefault(key, []).append(value)
                            for key, value in walls.items():
                                run["wall_timings"].setdefault(key, []).append(value)
                        for target in ("wrapper", "force", "loss"):
                            _, _, memory, _ = execute(batch, memory_target=target)
                            run["memory"].update(memory)
                        rec.capture = True
                        rec.shapes.clear()
                        rec.index_samples.clear()
                        rec.kernel_specs.clear()
                        rec.metadata_stats.clear()
                        rec.missing.clear()
                        rec.derivative_requests.clear()
                        for sample in range(args.profile_repeat):
                            rec.enabled = True
                            with torch.profiler.profile(
                                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True,
                            ) as prof:
                                execute(batch)
                            rec.enabled = False
                            prof.export_chrome_trace(str(directory / f"trace_{version}_{frame}_{sample}.json"))
                            rows, total = attributed_kernels(prof.events())
                            run["profiles"].append(rows)
                            run.setdefault("profile_total_kernel_ms", []).append(total)
                        rec.capture = False
                        run["calls"] = rec.shapes.copy()
                        run["index_samples"] = rec.index_samples.copy()
                        run["kernel_specializations"] = list(rec.kernel_specs.values())
                        run["owner_metadata_calls"] = rec.metadata_stats.copy()
                        run["missing_cotangents"] = rec.missing.copy()
                        run["derivative_requests"] = rec.derivative_requests.copy()
                        for op in ("sym", "edge", "angle"):
                            if not any(f"/{op}/" in k for k in rec.shapes):
                                raise RuntimeError(f"No {op} calls captured; routing/model config mismatch")
                        # Never silently claim successful second-order attribution.
                        if not any(r["stage"] == "double_backward" for s in run["profiles"] for r in s):
                            run["attribution_warning"] = "No double-backward kernels attributed. Inspect trace/sequence metadata."
                        covered = {(r["op"], r["stage"]) for s in run["profiles"] for r in s}
                        run["attribution_missing"] = [
                            f"{op}/{stage}" for op in ("sym", "edge", "angle")
                            for stage in ("forward", "backward", "double_backward")
                            if (op, stage) not in covered]
                        print(f"[{version} frame={frame}] correctness={run['correctness']['pass']}", flush=True)
                    except Exception:
                        run["error"] = traceback.format_exc()
                        print(run["error"], flush=True)
                        render_reports(directory, runs)
                        if not args.continue_on_error or version == "original":
                            raise
                    finally:
                        rec.enabled = rec.capture = False
                        rec.clear_graph_hooks()
                    render_reports(directory, runs)
    print(f"Reports: {directory}", flush=True)
    if any(r.get("error") or not r.get("correctness", {}).get("pass") for r in runs):
        raise RuntimeError("Benchmark completed with failed correctness/runtime rows; see reports")


def main():
    args = arguments()
    args.versions = list(dict.fromkeys(args.versions))
    if min(args.frames, args.warmup, args.repeat, args.profile_repeat) < 1:
        raise ValueError("frames/warmup/repeat/profile-repeat must be positive")
    args.input, args.finetune = args.input.resolve(), args.finetune.resolve()
    if not args.input.is_file() or not args.finetune.is_file():
        raise FileNotFoundError("Input JSON and finetune checkpoint must already exist")
    if os.environ.get("LOCAL_RANK") is not None:
        raise RuntimeError("Use one GPU, not torchrun/DDP, for this operator benchmark")
    directory = args.out.resolve() / time.strftime("run_%Y%m%d_%H%M%S")
    directory.mkdir(parents=True, exist_ok=False)
    config_path = prepare_config(args, directory)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Run this script in the GH100 CUDA/TileLang environment")
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    import importlib
    entry = importlib.import_module("deepmd.pt.entrypoints.main")
    if entry.training.JIT:
        raise RuntimeError("Disable DeePMD JIT; eager hooks are required")
    original_run = entry.training.Trainer.run
    previous = Path.cwd()
    try:
        os.chdir(directory)
        entry.training.Trainer.run = lambda self: benchmark(self, args, directory)
        entry.train(
            input_file=str(config_path), init_model=None, restart=None,
            finetune=str(args.finetune), init_frz_model=None,
            model_branch=args.model_branch, skip_neighbor_stat=args.skip_neighbor_stat,
            use_pretrain_script=False, force_load=False,
            output=str(directory / "effective_input.json"),
        )
    finally:
        entry.training.Trainer.run = original_run
        os.chdir(previous)


if __name__ == "__main__":
    main()
