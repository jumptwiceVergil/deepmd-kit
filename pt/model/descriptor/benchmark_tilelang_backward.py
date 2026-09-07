"""Compare RepFlow backward-only latency on a CUDA/TileLang server.

Run in the matching installed DeePMD environment:
    python -m deepmd.pt.model.descriptor.benchmark_tilelang_backward --help

Original forwards and saved intermediates are prepared once, outside timing.
Each original backward uses create_graph=True and retain_graph=True.
Backward.apply invokes the requested Backward.forward with a real autograd ctx;
no second backward or fused primal forward is timed. Returned graphs are released
after each measurement. CUDA events measure the stream interval (including gaps
between launches); wall time additionally includes Python and synchronization.
The baseline requests the full matrix gradient (including split backward).
Fused backward returns separate weight gradients: concatenation for comparison
is NOT timed. Thus these are Backward-class timings, not end-to-end layer timings.
Peak memory uses PyTorch allocated bytes, not reserved memory or total device use.
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch


def measure(fn, warmup, repeat):
    for _ in range(warmup):
        result = fn()
        del result
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    gpu, wall = [], []
    for _ in range(repeat):
        t0 = time.perf_counter()
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - t0) * 1000)
        gpu.append(start.elapsed_time(end))
        del result
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    live = torch.cuda.memory_allocated()
    del result
    return {
        "cuda_median_ms": statistics.median(gpu),
        "cuda_min_ms": min(gpu),
        "cuda_p90_ms": sorted(gpu)[min(int(.9 * repeat), repeat - 1)],
        "wall_median_ms": statistics.median(wall),
        "baseline_allocated_MiB": baseline / 2**20,
        "peak_extra_allocated_MiB": (peak - baseline) / 2**20,
        "live_extra_allocated_MiB": (live - baseline) / 2**20,
    }


def check(reference, fused, names, atol, rtol):
    report = {}
    for name, expected, actual in zip(names, reference, fused):
        actual = actual.reshape_as(expected)
        delta = (actual - expected).abs()
        report[name] = {
            "pass": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
            "max_abs": delta.max().item(),
            "relative_l2": (torch.linalg.vector_norm(delta)
                            / torch.linalg.vector_norm(expected).clamp_min(1e-30)).item(),
        }
    return report


def make_case(args, op, layer_cls, fused_classes):
    device = torch.device("cuda")

    def rand(*shape):
        return (torch.randn(*shape, device=device, dtype=torch.float32) * .2).requires_grad_()

    nb, nl, nx = args.batch, args.nloc, args.nall
    nn, ne = nb * nl, nb * nl * args.neighbors
    dn, de, da = args.node_dim, args.edge_dim, args.angle_dim
    owners = torch.arange(nn, device=device).repeat_interleave(args.neighbors)
    ext = (owners // nl) * nx + torch.randint(nx, (ne,), device=device)
    layer = SimpleNamespace()
    if op == "sym":
        if nb != 1:
            raise ValueError("sym backward currently requires --batch 1")
        edge, h, sw = rand(ne, de), rand(ne, 3), rand(ne)
        normalization = args.sym_normalization if args.sym_normalization is not None else args.neighbors
        scale = normalization ** -.5
        layer._cal_hg_dynamic = layer_cls._cal_hg_dynamic
        layer._cal_grrg = layer_cls._cal_grrg
        y = layer_cls.symmetrization_op_dynamic(
            layer, edge, h, sw, owners, nn, nb, nl, scale, args.axis)
        with torch.no_grad():
            H = layer_cls._cal_hg_dynamic(edge, h, sw, owners, nn, nb, nl, scale)
        upstream = rand(*y.shape).contiguous()
        inputs = (edge, h, sw)
        names = ("edge", "h2", "sw")

        def fused():
            return fused_classes[0].apply(
                upstream, edge, h, sw, owners, H, nb, nl, nn, scale, args.axis)

        normalize = lambda result: result
        sizes = {"edges": ne, "owners": nn, "edge_dim": de, "axis": args.axis,
                 "scale_factor": scale}
    elif op == "edge":
        node, node_ext, edge = rand(nb, nl, dn), rand(nb, nx, dn), rand(ne, de)
        k = args.edge_out_dim
        matrix, bias = rand(2 * dn + de, k), rand(k)
        layer.node_edge_linear = SimpleNamespace(matrix=matrix, bias=bias)
        y = layer_cls.optim_edge_update_dynamic(layer, node, node_ext, edge, owners, ext)
        upstream = rand(*y.shape).contiguous()
        weights = matrix.split((dn, dn, de), dim=0)
        inputs = (node, node_ext, edge, matrix, bias)
        names = ("node", "node_ext", "edge", "matrix", "bias")
        flat_node, flat_ext = node.reshape(-1, dn), node_ext.reshape(-1, dn)

        def fused():
            return fused_classes[1].apply(
                upstream, flat_node, flat_ext, edge, owners, ext, *weights)

        def normalize(result):
            return (*result[:3], torch.cat(result[5:8], dim=0), result[8])

        sizes = {"edges": ne, "nodes": nn, "extended_nodes": nb * nx,
                 "node_dim": dn, "edge_dim": de, "out_dim": k}
    else:
        # Construct all ordered (j,k) pairs within each owner's angle neighbors.
        an = args.angle_neighbors
        counts = args.angle_counts if args.angle_counts is not None else [an] * nn
        ni = torch.cat([torch.full((count * count,), owner, device=device, dtype=torch.int64)
                        for owner, count in enumerate(counts)])
        local_j = torch.cat([torch.arange(count, device=device).repeat_interleave(count) for count in counts])
        local_k = torch.cat([torch.arange(count, device=device).repeat(count) for count in counts])
        ij, ik = ni * args.neighbors + local_j, ni * args.neighbors + local_k
        ma = ni.numel()
        angle, node, edge = rand(ma, da), rand(nb, nl, dn), rand(ne, de)
        k = args.angle_out_dim
        matrix, bias = rand(da + dn + 2 * de, k), rand(k)
        layer.edge_angle_linear1 = SimpleNamespace(matrix=matrix, bias=bias)
        y = layer_cls.optim_angle_update_dynamic(layer, angle, node, edge, ni, ij, ik)
        upstream = rand(*y.shape).contiguous()
        weights = matrix.split((da, dn, de, de), dim=0)
        inputs = (angle, node, edge, matrix, bias)
        names = ("angle", "node", "edge", "matrix", "bias")
        flat_node = node.reshape(-1, dn)

        def fused():
            return fused_classes[2].apply(
                upstream, angle, flat_node, edge, ni, ij, ik, *weights, node.shape)

        def normalize(result):
            return (*result[:3], torch.cat(result[3:7], dim=0), result[7])

        sizes = {"angles": ma, "edges": ne, "nodes": nn, "angle_dim": da,
                 "node_dim": dn, "edge_dim": de, "out_dim": k}

    def original():
        return torch.autograd.grad(
            y, inputs, grad_outputs=upstream, create_graph=True, retain_graph=True)

    return original, fused, normalize, names, sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ops", nargs="+", choices=("sym", "edge", "angle"), default=["sym", "edge", "angle"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--nloc", type=int, default=128)
    parser.add_argument("--nall", type=int, default=160)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--angle-neighbors", type=int, default=8)
    parser.add_argument("--angle-counts", type=int, nargs="+",
                        help="Optional per-owner angle neighbor counts, in flattened batch order")
    parser.add_argument("--sym-normalization", type=float,
                        help="Sym scale is this value ** -0.5; for the supplied config use 120")
    parser.add_argument("--node-dim", type=int, default=128)
    parser.add_argument("--edge-dim", type=int, default=16)
    parser.add_argument("--angle-dim", type=int, default=64)
    parser.add_argument("--axis", type=int, default=4)
    parser.add_argument("--edge-out-dim", type=int, default=128)
    parser.add_argument("--angle-out-dim", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--json", type=Path, help="Optional machine-readable result path")
    args = parser.parse_args()
    for name in ("batch", "nloc", "nall", "neighbors", "angle_neighbors", "node_dim",
                 "edge_dim", "angle_dim", "axis", "edge_out_dim", "angle_out_dim", "warmup", "repeat"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.axis > args.edge_dim or args.angle_neighbors > args.neighbors or args.nall < args.nloc:
        parser.error("Require axis <= edge-dim, angle-neighbors <= neighbors, nall >= nloc")
    if args.sym_normalization is not None and args.sym_normalization <= 0:
        parser.error("--sym-normalization must be positive")
    if args.angle_counts is not None:
        if (len(args.angle_counts) != args.batch * args.nloc
                or any(c < 0 or c > args.neighbors for c in args.angle_counts)
                or sum(args.angle_counts) == 0):
            parser.error("--angle-counts needs batch*nloc counts in [0, neighbors], with at least one nonzero")
    if not torch.cuda.is_available():
        parser.error("A CUDA-enabled PyTorch installation and GPU are required")
    # Delay project/TileLang imports so --help works without either installed.
    import tilelang
    from deepmd.pt.model.descriptor.repflow_layer import RepFlowLayer
    from deepmd.pt.model.descriptor.utils_tilelang import (
        FusedSymmetrizationOpDynamicBackward,
        FusedEdgeUpdateFunctionBackward,
        FusedAngleUpdateFunctionBackward,
    )

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    environment = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "tilelang": getattr(tilelang, "__version__", "unknown"),
                   "dtype": "float32", "torch_tf32": False,
                   "repflow_source": str(__import__(RepFlowLayer.__module__, fromlist=["__file__"]).__file__)}
    print(json.dumps(environment, indent=2))
    print("Backward only; original: create_graph=True, retain_graph=True; fused: Backward.apply")
    print("Original returns full matrix gradient; fused returns weight blocks (concat excluded).")
    print("Peak extra = allocated peak minus pre-call allocation; excludes saved primal graph.\n")
    classes = (FusedSymmetrizationOpDynamicBackward, FusedEdgeUpdateFunctionBackward,
               FusedAngleUpdateFunctionBackward)
    reports = {}
    for op in args.ops:
        original, fused, normalize, names, sizes = make_case(args, op, RepFlowLayer, classes)
        # This first call also compiles the fused kernels, before timing or memory measurement.
        reference, actual = original(), fused()
        correctness = check(reference, normalize(actual), names, args.atol, args.rtol)
        del reference, actual
        passed = all(item["pass"] for item in correctness.values())
        print(f"{op}: {sizes}; correctness={'PASS' if passed else 'FAIL'}", flush=True)
        for name, result in correctness.items():
            print(f"  {name}: {'PASS' if result['pass'] else 'FAIL'} max_abs={result['max_abs']:.3e} relative_l2={result['relative_l2']:.3e}")
        baseline = measure(original, args.warmup, args.repeat)
        optimized = measure(fused, args.warmup, args.repeat)
        speedup = baseline["cuda_median_ms"] / optimized["cuda_median_ms"]
        for label, result in (("original", baseline), ("fused", optimized)):
            print(f"  {label:8s} CUDA median={result['cuda_median_ms']:.4f} ms "
                  f"p90={result['cuda_p90_ms']:.4f} ms wall={result['wall_median_ms']:.4f} ms "
                  f"peak_extra={result['peak_extra_allocated_MiB']:.2f} MiB")
        print(f"  speedup={speedup:.3f}x" + (" (INVALID correctness; performance only)" if not passed else ""), flush=True)
        reports[op] = {"sizes": sizes, "correctness": correctness, "original": baseline,
                       "fused": optimized, "speedup": speedup, "valid": passed}
        del original, fused, normalize
        gc.collect()
        torch.cuda.synchronize()
    if args.json:
        args.json.write_text(json.dumps({"environment": environment,
                                        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                        "results": reports}, indent=2), encoding="utf-8")
    if not all(result["valid"] for result in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
