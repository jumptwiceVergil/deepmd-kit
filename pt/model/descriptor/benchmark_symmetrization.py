# benchmark_symmetrization.py

import torch

from deepmd.pt.model.descriptor.repflow_layer import RepFlowLayer


def benchmark(fn, warmup=20, repeat=100):
    # Warmup
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []

    for _ in range(repeat):
        start.record()

        fn()

        end.record()
        end.synchronize()

        times.append(start.elapsed_time(end))

    times.sort()

    n = len(times)

    return {
        "min": times[0],
        "median": times[n // 2],
        "mean": sum(times) / n,
        "p90": times[int(n * 0.90)],
        "p95": times[int(n * 0.95)],
    }


def main():
    device = "cuda"

    # ============================================================
    # Load real training input
    # ============================================================
    data = torch.load(
        "/workspace/DeepModelingCommunity/fused_sym_real_input.pt",
        map_location=device,
    )

    edge_ebd = data["edge_ebd"]
    h2 = data["h2"]
    sw = data["sw"]
    owner = data["owner"]

    num_owner = data["num_owner"]
    nb = data["nb"]
    nloc = data["nloc"]
    scale_factor = data["scale_factor"]
    axis_neuron = data["axis_neuron"]

    print("========== Input ==========")
    print("edge_ebd:", edge_ebd.shape, edge_ebd.dtype)
    print("h2:", h2.shape, h2.dtype)
    print("sw:", sw.shape, sw.dtype)
    print("owner:", owner.shape, owner.dtype)
    print("num_owner:", num_owner)
    print("nb:", nb)
    print("nloc:", nloc)
    print("scale_factor:", scale_factor)
    print("axis_neuron:", axis_neuron)
    print("============================")

    # ============================================================
    # Create RepFlowLayer instance
    # ============================================================
    layer = RepFlowLayer.__new__(RepFlowLayer)

    # ============================================================
    # Original
    # ============================================================
    def original_fn():
        return layer.symmetrization_op_dynamic(
            edge_ebd,
            h2,
            sw,
            owner=owner,
            num_owner=num_owner,
            nb=nb,
            nloc=nloc,
            scale_factor=scale_factor,
            axis_neuron=axis_neuron,
        )

    # ============================================================
    # Fused
    # ============================================================
    def fused_fn():
        return layer.fused_symmetrization_op_dynamic(
            edge_ebd,
            h2,
            sw,
            owner=owner,
            num_owner=num_owner,
            nb=nb,
            nloc=nloc,
            scale_factor=scale_factor,
            axis_neuron=axis_neuron,
        )

    # ============================================================
    # Correctness check
    # ============================================================
    print("\nChecking correctness...")

    out_original = original_fn()
    torch.cuda.synchronize()

    out_fused = fused_fn()
    torch.cuda.synchronize()

    diff = (out_original - out_fused).abs()

    max_abs_diff = diff.max().item()

    denominator = out_original.abs().clamp_min(1e-12)
    max_rel_diff = (diff / denominator).max().item()

    print("original shape:", out_original.shape)
    print("fused shape   :", out_fused.shape)
    print("max_abs_diff  :", max_abs_diff)
    print("max_rel_diff  :", max_rel_diff)

    # ============================================================
    # Benchmark
    # ============================================================
    print("\nBenchmarking original...")
    original_time = benchmark(original_fn)

    print("Benchmarking fused...")
    fused_time = benchmark(fused_fn)

    # ============================================================
    # Report
    # ============================================================
    print("\n========== Benchmark Result ==========")

    print("\nOriginal symmetrization_op_dynamic:")
    for key, value in original_time.items():
        print(f"  {key:>6}: {value:.4f} ms")

    print("\nFused fused_symmetrization_op_dynamic:")
    for key, value in fused_time.items():
        print(f"  {key:>6}: {value:.4f} ms")

    speedup = original_time["median"] / fused_time["median"]

    print("\n--------------------------------------")
    print(f"Speedup       : {speedup:.3f}x")
    print(
        f"Time reduction: "
        f"{(1.0 - fused_time['median'] / original_time['median']) * 100:.2f}%"
    )
    print("=======================================")


if __name__ == "__main__":
    main()
