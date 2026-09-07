# benchmark_symmetrization.py

import torch

from utils_tilelang import FusedAngleUpdateFunction

def optim_angle_update_dynamic(
    matrix: torch.Tensor,
    bias: torch.Tensor,
    flat_angle_ebd: torch.Tensor,
    node_ebd: torch.Tensor,
    flat_edge_ebd: torch.Tensor,
    n2a_index: torch.Tensor,
    eij2a_index: torch.Tensor,
    eik2a_index: torch.Tensor,
    feat: str = "edge",
) -> torch.Tensor:
    nf, nloc, node_dim = node_ebd.shape
    edge_dim = flat_edge_ebd.shape[-1]
    angle_dim = flat_angle_ebd.shape[-1]
    # angle_dim, node_dim, edge_dim, edge_dim
    sub_angle, sub_node, sub_edge_ik, sub_edge_ij = torch.split(
        matrix, [angle_dim, node_dim, edge_dim, edge_dim]
    )

    # n_angle * angle_dim
    sub_angle_update = torch.matmul(flat_angle_ebd, sub_angle)

    # nf * nloc * angle_dim
    sub_node_update = torch.matmul(node_ebd, sub_node)
    # n_angle * angle_dim
    sub_node_update = torch.index_select(
        sub_node_update.reshape(nf * nloc, sub_node_update.shape[-1]), 0, n2a_index
    )

    # n_edge * angle_dim
    sub_edge_update_ik = torch.matmul(flat_edge_ebd, sub_edge_ik)
    sub_edge_update_ij = torch.matmul(flat_edge_ebd, sub_edge_ij)
    # n_angle * angle_dim
    sub_edge_update_ik = torch.index_select(sub_edge_update_ik, 0, eik2a_index)
    sub_edge_update_ij = torch.index_select(sub_edge_update_ij, 0, eij2a_index)

    result_update = (
        bias
        + sub_node_update
        + sub_edge_update_ik
        + sub_edge_update_ij
        + sub_angle_update
    )
    return result_update

def fused_optim_angle_update_dynamic(
    matrix: torch.Tensor,
    bias: torch.Tensor,
    flat_angle_ebd: torch.Tensor,
    node_ebd: torch.Tensor,
    flat_edge_ebd: torch.Tensor,
    n2a_index: torch.Tensor,
    eij2a_index: torch.Tensor,
    eik2a_index: torch.Tensor,
    feat: str = "edge",
) -> torch.Tensor:
    nf, nloc, node_dim = node_ebd.shape
    edge_dim = flat_edge_ebd.shape[-1]
    angle_dim = flat_angle_ebd.shape[-1]

    sub_angle, sub_node, sub_edge_ik, sub_edge_ij = torch.split(
        matrix, [angle_dim, node_dim, edge_dim, edge_dim]
    )

    result_update = FusedAngleUpdateFunction.apply(
        flat_angle_ebd,
        node_ebd,
        flat_edge_ebd,
        n2a_index,
        eij2a_index,
        eik2a_index,
        sub_angle,
        sub_node,
        sub_edge_ik,
        sub_edge_ij,
        bias,
    )

    return result_update

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
        "/workspace/DeepModelingCommunity/fused_optim_angle_real_input.pt",
        map_location=device,
    )
    
    flat_angle_ebd = data["flat_angle_ebd"].cuda()
    node_ebd = data["node_ebd"].cuda()
    flat_edge_ebd = data["flat_edge_ebd"].cuda()
    n2a_index = data["n2a_index"].cuda()
    eij2a_index = data["eij2a_index"].cuda()
    eik2a_index = data["eik2a_index"].cuda()
    matrix = data["matrix"].cuda()
    bias = data["bias"].cuda()

    print("========== Input ==========")
    print("flat_angle_ebd:", flat_angle_ebd.shape, flat_angle_ebd.dtype)
    print("node_ebd:", node_ebd.shape, node_ebd.dtype)
    print("flat_edge_ebd:", flat_edge_ebd.shape, flat_edge_ebd.dtype)
    print("n2a_index:", n2a_index.shape, n2a_index.dtype)
    print("eij2a_index:", eij2a_index.shape, eij2a_index.dtype)
    print("eik2a_index:", eik2a_index.shape, eik2a_index.dtype)
    print("matrix:", matrix.shape, matrix.dtype)
    print("bias:", bias.shape, bias.dtype)
    print("============================")

    # ============================================================
    # Original
    # ============================================================
    def original_fn():
        return optim_angle_update_dynamic(
            matrix,
            bias,
            flat_angle_ebd,
            node_ebd,
            flat_edge_ebd,
            n2a_index,
            eij2a_index,
            eik2a_index,
            "edge"
        )

    # ============================================================
    # Fused
    # ============================================================
    def fused_fn():
        return fused_optim_angle_update_dynamic(
            matrix,
            bias,
            flat_angle_ebd,
            node_ebd,
            flat_edge_ebd,
            n2a_index,
            eij2a_index,
            eik2a_index,
            "edge"
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
