# benchmark_symmetrization.py

import torch

from utils_tilelang import FusedEdgeUpdateFunction

def optim_edge_update_dynamic(
    matrix: torch.Tensor,
    bias: torch.Tensor,
    node_ebd: torch.Tensor,
    node_ebd_ext: torch.Tensor,
    flat_edge_ebd: torch.Tensor,
    n2e_index: torch.Tensor,
    n_ext2e_index: torch.Tensor,
    feat: str = "node",
) -> torch.Tensor:
    assert bias is not None
    nf, nall, node_dim = node_ebd_ext.shape
    _, nloc, _ = node_ebd.shape
    edge_dim = flat_edge_ebd.shape[-1]
    # node_dim, node_dim, edge_dim
    node, node_ext, edge = torch.split(matrix, [node_dim, node_dim, edge_dim])

    # nf * nloc * node/edge_dim
    sub_node_update = torch.matmul(node_ebd, node)
    # n_edge * node/edge_dim
    sub_node_update = torch.index_select(
        sub_node_update.reshape(nf * nloc, sub_node_update.shape[-1]), 0, n2e_index
    )

    # nf * nall * node/edge_dim
    sub_node_ext_update = torch.matmul(node_ebd_ext, node_ext)
    # n_edge * node/edge_dim
    sub_node_ext_update = torch.index_select(
        sub_node_ext_update.reshape(nf * nall, sub_node_update.shape[-1]),
        0,
        n_ext2e_index,
    )

    # n_edge * node/edge_dim
    sub_edge_update = torch.matmul(flat_edge_ebd, edge)

    result_update = bias + sub_node_update + sub_edge_update + sub_node_ext_update

    return result_update

def fused_optim_edge_update_dynamic(
    matrix: torch.Tensor,
    bias: torch.Tensor,
    node_ebd: torch.Tensor,
    node_ebd_ext: torch.Tensor,
    flat_edge_ebd: torch.Tensor,
    n2e_index: torch.Tensor,
    n_ext2e_index: torch.Tensor,
    feat: str = "node",
) -> torch.Tensor:
    assert bias is not None

    nf, nall, node_dim = node_ebd_ext.shape
    _, nloc, _ = node_ebd.shape
    edge_dim = flat_edge_ebd.shape[-1]

    # node_dim, node_dim, edge_dim
    node, node_ext, edge = torch.split(matrix, [node_dim, node_dim, edge_dim])

    result_update = FusedEdgeUpdateFunction.apply(
        node_ebd.reshape(-1, node_dim),
        node_ebd_ext.reshape(-1, node_dim),
        flat_edge_ebd,
        n2e_index,
        n_ext2e_index,
        node,
        node_ext,
        edge,
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
        "/workspace/DeepModelingCommunity/fused_optim_edge_real_input.pt",
        map_location=device,
    )

    data1 = torch.load(
        "/workspace/DeepModelingCommunity/fused_optim_edge_real_input_1.pt",
        map_location=device,
    )
    
    node_ebd = data["node_ebd"].cuda()
    node_ebd_ext = data["node_ebd_ext"].cuda()
    edge_ebd = data["edge_ebd"].cuda()
    n2e_index = data["n2e_index"].cuda()
    n_ext2e_index = data["n_ext2e_index"].cuda()
    matrix = data1["matrix"].cuda()
    bias = data1["bias"].cuda()

    print("========== Input ==========")
    print("node_ebd:", node_ebd.shape, node_ebd.dtype)
    print("node_ebd_ext:", node_ebd_ext.shape, node_ebd_ext.dtype)
    print("edge_ebd:", edge_ebd.shape, edge_ebd.dtype)
    print("n2e_index:", n2e_index.shape, n2e_index.dtype)
    print("n_ext2e_index:", n_ext2e_index.shape, n_ext2e_index.dtype)
    print("matrix:", matrix.shape, matrix.dtype)
    print("bias:", bias.shape, bias.dtype)
    print("============================")

    # ============================================================
    # Original
    # ============================================================
    def original_fn():
        return optim_edge_update_dynamic(
            matrix,
            bias,
            node_ebd,
            node_ebd_ext,
            edge_ebd,
            n2e_index,
            n_ext2e_index,
            "node",
        )

    # ============================================================
    # Fused
    # ============================================================
    def fused_fn():
        return fused_optim_edge_update_dynamic(
            matrix,
            bias,
            node_ebd,
            node_ebd_ext,
            edge_ebd,
            n2e_index,
            n_ext2e_index,
            "node",
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
