"""Compare the expanded Sym block with its direct PyTorch expression.

Run in the CUDA/TileLang environment with::

    python -m deepmd.pt.model.descriptor.check_expanded_sym --segmented

The report separates geometry/Gram, projection, and complete-block gradients.
"""

from __future__ import annotations

import argparse

import torch

from deepmd.pt.model.descriptor.utils_tilelang import (
    FusedDualSymGeometryFunction,
    FusedSymProjectionActResidualFunction,
    _sym_owner_metadata,
)


def silut(
    value: torch.Tensor,
    threshold: float,
    slope: float,
    const_value: float,
) -> torch.Tensor:
    return torch.where(
        value >= threshold,
        torch.tanh(slope * (value - threshold)) + const_value,
        value * torch.sigmoid(value),
    )


def geometry_reference(
    edge_ebd: torch.Tensor,
    node_ebd_ext: torch.Tensor,
    h2: torch.Tensor,
    sw: torch.Tensor,
    owner: torch.Tensor,
    n_ext2e_index: torch.Tensor,
    num_owner: int,
    scale: float,
    axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_term = h2[:, :, None] * (edge_ebd * sw[:, None])[:, None, :]
    node_term = h2[:, :, None] * (
        node_ebd_ext[n_ext2e_index] * sw[:, None]
    )[:, None, :]
    h_edge = torch.zeros(
        num_owner, 3, edge_ebd.shape[1], device=edge_ebd.device
    ).index_add(0, owner, edge_term)
    h_node = torch.zeros(
        num_owner, 3, node_ebd_ext.shape[1], device=edge_ebd.device
    ).index_add(0, owner, node_term)
    h_edge = h_edge * scale
    h_node = h_node * scale
    q_edge = (h_edge[:, :, :axis].transpose(1, 2) @ h_edge / 3.0).flatten(1)
    q_node = (h_node[:, :, :axis].transpose(1, 2) @ h_node / 3.0).flatten(1)
    return q_edge, q_node


def projection_reference(
    node_base: torch.Tensor,
    q_edge: torch.Tensor,
    q_node: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    threshold: float,
    slope: float,
    const_value: float,
) -> torch.Tensor:
    value = torch.cat((q_edge, q_node), dim=-1) @ weight + bias
    return node_base + residual * silut(value, threshold, slope, const_value)


def error_line(name: str, actual: torch.Tensor, expected: torch.Tensor) -> bool:
    delta = (actual - expected).float()
    expected_norm = torch.linalg.vector_norm(expected.float())
    relative_l2 = torch.linalg.vector_norm(delta) / expected_norm.clamp_min(1.0e-20)
    max_abs = delta.abs().max() if delta.numel() else delta.new_zeros(())
    finite = bool(torch.isfinite(actual).all())
    close = torch.allclose(actual, expected, atol=3.0e-3, rtol=7.0e-3)
    print(
        f"{name:>25}: max_abs={max_abs.item():.6e}  "
        f"relative_l2={relative_l2.item():.6e}  finite={finite}  "
        f"{'PASS' if close else 'FAIL'}"
    )
    return close


def compare_gradients(
    prefix: str,
    actual_output: tuple[torch.Tensor, ...] | torch.Tensor,
    expected_output: tuple[torch.Tensor, ...] | torch.Tensor,
    inputs: list[torch.Tensor],
    names: list[str],
) -> bool:
    actual_tuple = actual_output if isinstance(actual_output, tuple) else (actual_output,)
    expected_tuple = (
        expected_output if isinstance(expected_output, tuple) else (expected_output,)
    )
    grad_outputs = tuple(torch.randn_like(item) for item in expected_tuple)
    expected_grads = torch.autograd.grad(
        expected_tuple, inputs, grad_outputs=grad_outputs, retain_graph=True
    )
    actual_grads = torch.autograd.grad(
        actual_tuple, inputs, grad_outputs=grad_outputs, retain_graph=True
    )
    ok = True
    for name, actual, expected in zip(names, actual_grads, expected_grads):
        ok = error_line(f"{prefix}_{name}", actual, expected) and ok
    return ok


def make_owner(num_owner: int, edges_per_owner: int, segmented: bool) -> torch.Tensor:
    owner = torch.arange(num_owner, device="cuda", dtype=torch.int64).repeat_interleave(
        edges_per_owner
    )
    if not segmented:
        return owner
    counts = torch.full(
        (num_owner,), edges_per_owner, device="cuda", dtype=torch.int64
    )
    shift = max(1, edges_per_owner // 2)
    counts[0] -= shift
    counts[-1] += shift
    owner = torch.arange(num_owner, device="cuda", dtype=torch.int64).repeat_interleave(
        counts
    )
    return owner[torch.randperm(owner.numel(), device="cuda")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owners", type=int, default=8)
    parser.add_argument("--edges-per-owner", type=int, default=120)
    parser.add_argument("--extended-nodes", type=int, default=37)
    parser.add_argument("--axis", type=int, default=16)
    parser.add_argument("--segmented", action="store_true")
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This check requires the CUDA/TileLang training environment")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    c_node, c_edge = 128, 64
    owner = make_owner(args.owners, args.edges_per_owner, args.segmented)
    num_edges = owner.numel()
    n_ext2e_index = torch.randint(
        args.extended_nodes, (num_edges,), device="cuda", dtype=torch.int64
    )

    def variable(*shape: int, scale: float = 0.15) -> torch.Tensor:
        return (torch.randn(*shape, device="cuda") * scale).requires_grad_()

    node_base = variable(args.owners, c_node)
    edge_ebd = variable(num_edges, c_edge)
    node_ebd_ext = variable(args.extended_nodes, c_node)
    h2 = variable(num_edges, 3)
    sw = variable(num_edges, scale=0.5)
    d_projection = args.axis * (c_edge + c_node)
    weight = variable(d_projection, c_node, scale=0.02)
    bias = variable(c_node, scale=0.05)
    residual = variable(c_node, scale=0.1)
    geometry_inputs = [edge_ebd, node_ebd_ext, h2, sw]
    all_inputs = [node_base, *geometry_inputs, weight, bias, residual]

    threshold = 3.0
    sigmoid_threshold = torch.sigmoid(torch.tensor(threshold)).item()
    slope = sigmoid_threshold * (1.0 + threshold * (1.0 - sigmoid_threshold))
    const_value = threshold * sigmoid_threshold
    scale = 1.0 / (120.0**0.5)
    metadata = _sym_owner_metadata(owner, args.owners)

    ref_q = geometry_reference(
        edge_ebd, node_ebd_ext, h2, sw, owner, n_ext2e_index,
        args.owners, scale, args.axis,
    )
    fused_q = FusedDualSymGeometryFunction.apply(
        edge_ebd, node_ebd_ext, h2, sw, owner, n_ext2e_index,
        args.owners, scale, args.axis, metadata,
    )
    print(
        f"case: owners={args.owners}, edges={num_edges}, axis={args.axis}, "
        f"segmented={args.segmented}"
    )
    ok = error_line("geometry_q_edge", fused_q[0], ref_q[0])
    ok = error_line("geometry_q_node", fused_q[1], ref_q[1]) and ok
    ok = compare_gradients(
        "geometry", fused_q, ref_q, geometry_inputs,
        ["grad_edge", "grad_node_ext", "grad_h2", "grad_sw"],
    ) and ok

    ref_projection = projection_reference(
        node_base, *ref_q, weight, bias, residual,
        threshold, slope, const_value,
    )
    fused_projection = FusedSymProjectionActResidualFunction.apply(
        node_base, *fused_q, weight, bias, residual,
        threshold, slope, const_value,
    )
    ok = error_line("complete_forward", fused_projection, ref_projection) and ok
    ok = compare_gradients(
        "complete", fused_projection, ref_projection, all_inputs,
        [
            "grad_node_base", "grad_edge", "grad_node_ext", "grad_h2",
            "grad_sw", "grad_weight", "grad_bias", "grad_residual",
        ],
    ) and ok
    torch.cuda.synchronize()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
