"""Numerically compare the expanded Edge operator with a PyTorch reference.

Run this script in the same CUDA/TileLang environment as training with
``python -m deepmd.pt.model.descriptor.check_expanded_edge``.  It checks the
forward values and every first-order gradient separately so a force error can
be assigned to a specific output of the custom backward.
"""

from __future__ import annotations

import argparse

import torch

from deepmd.pt.model.descriptor.utils_tilelang import (
    FusedEdgeBlockFunction,
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


def reference(
    node_partial: torch.Tensor,
    node_ebd: torch.Tensor,
    node_ebd_ext: torch.Tensor,
    edge_ebd: torch.Tensor,
    sw: torch.Tensor,
    owner: torch.Tensor,
    n_ext2e_index: torch.Tensor,
    node_weight: torch.Tensor,
    node_bias: torch.Tensor,
    node_residual: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_bias: torch.Tensor,
    edge_residual: torch.Tensor,
    scale: float,
    threshold: float,
    slope: float,
    const_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_info = torch.cat(
        (node_ebd[owner], node_ebd_ext[n_ext2e_index], edge_ebd), dim=-1
    )
    node_message = silut(
        edge_info @ node_weight + node_bias, threshold, slope, const_value
    )
    reduced = torch.zeros_like(node_partial).index_add(
        0, owner, sw[:, None] * node_message
    )
    node_output = node_partial + node_residual * (scale * reduced)
    edge_message = silut(
        edge_info @ edge_weight + edge_bias, threshold, slope, const_value
    )
    edge_output = edge_ebd + edge_residual * edge_message
    return node_output, edge_output


def error_line(name: str, actual: torch.Tensor, expected: torch.Tensor) -> bool:
    delta = (actual - expected).float()
    expected_norm = torch.linalg.vector_norm(expected.float())
    relative_l2 = torch.linalg.vector_norm(delta) / expected_norm.clamp_min(1.0e-20)
    max_abs = delta.abs().max() if delta.numel() else delta.new_zeros(())
    finite = bool(torch.isfinite(actual).all())
    close = torch.allclose(actual, expected, atol=2.0e-3, rtol=5.0e-3)
    print(
        f"{name:>22}: max_abs={max_abs.item():.6e}  "
        f"relative_l2={relative_l2.item():.6e}  finite={finite}  "
        f"{'PASS' if close else 'FAIL'}"
    )
    return close


def make_owner(num_owner: int, edges_per_owner: int, segmented: bool) -> torch.Tensor:
    if not segmented:
        return torch.arange(num_owner, device="cuda", dtype=torch.int64).repeat_interleave(
            edges_per_owner
        )
    total = num_owner * edges_per_owner
    counts = torch.full(
        (num_owner,), edges_per_owner, device="cuda", dtype=torch.int64
    )
    shift = max(1, edges_per_owner // 2)
    counts[0] -= shift
    counts[-1] += shift
    owner = torch.arange(num_owner, device="cuda", dtype=torch.int64).repeat_interleave(
        counts
    )
    # Exercise the metadata sorting path as well as unequal owner counts.
    return owner[torch.randperm(total, device="cuda")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owners", type=int, default=8)
    parser.add_argument("--edges-per-owner", type=int, default=120)
    parser.add_argument("--extended-nodes", type=int, default=37)
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
    d = 2 * c_node + c_edge
    owner = make_owner(args.owners, args.edges_per_owner, args.segmented)
    num_edges = owner.numel()
    n_ext2e_index = torch.randint(
        args.extended_nodes, (num_edges,), device="cuda", dtype=torch.int64
    )

    def variable(*shape: int, scale: float = 0.15) -> torch.Tensor:
        return (torch.randn(*shape, device="cuda") * scale).requires_grad_()

    inputs = [
        variable(args.owners, c_node),
        variable(args.owners, c_node),
        variable(args.extended_nodes, c_node),
        variable(num_edges, c_edge),
        variable(num_edges, scale=0.5),
        variable(d, c_node, scale=0.05),
        variable(c_node, scale=0.05),
        variable(c_node, scale=0.1),
        variable(d, c_edge, scale=0.05),
        variable(c_edge, scale=0.05),
        variable(c_edge, scale=0.1),
    ]
    names = [
        "grad_node_partial",
        "grad_node",
        "grad_node_ext",
        "grad_edge",
        "grad_sw",
        "grad_node_weight",
        "grad_node_bias",
        "grad_node_residual",
        "grad_edge_weight",
        "grad_edge_bias",
        "grad_edge_residual",
    ]
    (
        node_partial,
        node_ebd,
        node_ebd_ext,
        edge_ebd,
        sw,
        node_weight,
        node_bias,
        node_residual,
        edge_weight,
        edge_bias,
        edge_residual,
    ) = inputs
    threshold = 3.0
    threshold_sigmoid = torch.sigmoid(torch.tensor(threshold)).item()
    slope = threshold_sigmoid * (
        1.0 + threshold * (1.0 - threshold_sigmoid)
    )
    const_value = threshold * threshold_sigmoid
    scale = 1.0 / 120.0

    ref_output = reference(
        *inputs[:5], owner, n_ext2e_index, *inputs[5:], scale,
        threshold, slope, const_value,
    )
    fused_output = FusedEdgeBlockFunction.apply(
        *inputs[:5], owner, n_ext2e_index, *inputs[5:], args.owners, scale,
        threshold, slope, const_value, _sym_owner_metadata(owner, args.owners),
    )

    print(
        f"case: owners={args.owners}, edges={num_edges}, "
        f"segmented={args.segmented}"
    )
    ok = error_line("forward_node", fused_output[0], ref_output[0])
    ok = error_line("forward_edge", fused_output[1], ref_output[1]) and ok

    grad_node_output = torch.randn_like(ref_output[0])
    grad_edge_output = torch.randn_like(ref_output[1])
    ref_grads = torch.autograd.grad(
        ref_output,
        inputs,
        grad_outputs=(grad_node_output, grad_edge_output),
        retain_graph=True,
    )
    fused_grads = torch.autograd.grad(
        fused_output,
        inputs,
        grad_outputs=(grad_node_output, grad_edge_output),
        retain_graph=True,
    )
    for name, actual, expected in zip(names, fused_grads, ref_grads):
        ok = error_line(name, actual, expected) and ok
    torch.cuda.synchronize()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
