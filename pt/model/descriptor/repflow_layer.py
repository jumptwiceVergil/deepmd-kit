# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.descriptor.repformer_layer import (
    _apply_nlist_mask,
    _apply_switch,
    _make_nei_g1,
    get_residual,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.model.network.utils import (
    aggregate,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from torch.utils.cpp_extension import load

import tilelang
import tilelang.language as T

# tilelang fuse kernel
@tilelang.jit
def fused_edge_update_dynamic_jit(
    N_EDGES: int,
    # N_NODES_LOC: int,
    # N_NODES_EXT: int,
    NODE_DIM: int,
    EDGE_DIM: int,
    OUT_DIM: int,
    BLK_M: int = 128,
    BLK_N: int = 64,
    BLK_K: int = 64
):
    # BLK_K_MAX = max(BLK_K_NODE, BLK_K_EDGE)

    @T.prim_func
    def matmul_gather_edge(
        node_ebd: T.Buffer((N_EDGES, NODE_DIM), "float32"),
        node_ebd_ext: T.Buffer((N_EDGES, NODE_DIM), "float32"),
        flat_edge_ebd: T.Buffer((N_EDGES, EDGE_DIM), "float32"),
        node_weight: T.Buffer((NODE_DIM, OUT_DIM), "float32"),
        node_ext_weight: T.Buffer((NODE_DIM, OUT_DIM), "float32"),
        edge_weight: T.Buffer((EDGE_DIM, OUT_DIM), "float32"),
        # n2e_index: T.Buffer((N_EDGES,), "int64"),
        # n_ext2e_index: T.Buffer((N_EDGES,), "int64"),
        bias: T.Buffer((OUT_DIM,), "float32"),
        out: T.Buffer((N_EDGES, OUT_DIM), "float32"),
    ):
        with T.Kernel(T.ceildiv(N_EDGES, BLK_M), T.ceildiv(OUT_DIM, BLK_N), threads=128) as (bx, by):
            A = T.alloc_shared((BLK_M, BLK_K), "float32")
            B = T.alloc_shared((BLK_K, BLK_N), "float32")

            acc = T.alloc_fragment((BLK_M, BLK_N), "float32")
            T.clear(acc)

            # prefetch index
            # n2e_shm = T.alloc_shared((BLK_M,), "int64")
            # n_ext2e_shm = T.alloc_shared((BLK_M,), "int64")
            # for i in T.Parallel(BLK_M):
            #     row_idx = bx * BLK_M + i
            #     if row_idx < N_EDGES:
            #         n2e_shm[i] = n2e_index[row_idx]
            #         n_ext2e_shm[i] = n_ext2e_index[row_idx]
            #     else:
            #         n2e_shm[i] = T.int64(0)
            #         n_ext2e_shm[i] = T.int64(0)

            # calculate sub_node_update and add to acc
            for k in T.Pipelined(T.ceildiv(NODE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    row_idx = bx * BLK_M + i
                    col_idx = k * BLK_K + j
                    if row_idx < N_EDGES and col_idx < NODE_DIM:
                        # src_idx = n2e_shm[i]
                        A[i, j] = node_ebd[row_idx, col_idx]
                    else:
                        A[i, j] = T.float32(0)

                for i, j in T.Parallel(BLK_K, BLK_N):
                    row_idx = k * BLK_K + i
                    col_idx = by * BLK_N + j
                    if row_idx < NODE_DIM and col_idx < OUT_DIM:
                        B[i, j] = node_weight[row_idx, col_idx]
                    else:
                        B[i, j] = T.float32(0)

                T.gemm(A, B, acc)

            # calculate sub_node_ext_update and add to acc
            for k in T.Pipelined(T.ceildiv(NODE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    row_idx = bx * BLK_M + i
                    col_idx = k * BLK_K + j
                    if row_idx < N_EDGES and col_idx < NODE_DIM:
                        # src_idx = n_ext2e_shm[i]
                        A[i, j] = node_ebd_ext[row_idx, col_idx]
                    else:
                        A[i, j] = T.float32(0)

                for i, j in T.Parallel(BLK_K, BLK_N):
                    row_idx = k * BLK_K + i
                    col_idx = by * BLK_N + j
                    if row_idx < NODE_DIM and col_idx < OUT_DIM:
                        B[i, j] = node_ext_weight[row_idx, col_idx]
                    else:
                        B[i, j] = T.float32(0)

                T.gemm(A, B, acc)

            # calculate sub_edge_update and add to acc
            for k in T.Pipelined(T.ceildiv(EDGE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    row_idx = bx * BLK_M + i
                    col_idx = k * BLK_K + j
                    if row_idx < N_EDGES and col_idx < EDGE_DIM:
                        A[i, j] = flat_edge_ebd[row_idx, col_idx]
                    else:
                        A[i, j] = T.float32(0)

                for i, j in T.Parallel(BLK_K, BLK_N):
                    row_idx = k * BLK_K + i
                    col_idx = by * BLK_N + j
                    if row_idx < EDGE_DIM and col_idx < OUT_DIM:
                        B[i, j] = edge_weight[row_idx, col_idx]
                    else:
                        B[i, j] = T.float32(0)

                T.gemm(A, B, acc)

            # bias add to acc
            for i, j in T.Parallel(BLK_M, BLK_N):
                if bx * BLK_M + i < N_EDGES and by * BLK_N + j < OUT_DIM:
                    out[bx * BLK_M + i, by * BLK_N + j] = T.cast(acc[i, j], "float32") + bias[by * BLK_N + j]

    return matmul_gather_edge

@tilelang.jit
def fused_angle_update_jit(
    N_ANGLES: int,
    # N_NODES: int,
    # N_EDGES: int,
    ANGLE_DIM: int,
    NODE_DIM: int,
    EDGE_DIM: int,
    OUT_DIM: int,
    BLK_M: int = 128,
    BLK_N: int = 64,
    BLK_K: int = 64
):
    @T.prim_func
    def matmul_gather_angle(
        flat_angle_ebd: T.Buffer((N_ANGLES, ANGLE_DIM), "float32"),
        flat_node_ebd: T.Buffer((N_ANGLES, NODE_DIM), "float32"),
        flat_edge_ebd_ik: T.Buffer((N_ANGLES, EDGE_DIM), "float32"),
        flat_edge_ebd_ij: T.Buffer((N_ANGLES, EDGE_DIM), "float32"),
        sub_angle_w: T.Buffer((ANGLE_DIM, OUT_DIM), "float32"),
        sub_node_w: T.Buffer((NODE_DIM, OUT_DIM), "float32"),
        sub_edge_ik_w: T.Buffer((EDGE_DIM, OUT_DIM), "float32"),
        sub_edge_ij_w: T.Buffer((EDGE_DIM, OUT_DIM), "float32"),
        # n2a_index: T.Buffer((N_ANGLES,), "int64"),
        # eik2a_index: T.Buffer((N_ANGLES,), "int64"),
        # eij2a_index: T.Buffer((N_ANGLES,), "int64"),
        bias: T.Buffer((OUT_DIM,), "float32"),
        out: T.Buffer((N_ANGLES, OUT_DIM), "float32"),
    ):
        with T.Kernel(T.ceildiv(N_ANGLES, BLK_M), T.ceildiv(OUT_DIM, BLK_N), threads=128) as (bx, by):
            A_shared = T.alloc_shared((BLK_M, BLK_K), "float32")
            B_shared = T.alloc_shared((BLK_K, BLK_N), "float32")

            acc = T.alloc_fragment((BLK_M, BLK_N), "float32")
            T.clear(acc)

            # prefetch index
            # n2a_shm = T.alloc_shared((BLK_M,), "int64")
            # eik2a_shm = T.alloc_shared((BLK_M,), "int64")
            # eij2a_shm = T.alloc_shared((BLK_M,), "int64")
            # for i in T.Parallel(BLK_M):
            #     row_idx = bx * BLK_M + i
            #     if row_idx < N_ANGLES:
            #         n2a_shm[i] = n2a_index[row_idx]
            #         eik2a_shm[i] = eik2a_index[row_idx]
            #         eij2a_shm[i] = eij2a_index[row_idx]
            #     else:
            #         n2a_shm[i] = T.int64(0)
            #         eik2a_shm[i] = T.int64(0)
            #         eij2a_shm[i] = T.int64(0)

            for k in T.Pipelined(T.ceildiv(ANGLE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    row_idx = bx * BLK_M + i
                    col_idx = k * BLK_K + j
                    if row_idx < N_ANGLES and col_idx < ANGLE_DIM:
                        A_shared[i, j] = flat_angle_ebd[row_idx, col_idx]
                    else:
                        A_shared[i, j] = T.float32(0)

                for i, j in T.Parallel(BLK_K, BLK_N):
                    row_idx = k * BLK_K + i
                    col_idx = by * BLK_N + j
                    if row_idx < ANGLE_DIM and col_idx < OUT_DIM:
                        B_shared[i, j] = sub_angle_w[row_idx, col_idx]
                    else:
                        B_shared[i, j] = T.float32(0)

                T.gemm(A_shared, B_shared, acc)

            for k in T.Pipelined(T.ceildiv(NODE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    row_idx = bx * BLK_M + i
                    col_idx = k * BLK_K + j
                    if row_idx < N_ANGLES and col_idx < NODE_DIM:
                        # src_idx = n2a_shm[i]
                        A_shared[i, j] = flat_node_ebd[row_idx, col_idx]
                    else:
                        A_shared[i, j] = T.float32(0)

                for i, j in T.Parallel(BLK_K, BLK_N):
                    row_idx = k * BLK_K + i
                    col_idx = by * BLK_N + j
                    if row_idx < NODE_DIM and col_idx < OUT_DIM:
                        B_shared[i, j] = sub_node_w[row_idx, col_idx]
                    else:
                        B_shared[i, j] = T.float32(0)

                T.gemm(A_shared, B_shared, acc)

            for k in T.Pipelined(T.ceildiv(EDGE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    row_idx = bx * BLK_M + i
                    col_idx = k * BLK_K + j
                    if row_idx < N_ANGLES and col_idx < EDGE_DIM:
                        # src_idx = eik2a_shm[i]
                        A_shared[i, j] = flat_edge_ebd_ik[row_idx, col_idx]
                    else:
                        A_shared[i, j] = T.float32(0)

                for i, j in T.Parallel(BLK_K, BLK_N):
                    row_idx = k * BLK_K + i
                    col_idx = by * BLK_N + j
                    if row_idx < EDGE_DIM and col_idx < OUT_DIM:
                        B_shared[i, j] = sub_edge_ik_w[row_idx, col_idx]
                    else:
                        B_shared[i, j] = T.float32(0)

                T.gemm(A_shared, B_shared, acc)

            for k in T.Pipelined(T.ceildiv(EDGE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    row_idx = bx * BLK_M + i
                    col_idx = k * BLK_K + j
                    if row_idx < N_ANGLES and col_idx < EDGE_DIM:
                        # src_idx = eij2a_shm[i]
                        A_shared[i, j] = flat_edge_ebd_ij[row_idx, col_idx]
                    else:
                        A_shared[i, j] = T.float32(0)

                for i, j in T.Parallel(BLK_K, BLK_N):
                    row_idx = k * BLK_K + i
                    col_idx = by * BLK_N + j
                    if row_idx < EDGE_DIM and col_idx < OUT_DIM:
                        B_shared[i, j] = sub_edge_ij_w[row_idx, col_idx]
                    else:
                        B_shared[i, j] = T.float32(0)

                T.gemm(A_shared, B_shared, acc)

            for i, j in T.Parallel(BLK_M, BLK_N):
                if bx * BLK_M + i < N_ANGLES and by * BLK_N + j < OUT_DIM:
                    out[bx * BLK_M + i, by * BLK_N + j] = T.cast(acc[i, j], "float32") + bias[by * BLK_N + j]

    return matmul_gather_angle



@tilelang.jit
def fused_edge_update_forward(
    N_EDGES: int,
    N_NODES_LOC: int,
    N_NODES_EXT: int,
    NODE_DIM: int,
    EDGE_DIM: int,
    OUT_DIM: int,
    BLK_M: int = 128,
    BLK_N: int = 64,
    BLK_K: int = 64,
):
    @T.prim_func
    def kernel(
        node_ebd: T.Buffer((N_NODES_LOC, NODE_DIM), "float32"),
        node_ebd_ext: T.Buffer((N_NODES_EXT, NODE_DIM), "float32"),
        flat_edge_ebd: T.Buffer((N_EDGES, EDGE_DIM), "float32"),
        n2e_index: T.Buffer((N_EDGES,), "int64"),
        n_ext2e_index: T.Buffer((N_EDGES,), "int64"),
        node: T.Buffer((NODE_DIM, OUT_DIM), "float32"),
        node_ext: T.Buffer((NODE_DIM, OUT_DIM), "float32"),
        edge: T.Buffer((EDGE_DIM, OUT_DIM), "float32"),
        bias: T.Buffer((OUT_DIM,), "float32"),
        out: T.Buffer((N_EDGES, OUT_DIM), "float32"),
        # sub_node_update: T.Buffer((N_EDGES, OUT_DIM), "float32"),
    ):

        with T.Kernel(T.ceildiv(N_EDGES, BLK_M), T.ceildiv(OUT_DIM, BLK_N), threads=128) as (bx, by):
            A = T.alloc_shared((BLK_M, BLK_K), "float32")
            B = T.alloc_shared((BLK_K, BLK_N), "float32")
            acc = T.alloc_fragment((BLK_M, BLK_N), "float32")
            # sub_node_update = T.alloc_fragment((BLK_M, BLK_N), "float32")

            T.clear(acc)
            # T.clear(sub_node_update)

            # node
            for k in T.Pipelined(T.ceildiv(NODE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    edge_id = bx * BLK_M + i
                    k_id = k * BLK_K + j

                    if (edge_id < N_EDGES and k_id < NODE_DIM):
                        src = n2e_index[edge_id]
                        A[i, j] = node_ebd[src, k_id]
                    else:
                        A[i, j] = T.float32(0)

                for i, j in T.Parallel(BLK_K, BLK_N):
                    k_id = k * BLK_K + i
                    out_id = by * BLK_N + j

                    if (k_id < NODE_DIM and out_id < OUT_DIM):
                        B[i, j] = node[k_id, out_id]
                    else:
                        B[i, j] = T.float32(0)

                T.gemm(A, B, acc)
            
            # node_ext
            for k in T.Pipelined(T.ceildiv(NODE_DIM, BLK_K), num_stages=2):

                for i, j in T.Parallel(BLK_M, BLK_K):
                    edge_id = bx * BLK_M + i
                    k_id = k * BLK_K + j

                    if (edge_id < N_EDGES and k_id < NODE_DIM):
                        src = n_ext2e_index[edge_id]
                        A[i, j] = node_ebd_ext[src, k_id]
                    else:
                        A[i, j] = T.float32(0)

                for i, j in T.Parallel(BLK_K, BLK_N):
                    k_id = k * BLK_K + i
                    out_id = by * BLK_N + j

                    if (k_id < NODE_DIM and out_id < OUT_DIM):
                        B[i, j] = node_ext[k_id, out_id]
                    else:
                        B[i, j] = T.float32(0)

                T.gemm(A, B, acc)

            # edge
            for k in T.Pipelined(T.ceildiv(EDGE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    edge_id = bx * BLK_M + i
                    k_id = k * BLK_K + j

                    if (edge_id < N_EDGES and k_id < EDGE_DIM):
                        A[i, j] = flat_edge_ebd[edge_id, k_id]
                    else:
                        A[i, j] = T.float32(0)

                for i, j in T.Parallel(BLK_K, BLK_N):
                    k_id = k * BLK_K + i
                    out_id = by * BLK_N + j

                    if (k_id < EDGE_DIM and out_id < OUT_DIM):
                        B[i, j] = edge[k_id, out_id]
                    else:
                        B[i, j] = T.float32(0)

                T.gemm(A, B, acc)

            for i, j in T.Parallel(BLK_M, BLK_N):
                if (bx * BLK_M + i < N_EDGES and by * BLK_N + j < OUT_DIM):
                    out[bx * BLK_M + i, by * BLK_N + j] = acc[i, j] + bias[by * BLK_N + j]
    return kernel

@tilelang.jit
def fused_edge_update_backward(
    E,
    NODE,
    K,
    D,
    dtype="float32",
    accum_dtype="float32",
    block_M=32,
    block_N=64,
    block_K=32,
):
    """
    Fuse:

        grad_gathered_node =
            grad_out @ node_weight.T

        grad_node.index_add_(
            0, n2e_index, grad_gathered_node
        )

    and

        grad_gathered_node_ext =
            grad_out @ node_ext_weight.T

        grad_node_ext.index_add_(
            0, n_ext2e_index, grad_gathered_node_ext
        )

    Shapes:

        grad_out       : [E, K]
        node_weight    : [D, K]
        node_ext_weight: [D, K]

        n2e_index      : [E]
        n_ext2e_index  : [E]

        grad_node      : [NODE, D]
        grad_node_ext  : [NODE, D]
    """

    @T.prim_func
    def kernel(
        grad_out: T.Buffer((E, K), dtype),
        node_weight: T.Buffer((D, K), dtype),
        node_ext_weight: T.Buffer((D, K), dtype),
        n2e_index: T.Buffer((E,), T.int64),
        n_ext2e_index: T.Buffer((E,), T.int64),
        grad_node: T.Buffer((NODE, D), accum_dtype),
        grad_node_ext: T.Buffer((NODE, D), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, block_N), T.ceildiv(E, block_M), threads=128,) as (bx, by):
            grad_out_shared = T.alloc_shared((block_M, block_K), dtype)
            node_weight_shared = T.alloc_shared((block_N, block_K), dtype)
            node_ext_weight_shared = T.alloc_shared((block_N, block_K), dtype)
            
            grad_node_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            grad_node_ext_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(grad_node_local)
            T.clear(grad_node_ext_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                for i, k in T.Parallel(block_M, block_K):
                    edge_id = by * block_M + i
                    k_id = ko * block_K + k
                    if edge_id < E and k_id < K:
                        grad_out_shared[i, k] = grad_out[edge_id, k_id]
                    else:
                        grad_out_shared[i, k] = 0
                
                for j, k in T.Parallel(block_N, block_K):
                    dim_id = bx * block_N + j
                    k_id = ko * block_K + k
                    if dim_id < D and k_id < K:
                        node_weight_shared[j, k] = node_weight[dim_id, k_id]
                    else:
                        node_weight_shared[j, k] = 0
                
                for j, k in T.Parallel(block_N, block_K):
                    dim_id = bx * block_N + j
                    k_id = ko * block_K + k
                    if dim_id < D and k_id < K:
                        node_ext_weight_shared[j, k] = node_ext_weight[dim_id, k_id]
                    else:
                        node_ext_weight_shared[j, k] = 0

                T.gemm(grad_out_shared, node_weight_shared, grad_node_local, transpose_B=True)
                T.gemm(grad_out_shared, node_ext_weight_shared, grad_node_ext_local, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                edge_id = by * block_M + i
                dim_id = bx * block_N + j
                if edge_id < E and dim_id < D:
                    node_id = n2e_index[edge_id]
                    T.atomic_add(grad_node[node_id, dim_id], grad_node_local[i, j])
                    
                    node_ext_id = n_ext2e_index[edge_id]
                    T.atomic_add(grad_node_ext[node_ext_id, dim_id], grad_node_ext_local[i, j])
            
    return kernel

class FusedEdgeUpdateFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_ebd,
        node_ebd_ext,
        flat_edge_ebd,
        n2e_index,
        n_ext2e_index,
        node,
        node_ext,
        edge,
        bias,
    ):

        nf = 1

        n_nodes_loc = node_ebd.shape[0]
        n_nodes_ext = node_ebd_ext.shape[0]

        n_edges = flat_edge_ebd.shape[0]

        node_dim = node_ebd.shape[-1]
        edge_dim = flat_edge_ebd.shape[-1]
        out_dim = node.shape[-1]

        kernel = fused_edge_update_forward(
            N_EDGES=n_edges,
            N_NODES_LOC=n_nodes_loc,
            N_NODES_EXT=n_nodes_ext,
            NODE_DIM=node_dim,
            EDGE_DIM=edge_dim,
            OUT_DIM=out_dim,
        )

        out = torch.empty((n_edges, out_dim), device=node_ebd.device, dtype=node_ebd.dtype,)
        # sub_node_update = torch.empty((n_edges, out_dim), device=node_ebd.device, dtype=node_ebd.dtype)
        
        kernel(
            node_ebd,
            node_ebd_ext,
            flat_edge_ebd,
            n2e_index,
            n_ext2e_index,
            node,
            node_ext,
            edge,
            bias,
            out,
            # sub_node_update,
        )

        # torch.save(
        #     sub_node_update.detach().cpu(),
        #     "/workspace/DP/sub_node_update_fusion.pt"
        # )

        ctx.save_for_backward(
            node_ebd,
            node_ebd_ext,
            flat_edge_ebd,
            n2e_index,
            n_ext2e_index,
            node,
            node_ext,
            edge,
        )

        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            node_ebd,
            node_ebd_ext,
            flat_edge_ebd,
            n2e_index,
            n_ext2e_index,
            node_weight,
            node_ext_weight,
            edge_weight,
        ) = ctx.saved_tensors

        # print(f"node_ebd: {node_ebd.shape}")
        # print(f"node_ebd_ext: {node_ebd_ext.shape}")
        # print(f"flat_edge_ebd: {flat_edge_ebd.shape}")
        # print(f"grad_out: {grad_out.shape}")
        # print(f"n2e_index: {n2e_index.shape}")
        # print(f"n_ext2e_index: {n_ext2e_index.shape}")
        # print(f"node_weight: {node_weight.shape}")
        # print(f"node_ext_weight: {node_ext_weight.shape}")
        # print(f"edge_weight: {edge_weight.shape}")
        
        # grad_out: [E, K]
        E = grad_out.shape[0]
        K = grad_out.shape[1]
        # node_ebd: [N, D]
        N = node_ebd.shape[0]
        D = node_ebd.shape[1]

        assert node_ebd.shape == node_ebd_ext.shape
        assert node_weight.shape == (D, K)
        assert node_ext_weight.shape == (D, K)

        grad_node = torch.zeros_like(node_ebd)
        grad_node_ext = torch.zeros_like(node_ebd_ext)

        grad_edge_ebd = grad_out @ edge_weight.T

        grad_edge_weight = flat_edge_ebd.T @ grad_out

        gathered_node = node_ebd[n2e_index]

        grad_node_weight = gathered_node.T @ grad_out

        # grad_gathered_node = grad_out @ node_weight.T

        # grad_node.index_add_(
        #    0,
        #    n2e_index,
        #    grad_gathered_node,
        # )

        gathered_node_ext = node_ebd_ext[n_ext2e_index]

        grad_node_ext_weight = gathered_node_ext.T @ grad_out

        # grad_gathered_node_ext = grad_out @ node_ext_weight.T

        # grad_node_ext.index_add_(
        #     0,
        #     n_ext2e_index,
        #     grad_gathered_node_ext,
        # )
        
        edge_backward_kernel = fused_edge_update_backward(
            E=E,
            NODE=N,
            K=K,
            D=D,
            dtype="float32",
            accum_dtype="float32",
            block_M=32,
            block_N=64,
            block_K=32,
        )

        edge_backward_kernel(
            grad_out,
            node_weight,
            node_ext_weight,
            n2e_index,
            n_ext2e_index,
            grad_node,
            grad_node_ext,
        )
        
        print(f"grad_out:")
        print(f"requires_grad: {grad_out.requires_grad}")
        print(f"grad_fn: {grad_out.grad_fn}")
        
        print(f"grad_node:")
        print(f"requires_grad: {grad_node.requires_grad}")
        print(f"grad_fn: {grad_node.grad_fn}")

        print(f"grad_node_ext:")
        print(f"requires_grad: {grad_node_ext.requires_grad}")
        print(f"grad_fn: {grad_node_ext.grad_fn}")

        # bias
        grad_bias = grad_out.sum(dim=0)

        return (
            grad_node,
            grad_node_ext,
            grad_edge_ebd,
            None,
            None,
            grad_node_weight,
            grad_node_ext_weight,
            grad_edge_weight,
            grad_bias,
        )

@tilelang.jit
def fused_angle_update_forward(
    N_ANGLE: int,
    N_NODE: int,
    N_EDGE: int,
    ANGLE_DIM: int,
    NODE_DIM: int,
    EDGE_DIM: int,
    OUT_DIM: int,
    BLK_M: int = 128,
    BLK_N: int = 64,
    BLK_K: int = 64,
):
    @T.prim_func
    def fused_angle_update(
        flat_angle_ebd: T.Buffer((N_ANGLE, ANGLE_DIM), "float32"),
        node_ebd: T.Buffer((N_NODE, NODE_DIM), "float32"),
        flat_edge_ebd: T.Buffer((N_EDGE, EDGE_DIM), "float32"),
        n2a_index: T.Buffer((N_ANGLE,), "int64"),
        eij2a_index: T.Buffer((N_ANGLE,), "int64"),
        eik2a_index: T.Buffer((N_ANGLE,), "int64"),
        angle_weight: T.Buffer((ANGLE_DIM, OUT_DIM), "float32"),
        node_weight: T.Buffer((NODE_DIM, OUT_DIM), "float32"),
        edge_ik_weight: T.Buffer((EDGE_DIM, OUT_DIM), "float32"),
        edge_ij_weight: T.Buffer((EDGE_DIM, OUT_DIM), "float32"),
        bias: T.Buffer((OUT_DIM,), "float32"),
        out: T.Buffer((N_ANGLE, OUT_DIM), "float32"),
    ):
        with T.Kernel(T.ceildiv(N_ANGLE, BLK_M), T.ceildiv(OUT_DIM, BLK_N), threads=128) as (bx, by):
            A = T.alloc_shared((BLK_M, BLK_K), "float32")
            B = T.alloc_shared((BLK_K, BLK_N), "float32")

            acc = T.alloc_fragment((BLK_M, BLK_N), "float32")
            T.clear(acc)
            
            # sub_angle
            for k in T.Pipelined(T.ceildiv(ANGLE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    row_idx = bx * BLK_M + i
                    col_idx = k * BLK_K + j

                    if (row_idx < N_ANGLE and col_idx < ANGLE_DIM):
                        A[i, j] = flat_angle_ebd[row_idx, col_idx]
                    else:
                        A[i, j] = T.float32(0)

                for i, j in T.Parallel(BLK_K, BLK_N):
                    row_idx = k * BLK_K + i
                    col_idx = by * BLK_N + j

                    if (row_idx < ANGLE_DIM and col_idx < OUT_DIM):
                        B[i, j] = angle_weight[row_idx, col_idx]
                    else:
                        B[i, j] = T.float32(0)

                T.gemm(A, B, acc)
            
            # sub_node
            for k in T.Pipelined(T.ceildiv(NODE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    angle_idx = bx * BLK_M + i
                    col_idx = k * BLK_K + j

                    if (angle_idx < N_ANGLE and col_idx < NODE_DIM):
                        node_idx = n2a_index[angle_idx]
                        A[i, j] = node_ebd[node_idx, col_idx]
                    else:
                        A[i, j] = T.float32(0)

                for i, j in T.Parallel(BLK_K, BLK_N):
                    row_idx = k * BLK_K + i
                    col_idx = by * BLK_N + j

                    if (row_idx < NODE_DIM and col_idx < OUT_DIM):
                        B[i, j] = node_weight[row_idx, col_idx]
                    else:
                        B[i, j] = T.float32(0)

                T.gemm(A, B, acc)
            
            # sub_edge_ik
            for k in T.Pipelined(T.ceildiv(EDGE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    angle_idx = bx * BLK_M + i
                    col_idx = k * BLK_K + j

                    if (angle_idx < N_ANGLE and col_idx < EDGE_DIM):
                        edge_idx = eik2a_index[angle_idx]
                        A[i, j] = flat_edge_ebd[edge_idx, col_idx]
                    else:
                        A[i, j] = T.float32(0)

                for i, j in T.Parallel(BLK_K, BLK_N):
                    row_idx = k * BLK_K + i
                    col_idx = by * BLK_N + j

                    if (row_idx < EDGE_DIM and col_idx < OUT_DIM):
                        B[i, j] = edge_ik_weight[row_idx, col_idx]
                    else:
                        B[i, j] = T.float32(0)

                T.gemm(A, B, acc)
            
            # sub_edge_ij
            for k in T.Pipelined(T.ceildiv(EDGE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    angle_idx = bx * BLK_M + i
                    col_idx = k * BLK_K + j

                    if (angle_idx < N_ANGLE and col_idx < EDGE_DIM):
                        edge_idx = eij2a_index[angle_idx]
                        A[i, j] = flat_edge_ebd[edge_idx, col_idx]
                    else:
                        A[i, j] = T.float32(0)

                for i, j in T.Parallel(BLK_K, BLK_N):
                    row_idx = k * BLK_K + i
                    col_idx = by * BLK_N + j

                    if (row_idx < EDGE_DIM and col_idx < OUT_DIM):
                        B[i, j] = edge_ij_weight[row_idx, col_idx]
                    else:
                        B[i, j] = T.float32(0)

                T.gemm(A, B, acc)

            for i, j in T.Parallel(BLK_M, BLK_N):
                if (bx * BLK_M + i < N_ANGLE and by * BLK_N + j < OUT_DIM):
                    out[bx * BLK_M + i, by * BLK_N + j] = T.cast(acc[i, j], "float32") + bias[by * BLK_N + j]
    return fused_angle_update

class FusedAngleUpdateFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
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
    ):
        nf, nloc, node_dim = node_ebd.shape
        angle_dim = flat_angle_ebd.shape[-1]
        edge_dim = flat_edge_ebd.shape[-1]
        n_angle = flat_angle_ebd.shape[0]
        n_node = nf * nloc
        n_edge = flat_edge_ebd.shape[0]
        
        out_dim = sub_angle.shape[-1]

        flat_node_ebd = node_ebd.reshape(n_node, node_dim).contiguous()
        flat_angle_ebd = flat_angle_ebd.contiguous()
        flat_edge_ebd = flat_edge_ebd.contiguous()

        n2a_index = n2a_index.contiguous()
        eij2a_index = eij2a_index.contiguous()
        eik2a_index = eik2a_index.contiguous()

        sub_angle = sub_angle.contiguous()
        sub_node = sub_node.contiguous()
        sub_edge_ik = sub_edge_ik.contiguous()
        sub_edge_ij = sub_edge_ij.contiguous()
        bias = bias.contiguous()

        result_update = torch.empty((n_angle, out_dim), device=flat_angle_ebd.device, dtype=flat_angle_ebd.dtype)

        kernel = fused_angle_update_forward(
            N_ANGLE=n_angle,
            N_NODE=n_node,
            N_EDGE=n_edge,
            ANGLE_DIM=angle_dim,
            NODE_DIM=node_dim,
            EDGE_DIM=edge_dim,
            OUT_DIM=out_dim,
        )

        kernel(
            flat_angle_ebd,
            flat_node_ebd,
            flat_edge_ebd,
            n2a_index,
            eij2a_index,
            eik2a_index,
            sub_angle,
            sub_node,
            sub_edge_ik,
            sub_edge_ij,
            bias,
            result_update,
        )

        ctx.save_for_backward(
            flat_angle_ebd,
            flat_node_ebd,
            flat_edge_ebd,
            n2a_index,
            eij2a_index,
            eik2a_index,
            sub_angle,
            sub_node,
            sub_edge_ik,
            sub_edge_ij,
        )
        ctx.node_ebd_shape = node_ebd.shape

        return result_update

    @staticmethod
    def backward(ctx, grad_output):
        (
            flat_angle_ebd,
            flat_node_ebd,
            flat_edge_ebd,
            n2a_index,
            eij2a_index,
            eik2a_index,
            sub_angle,
            sub_node,
            sub_edge_ik,
            sub_edge_ij,
        ) = ctx.saved_tensors

        grad_output = grad_output.contiguous()

        grad_flat_angle_ebd = torch.matmul(grad_output, sub_angle.transpose(0, 1))

        grad_gathered_node = torch.matmul(grad_output, sub_node.transpose(0, 1))

        grad_flat_node_ebd = torch.zeros(flat_node_ebd.shape, device=grad_output.device, dtype=grad_output.dtype)

        grad_flat_node_ebd.index_add_(
            0,
            n2a_index,
            grad_gathered_node,
        )

        grad_node_ebd = grad_flat_node_ebd.reshape(ctx.node_ebd_shape)
        grad_gathered_edge_ik = torch.matmul(grad_output, sub_edge_ik.transpose(0, 1))
        grad_gathered_edge_ij = torch.matmul(grad_output, sub_edge_ij.transpose(0, 1))

        grad_flat_edge_ebd = torch.zeros(flat_edge_ebd.shape, device=grad_output.device, dtype=grad_output.dtype)
        grad_flat_edge_ebd.index_add_(
            0,
            eik2a_index,
            grad_gathered_edge_ik,
        )

        grad_flat_edge_ebd.index_add_(
            0,
            eij2a_index,
            grad_gathered_edge_ij,
        )
        
        grad_sub_angle = torch.matmul(flat_angle_ebd.transpose(0, 1), grad_output)

        gathered_node = torch.index_select(
            flat_node_ebd,
            0,
            n2a_index,
        )
        grad_sub_node = torch.matmul(gathered_node.transpose(0, 1), grad_output)

        gathered_edge_ik = torch.index_select(
            flat_edge_ebd,
            0,
            eik2a_index,
        )

        grad_sub_edge_ik = torch.matmul(gathered_edge_ik.transpose(0, 1), grad_output)

        gathered_edge_ij = torch.index_select(
            flat_edge_ebd,
            0,
            eij2a_index,
        )

        grad_sub_edge_ij = torch.matmul(gathered_edge_ij.transpose(0, 1), grad_output)

        grad_bias = grad_output.sum(dim=0)

        return (
            grad_flat_angle_ebd,
            grad_node_ebd,
            grad_flat_edge_ebd,
            None,
            None,
            None,
            grad_sub_angle,
            grad_sub_node,
            grad_sub_edge_ik,
            grad_sub_edge_ij,
            grad_bias,
        )

class RepFlowLayer(torch.nn.Module):
    def __init__(
        self,
        e_rcut: float,
        e_rcut_smth: float,
        e_sel: int,
        a_rcut: float,
        a_rcut_smth: float,
        a_sel: int,
        ntypes: int,
        n_dim: int = 128,
        e_dim: int = 16,
        a_dim: int = 64,
        a_compress_rate: int = 0,
        a_compress_use_split: bool = False,
        a_compress_e_rate: int = 1,
        n_multi_edge_message: int = 1,
        axis_neuron: int = 4,
        update_angle: bool = True,
        optim_update: bool = True,
        use_dynamic_sel: bool = False,
        sel_reduce_factor: float = 10.0,
        smooth_edge_update: bool = False,
        activation_function: str = "silu",
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        precision: str = "float64",
        seed: Optional[Union[int, list[int]]] = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.epsilon = 1e-4  # protection of 1./nnei
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.ntypes = ntypes
        e_sel = [e_sel] if isinstance(e_sel, int) else e_sel
        self.nnei = sum(e_sel)
        assert len(e_sel) == 1
        self.e_sel = e_sel
        self.sec = self.e_sel
        self.a_rcut = a_rcut
        self.a_rcut_smth = a_rcut_smth
        self.a_sel = a_sel
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.a_compress_rate = a_compress_rate
        if a_compress_rate != 0:
            assert (a_dim * a_compress_e_rate) % (2 * a_compress_rate) == 0, (
                f"For a_compress_rate of {a_compress_rate}, a_dim*a_compress_e_rate must be divisible by {2 * a_compress_rate}. "
                f"Currently, a_dim={a_dim} and a_compress_e_rate={a_compress_e_rate} is not valid."
            )
        self.n_multi_edge_message = n_multi_edge_message
        assert self.n_multi_edge_message >= 1, "n_multi_edge_message must >= 1!"
        self.axis_neuron = axis_neuron
        self.update_angle = update_angle
        self.activation_function = activation_function
        self.act = ActivationFn(activation_function)
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.a_compress_e_rate = a_compress_e_rate
        self.a_compress_use_split = a_compress_use_split
        self.precision = precision
        self.seed = seed
        self.prec = PRECISION_DICT[precision]
        self.optim_update = optim_update
        self.smooth_edge_update = smooth_edge_update
        self.use_dynamic_sel = use_dynamic_sel
        self.sel_reduce_factor = sel_reduce_factor
        self.dynamic_e_sel = self.nnei / self.sel_reduce_factor
        self.dynamic_a_sel = self.a_sel / self.sel_reduce_factor

        assert update_residual_init in [
            "norm",
            "const",
        ], "'update_residual_init' only support 'norm' or 'const'!"

        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.n_residual = []
        self.e_residual = []
        self.a_residual = []
        self.edge_info_dim = self.n_dim * 2 + self.e_dim

        # node self mlp
        self.node_self_mlp = MLPLayer(
            n_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 0),
            trainable=trainable,
        )
        if self.update_style == "res_residual":
            self.n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 1),
                    trainable=trainable,
                )
            )

        # node sym (grrg + drrd)
        self.n_sym_dim = n_dim * self.axis_neuron + e_dim * self.axis_neuron
        self.node_sym_linear = MLPLayer(
            self.n_sym_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 2),
            trainable=trainable,
        )
        if self.update_style == "res_residual":
            self.n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 3),
                    trainable=trainable,
                )
            )

        # node edge message
        self.node_edge_linear = MLPLayer(
            self.edge_info_dim,
            self.n_multi_edge_message * n_dim,
            precision=precision,
            seed=child_seed(seed, 4),
            trainable=trainable,
        )
        if self.update_style == "res_residual":
            for head_index in range(self.n_multi_edge_message):
                self.n_residual.append(
                    get_residual(
                        n_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(child_seed(seed, 5), head_index),
                        trainable=trainable,
                    )
                )

        # edge self message
        self.edge_self_linear = MLPLayer(
            self.edge_info_dim,
            e_dim,
            precision=precision,
            seed=child_seed(seed, 6),
            trainable=trainable,
        )
        if self.update_style == "res_residual":
            self.e_residual.append(
                get_residual(
                    e_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 7),
                    trainable=trainable,
                )
            )

        if self.update_angle:
            self.angle_dim = self.a_dim
            if self.a_compress_rate == 0:
                # angle + node + edge * 2
                self.angle_dim += self.n_dim + 2 * self.e_dim
                self.a_compress_n_linear = None
                self.a_compress_e_linear = None
                self.e_a_compress_dim = e_dim
                self.n_a_compress_dim = n_dim
            else:
                # angle + a_dim/c + a_dim/2c * 2 * e_rate
                self.angle_dim += (1 + self.a_compress_e_rate) * (
                    self.a_dim // self.a_compress_rate
                )
                self.e_a_compress_dim = (
                    self.a_dim // (2 * self.a_compress_rate) * self.a_compress_e_rate
                )
                self.n_a_compress_dim = self.a_dim // self.a_compress_rate
                if not self.a_compress_use_split:
                    self.a_compress_n_linear = MLPLayer(
                        self.n_dim,
                        self.n_a_compress_dim,
                        precision=precision,
                        bias=False,
                        seed=child_seed(seed, 8),
                        trainable=trainable,
                    )
                    self.a_compress_e_linear = MLPLayer(
                        self.e_dim,
                        self.e_a_compress_dim,
                        precision=precision,
                        bias=False,
                        seed=child_seed(seed, 9),
                        trainable=trainable,
                    )
                else:
                    self.a_compress_n_linear = None
                    self.a_compress_e_linear = None

            # edge angle message
            self.edge_angle_linear1 = MLPLayer(
                self.angle_dim,
                self.e_dim,
                precision=precision,
                seed=child_seed(seed, 10),
                trainable=trainable,
            )
            self.edge_angle_linear2 = MLPLayer(
                self.e_dim,
                self.e_dim,
                precision=precision,
                seed=child_seed(seed, 11),
                trainable=trainable,
            )
            if self.update_style == "res_residual":
                self.e_residual.append(
                    get_residual(
                        self.e_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 12),
                        trainable=trainable,
                    )
                )

            # angle self message
            self.angle_self_linear = MLPLayer(
                self.angle_dim,
                self.a_dim,
                precision=precision,
                seed=child_seed(seed, 13),
                trainable=trainable,
            )
            if self.update_style == "res_residual":
                self.a_residual.append(
                    get_residual(
                        self.a_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 14),
                        trainable=trainable,
                    )
                )
        else:
            self.angle_self_linear = None
            self.edge_angle_linear1 = None
            self.edge_angle_linear2 = None
            self.a_compress_n_linear = None
            self.a_compress_e_linear = None
            self.angle_dim = 0

        self.n_residual = nn.ParameterList(self.n_residual)
        self.e_residual = nn.ParameterList(self.e_residual)
        self.a_residual = nn.ParameterList(self.a_residual)

    @staticmethod
    def _cal_hg(
        edge_ebd: torch.Tensor,
        h2: torch.Tensor,
        nlist_mask: torch.Tensor,
        sw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the transposed rotation matrix.

        Parameters
        ----------
        edge_ebd
            Neighbor-wise/Pair-wise edge embeddings, with shape nb x nloc x nnei x e_dim.
        h2
            Neighbor-wise/Pair-wise equivariant rep tensors, with shape nb x nloc x nnei x 3.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.

        Returns
        -------
        hg
            The transposed rotation matrix, with shape nb x nloc x 3 x e_dim.
        """
        # edge_ebd:  nb x nloc x nnei x e_dim
        # h2:  nb x nloc x nnei x 3
        # msk: nb x nloc x nnei
        nb, nloc, nnei, _ = edge_ebd.shape
        e_dim = edge_ebd.shape[-1]
        # nb x nloc x nnei x e_dim
        edge_ebd = _apply_nlist_mask(edge_ebd, nlist_mask)
        edge_ebd = _apply_switch(edge_ebd, sw)
        invnnei = torch.rsqrt(
            float(nnei)
            * torch.ones((nb, nloc, 1, 1), dtype=edge_ebd.dtype, device=edge_ebd.device)
        )
        # nb x nloc x 3 x e_dim
        h2g2 = torch.matmul(torch.transpose(h2, -1, -2), edge_ebd) * invnnei
        return h2g2

    @staticmethod
    def _cal_hg_dynamic(
        flat_edge_ebd: torch.Tensor,
        flat_h2: torch.Tensor,
        flat_sw: torch.Tensor,
        owner: torch.Tensor,
        num_owner: int,
        nb: int,
        nloc: int,
        scale_factor: float,
    ) -> torch.Tensor:
        """
        Calculate the transposed rotation matrix.

        Parameters
        ----------
        flat_edge_ebd
            Flatted neighbor-wise/pair-wise invariant rep tensors, with shape n_edge x e_dim.
        flat_h2
            Flatted neighbor-wise/pair-wise equivariant rep tensors, with shape n_edge x 3.
        flat_sw
            Flatted switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape n_edge.
        owner
            The owner index of the neighbor to reduce on.
        num_owner : int
            The total number of the owner.
        nb : int
            The number of batches.
        nloc : int
            The number of local atoms.
        scale_factor : float
            The scale factor to apply after reduce.

        Returns
        -------
        hg
            The transposed rotation matrix, with shape nf x nloc x 3 x e_dim.
        """
        n_edge, e_dim = flat_edge_ebd.shape
        # n_edge x e_dim
        flat_edge_ebd = flat_edge_ebd * flat_sw.unsqueeze(-1)
        # n_edge x 3 x e_dim
        flat_h2g2 = (flat_h2.unsqueeze(-1) * flat_edge_ebd.unsqueeze(-2)).reshape(
            -1, 3 * e_dim
        )
        # nf x nloc x 3 x e_dim
        h2g2 = (
            aggregate(flat_h2g2, owner, average=False, num_owner=num_owner).reshape(
                nb, nloc, 3, e_dim
            )
            * scale_factor
        )
        return h2g2

    @staticmethod
    def _cal_grrg(h2g2: torch.Tensor, axis_neuron: int) -> torch.Tensor:
        """
        Calculate the atomic invariant rep.

        Parameters
        ----------
        h2g2
            The transposed rotation matrix, with shape nb x nloc x 3 x e_dim.
        axis_neuron
            Size of the submatrix.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x e_dim)
        """
        # nb x nloc x 3 x e_dim
        nb, nloc, _, e_dim = h2g2.shape
        # nb x nloc x 3 x axis
        h2g2m = h2g2[..., :axis_neuron]
        # nb x nloc x axis x e_dim
        g1_13 = torch.matmul(torch.transpose(h2g2m, -1, -2), h2g2) / (3.0**1)
        # nb x nloc x (axisxng2)
        g1_13 = g1_13.view(nb, nloc, axis_neuron * e_dim)
        return g1_13

    def symmetrization_op(
        self,
        edge_ebd: torch.Tensor,
        h2: torch.Tensor,
        nlist_mask: torch.Tensor,
        sw: torch.Tensor,
        axis_neuron: int,
    ) -> torch.Tensor:
        """
        Symmetrization operator to obtain atomic invariant rep.

        Parameters
        ----------
        edge_ebd
            Neighbor-wise/Pair-wise invariant rep tensors, with shape nb x nloc x nnei x e_dim.
        h2
            Neighbor-wise/Pair-wise equivariant rep tensors, with shape nb x nloc x nnei x 3.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.
        axis_neuron
            Size of the submatrix.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x e_dim)
        """
        # edge_ebd:  nb x nloc x nnei x e_dim
        # h2:  nb x nloc x nnei x 3
        # msk: nb x nloc x nnei
        nb, nloc, nnei, _ = edge_ebd.shape
        # nb x nloc x 3 x e_dim
        h2g2 = self._cal_hg(
            edge_ebd,
            h2,
            nlist_mask,
            sw,
        )
        # nb x nloc x (axisxng2)
        g1_13 = self._cal_grrg(h2g2, axis_neuron)
        return g1_13

    def symmetrization_op_dynamic(
        self,
        flat_edge_ebd: torch.Tensor,
        flat_h2: torch.Tensor,
        flat_sw: torch.Tensor,
        owner: torch.Tensor,
        num_owner: int,
        nb: int,
        nloc: int,
        scale_factor: float,
        axis_neuron: int,
    ) -> torch.Tensor:
        """
        Symmetrization operator to obtain atomic invariant rep.

        Parameters
        ----------
        flat_edge_ebd
            Flatted neighbor-wise/pair-wise invariant rep tensors, with shape n_edge x e_dim.
        flat_h2
            Flatted neighbor-wise/pair-wise equivariant rep tensors, with shape n_edge x 3.
        flat_sw
            Flatted switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape n_edge.
        owner
            The owner index of the neighbor to reduce on.
        num_owner : int
            The total number of the owner.
        nb : int
            The number of batches.
        nloc : int
            The number of local atoms.
        scale_factor : float
            The scale factor to apply after reduce.
        axis_neuron
            Size of the submatrix.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x e_dim)
        """
        # nb x nloc x 3 x e_dim
        h2g2 = self._cal_hg_dynamic(
            flat_edge_ebd,
            flat_h2,
            flat_sw,
            owner,
            num_owner,
            nb,
            nloc,
            scale_factor,
        )
        # nb x nloc x (axis x e_dim)
        grrg = self._cal_grrg(h2g2, axis_neuron)
        return grrg

    def optim_angle_update(
        self,
        angle_ebd: torch.Tensor,
        node_ebd: torch.Tensor,
        edge_ebd: torch.Tensor,
        feat: str = "edge",
    ) -> torch.Tensor:
        if feat == "edge":
            assert self.edge_angle_linear1 is not None
            matrix, bias = self.edge_angle_linear1.matrix, self.edge_angle_linear1.bias
        elif feat == "angle":
            assert self.angle_self_linear is not None
            matrix, bias = self.angle_self_linear.matrix, self.angle_self_linear.bias
        else:
            raise NotImplementedError
        assert bias is not None

        angle_dim = angle_ebd.shape[-1]
        node_dim = node_ebd.shape[-1]
        edge_dim = edge_ebd.shape[-1]
        # angle_dim, node_dim, edge_dim, edge_dim
        sub_angle, sub_node, sub_edge_ik, sub_edge_ij = torch.split(
            matrix, [angle_dim, node_dim, edge_dim, edge_dim]
        )

        # nf * nloc * a_sel * a_sel * angle_dim
        sub_angle_update = torch.matmul(angle_ebd, sub_angle)
        # nf * nloc * angle_dim
        sub_node_update = torch.matmul(node_ebd, sub_node)
        # nf * nloc * a_nnei * angle_dim
        sub_edge_update_ik = torch.matmul(edge_ebd, sub_edge_ik)
        sub_edge_update_ij = torch.matmul(edge_ebd, sub_edge_ij)

        result_update = (
            bias
            + sub_node_update.unsqueeze(2).unsqueeze(3)
            + sub_edge_update_ik.unsqueeze(2)
            + sub_edge_update_ij.unsqueeze(3)
            + sub_angle_update
        )
        return result_update

    def optim_angle_update_dynamic(
        self,
        flat_angle_ebd: torch.Tensor,
        node_ebd: torch.Tensor,
        flat_edge_ebd: torch.Tensor,
        n2a_index: torch.Tensor,
        eij2a_index: torch.Tensor,
        eik2a_index: torch.Tensor,
        feat: str = "edge",
    ) -> torch.Tensor:
        if feat == "edge":
            matrix, bias = self.edge_angle_linear1.matrix, self.edge_angle_linear1.bias
        elif feat == "angle":
            matrix, bias = self.angle_self_linear.matrix, self.angle_self_linear.bias
        else:
            raise NotImplementedError
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
        self,
        flat_angle_ebd: torch.Tensor,
        node_ebd: torch.Tensor,
        flat_edge_ebd: torch.Tensor,
        n2a_index: torch.Tensor,
        eij2a_index: torch.Tensor,
        eik2a_index: torch.Tensor,
        feat: str = "edge",
    ) -> torch.Tensor:
        if feat == "edge":
            matrix, bias = self.edge_angle_linear1.matrix, self.edge_angle_linear1.bias
        elif feat == "angle":
            matrix, bias = self.angle_self_linear.matrix, self.angle_self_linear.bias
        else:
            raise NotImplementedError

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

    def optim_edge_update(
        self,
        node_ebd: torch.Tensor,
        node_ebd_ext: torch.Tensor,
        edge_ebd: torch.Tensor,
        nlist: torch.Tensor,
        feat: str = "node",
    ) -> torch.Tensor:
        if feat == "node":
            matrix, bias = self.node_edge_linear.matrix, self.node_edge_linear.bias
        elif feat == "edge":
            matrix, bias = self.edge_self_linear.matrix, self.edge_self_linear.bias
        else:
            raise NotImplementedError
        assert bias is not None

        node_dim = node_ebd.shape[-1]
        edge_dim = edge_ebd.shape[-1]
        # node_dim, node_dim, edge_dim
        node, node_ext, edge = torch.split(matrix, [node_dim, node_dim, edge_dim])

        # nf * nloc * node/edge_dim
        sub_node_update = torch.matmul(node_ebd, node)
        # nf * nall * node/edge_dim
        sub_node_ext_update = torch.matmul(node_ebd_ext, node_ext)
        # nf * nloc * nnei * node/edge_dim
        sub_node_ext_update = _make_nei_g1(sub_node_ext_update, nlist)
        # nf * nloc * nnei * node/edge_dim
        sub_edge_update = torch.matmul(edge_ebd, edge)

        result_update = (
            bias + sub_node_update.unsqueeze(2) + sub_edge_update + sub_node_ext_update
        )
        return result_update

    def optim_edge_update_dynamic(
        self,
        node_ebd: torch.Tensor,
        node_ebd_ext: torch.Tensor,
        flat_edge_ebd: torch.Tensor,
        n2e_index: torch.Tensor,
        n_ext2e_index: torch.Tensor,
        feat: str = "node",
    ) -> torch.Tensor:
        if feat == "node":
            matrix, bias = self.node_edge_linear.matrix, self.node_edge_linear.bias
        elif feat == "edge":
            matrix, bias = self.edge_self_linear.matrix, self.edge_self_linear.bias
        else:
            raise NotImplementedError
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
        # torch.save(
        #     sub_node_update.detach().cpu(),
        #     "/workspace/DP/sub_node_update_original.pt"
        # )
        return result_update

    def fused_optim_edge_update_dynamic(
        self,
        node_ebd: torch.Tensor,
        node_ebd_ext: torch.Tensor,
        flat_edge_ebd: torch.Tensor,
        n2e_index: torch.Tensor,
        n_ext2e_index: torch.Tensor,
        feat: str = "node",
    ) -> torch.Tensor:
        if feat == "node":
            matrix, bias = self.node_edge_linear.matrix, self.node_edge_linear.bias
        elif feat == "edge":
            matrix, bias = self.edge_self_linear.matrix, self.edge_self_linear.bias
        else:
            raise NotImplementedError
        assert bias is not None
        nf, nall, node_dim = node_ebd_ext.shape
        _, nloc, _ = node_ebd.shape
        edge_dim = flat_edge_ebd.shape[-1]

        n_nodes_loc = nf * nloc
        n_nodes_ext = nf * nall

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

    # a new pp split is to parallel the G1 and G2 of the same layer
    # the pp is like
    #   G1  G1
    #   G2  G2
    def forward_G1(
        self,
        node_ebd_ext: torch.Tensor,  # nf x nall x n_dim [OR] nf x nloc x n_dim when not parallel_mode
        edge_ebd: torch.Tensor,  # nf x nloc x nnei x e_dim
        h2: torch.Tensor,  # nf x nloc x nnei x 3
        nlist: torch.Tensor,  # nf x nloc x nnei
        nlist_mask: torch.Tensor,  # nf x nloc x nnei
        sw: torch.Tensor,  # switch func, nf x nloc x nnei
        a_nlist: torch.Tensor,  # nf x nloc x a_nnei
        a_nlist_mask: torch.Tensor,  # nf x nloc x a_nnei
        a_sw: torch.Tensor,  # switch func, nf x nloc x a_nnei
        edge_index: torch.Tensor,  # 2 x n_edge
        angle_index: torch.Tensor,  # 3 x n_angle
    ) -> torch.Tensor:
        nb, nloc, nnei = nlist.shape
        nall = node_ebd_ext.shape[1]
        node_ebd = node_ebd_ext[:, :nloc, :]
        assert (nb, nloc) == node_ebd.shape[:2]
        if not self.use_dynamic_sel:
            assert (nb, nloc, nnei, 3) == h2.shape
            n_edge = None
        else:
            # n_edge = int(nlist_mask.sum().item())
            # assert (n_edge, 3) == h2.shape
            n_edge = h2.shape[0]
        del a_nlist  # may be used in the future

        n2e_index, n_ext2e_index = edge_index[0], edge_index[1]
        n2a_index, eij2a_index, eik2a_index = (
            angle_index[0],
            angle_index[1],
            angle_index[2],
        )
        
        # nb x nloc x nnei x n_dim [OR] n_edge x n_dim
        with torch.cuda.nvtx.range("G1 graph gather"):
            nei_node_ebd = (
                _make_nei_g1(node_ebd_ext, nlist)
                if not self.use_dynamic_sel
                else torch.index_select(
                    node_ebd_ext.reshape(-1, self.n_dim), 0, n_ext2e_index
                )
            )

        n_update_list: list[torch.Tensor] = [node_ebd]
        e_update_list: list[torch.Tensor] = [edge_ebd]
        # a_update_list: list[torch.Tensor] = [angle_ebd]

        # node self mlp
        with torch.cuda.nvtx.range("node self update"):
            node_self_mlp = self.act(self.node_self_mlp(node_ebd))
            n_update_list.append(node_self_mlp)

        # node sym (grrg + drrd)
        torch.cuda.nvtx.range_push("geometry symmetrization")
        node_sym_list: list[torch.Tensor] = []
        node_sym_list.append(
            self.symmetrization_op(
                edge_ebd,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
            )
            if not self.use_dynamic_sel
            else self.symmetrization_op_dynamic(
                edge_ebd,
                h2,
                sw,
                owner=n2e_index,
                num_owner=nb * nloc,
                nb=nb,
                nloc=nloc,
                scale_factor=self.dynamic_e_sel ** (-0.5),
                axis_neuron=self.axis_neuron,
            )
        )
        node_sym_list.append(
            self.symmetrization_op(
                nei_node_ebd,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
            )
            if not self.use_dynamic_sel
            else self.symmetrization_op_dynamic(
                nei_node_ebd,
                h2,
                sw,
                owner=n2e_index,
                num_owner=nb * nloc,
                nb=nb,
                nloc=nloc,
                scale_factor=self.dynamic_e_sel ** (-0.5),
                axis_neuron=self.axis_neuron,
            )
        )
        node_sym = self.act(self.node_sym_linear(torch.cat(node_sym_list, dim=-1)))
        n_update_list.append(node_sym)
        torch.cuda.nvtx.range_pop()
        
        if not self.optim_update:
            if not self.use_dynamic_sel:
                # nb x nloc x nnei x (n_dim * 2 + e_dim)
                edge_info = torch.cat(
                    [
                        torch.tile(node_ebd.unsqueeze(-2), [1, 1, self.nnei, 1]),
                        nei_node_ebd,
                        edge_ebd,
                    ],
                    dim=-1,
                )
            else:
                # n_edge x (n_dim * 2 + e_dim)
                edge_info = torch.cat(
                    [
                        torch.index_select(
                            node_ebd.reshape(-1, self.n_dim), 0, n2e_index
                        ),
                        nei_node_ebd,
                        edge_ebd,
                    ],
                    dim=-1,
                )
        else:
            edge_info = None

        # node edge message
        # nb x nloc x nnei x (h * n_dim)
        torch.cuda.nvtx.range_push("G1 message pass")
        if not self.optim_update:
            assert edge_info is not None
            node_edge_update = self.act(
                self.node_edge_linear(edge_info)
            ) * sw.unsqueeze(-1)
        else:
            node_edge_update = self.fused_optim_edge_update_dynamic(
                node_ebd=node_ebd,
                node_ebd_ext=node_ebd_ext,
                flat_edge_ebd=edge_ebd,
                n2e_index=n2e_index,
                n_ext2e_index=n_ext2e_index,
                sw=sw,
                num_owner=nb * nloc,
                feat="node"
            )   
        node_edge_update = (
            (torch.sum(node_edge_update, dim=-2) / self.nnei)
            if not self.use_dynamic_sel
            else (
                aggregate(
                    node_edge_update,
                    n2e_index,
                    average=False,
                    num_owner=nb * nloc,
                ).reshape(nb, nloc, node_edge_update.shape[-1])
                / self.dynamic_e_sel
            )
        )
        torch.cuda.nvtx.range_pop()

        if self.n_multi_edge_message > 1:
            # nb x nloc x h x n_dim
            node_edge_update_mul_head = node_edge_update.view(
                nb, nloc, self.n_multi_edge_message, self.n_dim
            )
            for head_index in range(self.n_multi_edge_message):
                n_update_list.append(node_edge_update_mul_head[..., head_index, :])
        else:
            n_update_list.append(node_edge_update)
        # update node_ebd
        with torch.cuda.nvtx.range("node update"):
            n_updated = self.list_update(n_update_list, "node")

        return n_updated

    def forward_G2(
        self,
        node_ebd_ext: torch.Tensor,  # nf x nall x n_dim [OR] nf x nloc x n_dim when not parallel_mode
        edge_ebd: torch.Tensor,  # nf x nloc x nnei x e_dim
        h2: torch.Tensor,  # nf x nloc x nnei x 3
        angle_ebd: torch.Tensor,  # nf x nloc x a_nnei x a_nnei x a_dim
        nlist: torch.Tensor,  # nf x nloc x nnei
        nlist_mask: torch.Tensor,  # nf x nloc x nnei
        sw: torch.Tensor,  # switch func, nf x nloc x nnei
        a_nlist: torch.Tensor,  # nf x nloc x a_nnei
        a_nlist_mask: torch.Tensor,  # nf x nloc x a_nnei
        a_sw: torch.Tensor,  # switch func, nf x nloc x a_nnei
        edge_index: torch.Tensor,  # 2 x n_edge
        angle_index: torch.Tensor,  # 3 x n_angle
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        nb, nloc, nnei = nlist.shape
        node_ebd = node_ebd_ext[:, :nloc, :]
        assert (nb, nloc) == node_ebd.shape[:2]
        if not self.use_dynamic_sel:
            assert (nb, nloc, nnei, 3) == h2.shape
            n_edge = None
        else:
            # n_edge = int(nlist_mask.sum().item())
            # assert (n_edge, 3) == h2.shape
            n_edge = h2.shape[0]
        del a_nlist  # may be used in the future

        n2e_index, n_ext2e_index = edge_index[0], edge_index[1]
        n2a_index, eij2a_index, eik2a_index = (
            angle_index[0],
            angle_index[1],
            angle_index[2],
        )
        
        nei_node_ebd = (
            _make_nei_g1(node_ebd_ext, nlist)
            if not self.use_dynamic_sel
            else torch.index_select(
                node_ebd_ext.reshape(-1, self.n_dim), 0, n_ext2e_index
            )
        )

        e_update_list: list[torch.Tensor] = [edge_ebd]
        a_update_list: list[torch.Tensor] = [angle_ebd]

        if not self.optim_update:
            if not self.use_dynamic_sel:
                # nb x nloc x nnei x (n_dim * 2 + e_dim)
                edge_info = torch.cat(
                    [
                        torch.tile(node_ebd.unsqueeze(-2), [1, 1, self.nnei, 1]),
                        nei_node_ebd,
                        edge_ebd,
                    ],
                    dim=-1,
                )
            else:
                # n_edge x (n_dim * 2 + e_dim)
                edge_info = torch.cat(
                    [
                        torch.index_select(
                            node_ebd.reshape(-1, self.n_dim), 0, n2e_index
                        ),
                        nei_node_ebd,
                        edge_ebd,
                    ],
                    dim=-1,
                )
        else:
            edge_info = None

        # edge self message
        torch.cuda.nvtx.range_push("atom to bond")
        if not self.optim_update:
            assert edge_info is not None
            edge_self_update = self.act(self.edge_self_linear(edge_info))
        else:
            edge_self_update = self.act(
                self.optim_edge_update(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    nlist,
                    "edge",
                )
                if not self.use_dynamic_sel
                else self.optim_edge_update_dynamic(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    n2e_index,
                    n_ext2e_index,
                    "edge",
                )
            )
        e_update_list.append(edge_self_update)
        torch.cuda.nvtx.range_pop()
        
        if self.update_angle:
            assert self.angle_self_linear is not None
            assert self.edge_angle_linear1 is not None
            assert self.edge_angle_linear2 is not None
            # get angle info
            if self.a_compress_rate != 0:
                if not self.a_compress_use_split:
                    assert self.a_compress_n_linear is not None
                    assert self.a_compress_e_linear is not None
                    node_ebd_for_angle = self.a_compress_n_linear(node_ebd)
                    edge_ebd_for_angle = self.a_compress_e_linear(edge_ebd)
                else:
                    # use the first a_compress_dim dim for node and edge
                    node_ebd_for_angle = node_ebd[..., : self.n_a_compress_dim]
                    edge_ebd_for_angle = edge_ebd[..., : self.e_a_compress_dim]
            else:
                node_ebd_for_angle = node_ebd
                edge_ebd_for_angle = edge_ebd

            if not self.use_dynamic_sel:
                # nb x nloc x a_nnei x e_dim
                edge_ebd_for_angle = edge_ebd_for_angle[..., : self.a_sel, :]
                # nb x nloc x a_nnei x e_dim
                edge_ebd_for_angle = torch.where(
                    a_nlist_mask.unsqueeze(-1), edge_ebd_for_angle, 0.0
                )
            if not self.optim_update:
                # nb x nloc x a_nnei x a_nnei x n_dim [OR] n_angle x n_dim
                node_for_angle_info = (
                    torch.tile(
                        node_ebd_for_angle.unsqueeze(2).unsqueeze(2),
                        (1, 1, self.a_sel, self.a_sel, 1),
                    )
                    if not self.use_dynamic_sel
                    else torch.index_select(
                        node_ebd_for_angle.reshape(-1, self.n_a_compress_dim),
                        0,
                        n2a_index,
                    )
                )

                # nb x nloc x (a_nnei) x a_nnei x e_dim [OR] n_angle x e_dim
                edge_for_angle_k = (
                    torch.tile(
                        edge_ebd_for_angle.unsqueeze(2), (1, 1, self.a_sel, 1, 1)
                    )
                    if not self.use_dynamic_sel
                    else torch.index_select(edge_ebd_for_angle, 0, eik2a_index)
                )
                # nb x nloc x a_nnei x (a_nnei) x e_dim [OR] n_angle x e_dim
                edge_for_angle_j = (
                    torch.tile(
                        edge_ebd_for_angle.unsqueeze(3), (1, 1, 1, self.a_sel, 1)
                    )
                    if not self.use_dynamic_sel
                    else torch.index_select(edge_ebd_for_angle, 0, eij2a_index)
                )
                # nb x nloc x a_nnei x a_nnei x (e_dim + e_dim) [OR] n_angle x (e_dim + e_dim)
                edge_for_angle_info = torch.cat(
                    [edge_for_angle_k, edge_for_angle_j], dim=-1
                )
                angle_info_list = [angle_ebd]
                angle_info_list.append(node_for_angle_info)
                angle_info_list.append(edge_for_angle_info)
                # nb x nloc x a_nnei x a_nnei x (a + n_dim + e_dim*2) or (a + a/c + a/c)
                # [OR]
                # n_angle x (a + n_dim + e_dim*2) or (a + a/c + a/c)
                angle_info = torch.cat(angle_info_list, dim=-1)
            else:
                angle_info = None

            # edge angle message
            # nb x nloc x a_nnei x a_nnei x e_dim [OR] n_angle x e_dim
            torch.cuda.nvtx.range_push("G2 message pass")
            if not self.optim_update:
                assert angle_info is not None
                edge_angle_update = self.act(self.edge_angle_linear1(angle_info))
            else:
                edge_angle_update = self.act(
                    self.optim_angle_update(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        "edge",
                    )
                    if not self.use_dynamic_sel
                    else self.optim_angle_update_dynamic(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        n2a_index,
                        eij2a_index,
                        eik2a_index,
                        "edge",
                    )
                )
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("edge update")
            if not self.use_dynamic_sel:
                # nb x nloc x a_nnei x a_nnei x e_dim
                weighted_edge_angle_update = (
                    a_sw.unsqueeze(-1).unsqueeze(-1)
                    * a_sw.unsqueeze(-2).unsqueeze(-1)
                    * edge_angle_update
                )
                # nb x nloc x a_nnei x e_dim
                reduced_edge_angle_update = torch.sum(
                    weighted_edge_angle_update, dim=-2
                ) / (self.a_sel**0.5)
                # nb x nloc x nnei x e_dim
                padding_edge_angle_update = torch.concat(
                    [
                        reduced_edge_angle_update,
                        torch.zeros(
                            [nb, nloc, self.nnei - self.a_sel, self.e_dim],
                            dtype=edge_ebd.dtype,
                            device=edge_ebd.device,
                        ),
                    ],
                    dim=2,
                )
            else:
                # n_angle x e_dim
                weighted_edge_angle_update = edge_angle_update * a_sw.unsqueeze(-1)
                # n_edge x e_dim
                padding_edge_angle_update = aggregate(
                    weighted_edge_angle_update,
                    eij2a_index,
                    average=False,
                    num_owner=n_edge,
                ) / (self.dynamic_a_sel**0.5)

            if not self.smooth_edge_update:
                # will be deprecated in the future
                # not support dynamic index, will pass anyway
                if self.use_dynamic_sel:
                    raise NotImplementedError(
                        "smooth_edge_update must be True when use_dynamic_sel is True!"
                    )
                full_mask = torch.concat(
                    [
                        a_nlist_mask,
                        torch.zeros(
                            [nb, nloc, self.nnei - self.a_sel],
                            dtype=a_nlist_mask.dtype,
                            device=a_nlist_mask.device,
                        ),
                    ],
                    dim=-1,
                )
                padding_edge_angle_update = torch.where(
                    full_mask.unsqueeze(-1), padding_edge_angle_update, edge_ebd
                )
            e_update_list.append(
                self.act(self.edge_angle_linear2(padding_edge_angle_update))
            )
            # update edge_ebd
            e_updated = self.list_update(e_update_list, "edge")
            torch.cuda.nvtx.range_pop()
            
            # angle self message
            # nb x nloc x a_nnei x a_nnei x dim_a
            torch.cuda.nvtx.range_push("angle update")
            if not self.optim_update:
                assert angle_info is not None
                angle_self_update = self.act(self.angle_self_linear(angle_info))
            else:
                angle_self_update = self.act(
                    self.optim_angle_update(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        "angle",
                    )
                    if not self.use_dynamic_sel
                    else self.fused_optim_angle_update_dynamic(
                        angle_ebd, node_ebd_for_angle, edge_ebd_for_angle,
                        n2a_index, eij2a_index, eik2a_index, 
                        feat="angle",
                    )
                )
                """
                    self.optim_angle_update_dynamic(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        n2a_index,
                        eij2a_index,
                        eik2a_index,
                        "angle",
                    )
                """
            a_update_list.append(angle_self_update)
        else:
            # update edge_ebd
            with torch.cuda.nvtx.range("edge update"):
                e_updated = self.list_update(e_update_list, "edge")
            torch.cuda.nvtx.range_push("angle update")
        
        # update angle_ebd
        a_updated = self.list_update(a_update_list, "angle")
        torch.cuda.nvtx.range_pop()

        return e_updated, a_updated

    def forward(
        self,
        node_ebd_ext: torch.Tensor,  # nf x nall x n_dim [OR] nf x nloc x n_dim when not parallel_mode
        edge_ebd: torch.Tensor,  # nf x nloc x nnei x e_dim
        h2: torch.Tensor,  # nf x nloc x nnei x 3
        angle_ebd: torch.Tensor,  # nf x nloc x a_nnei x a_nnei x a_dim
        nlist: torch.Tensor,  # nf x nloc x nnei
        nlist_mask: torch.Tensor,  # nf x nloc x nnei
        sw: torch.Tensor,  # switch func, nf x nloc x nnei
        a_nlist: torch.Tensor,  # nf x nloc x a_nnei
        a_nlist_mask: torch.Tensor,  # nf x nloc x a_nnei
        a_sw: torch.Tensor,  # switch func, nf x nloc x a_nnei
        edge_index: torch.Tensor,  # 2 x n_edge
        angle_index: torch.Tensor,  # 3 x n_angle
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        node_ebd_ext : nf x nall x n_dim
            Extended node embedding.
        edge_ebd : nf x nloc x nnei x e_dim
            Edge embedding.
        h2 : nf x nloc x nnei x 3
            Pair-atom channel, equivariant.
        angle_ebd : nf x nloc x a_nnei x a_nnei x a_dim
            Angle embedding.
        nlist : nf x nloc x nnei
            Neighbor list. (padded neis are set to 0)
        nlist_mask : nf x nloc x nnei
            Masks of the neighbor list. real nei 1 otherwise 0
        sw : nf x nloc x nnei
            Switch function.
        a_nlist : nf x nloc x a_nnei
            Neighbor list for angle. (padded neis are set to 0)
        a_nlist_mask : nf x nloc x a_nnei
            Masks of the neighbor list for angle. real nei 1 otherwise 0
        a_sw : nf x nloc x a_nnei
            Switch function for angle.
        edge_index : Optional for dynamic sel, 2 x n_edge
            n2e_index : n_edge
                Broadcast indices from node(i) to edge(ij), or reduction indices from edge(ij) to node(i).
            n_ext2e_index : n_edge
                Broadcast indices from extended node(j) to edge(ij).
        angle_index : Optional for dynamic sel, 3 x n_angle
            n2a_index : n_angle
                Broadcast indices from extended node(j) to angle(ijk).
            eij2a_index : n_angle
                Broadcast indices from extended edge(ij) to angle(ijk), or reduction indices from angle(ijk) to edge(ij).
            eik2a_index : n_angle
                Broadcast indices from extended edge(ik) to angle(ijk).

        Returns
        -------
        n_updated:     nf x nloc x n_dim
            Updated node embedding.
        e_updated:     nf x nloc x nnei x e_dim
            Updated edge embedding.
        a_updated : nf x nloc x a_nnei x a_nnei x a_dim
            Updated angle embedding.
        """
        nb, nloc, nnei = nlist.shape
        nall = node_ebd_ext.shape[1]
        node_ebd = node_ebd_ext[:, :nloc, :]
        assert (nb, nloc) == node_ebd.shape[:2]
        if not self.use_dynamic_sel:
            assert (nb, nloc, nnei, 3) == h2.shape
            n_edge = None
        else:
            # n_edge = int(nlist_mask.sum().item())
            # assert (n_edge, 3) == h2.shape
            n_edge = h2.shape[0]
        del a_nlist  # may be used in the future

        n2e_index, n_ext2e_index = edge_index[0], edge_index[1]
        n2a_index, eij2a_index, eik2a_index = (
            angle_index[0],
            angle_index[1],
            angle_index[2],
        )
        
        # nb x nloc x nnei x n_dim [OR] n_edge x n_dim
        with torch.cuda.nvtx.range("G1 graph gather"):
            nei_node_ebd = (
                _make_nei_g1(node_ebd_ext, nlist)
                if not self.use_dynamic_sel
                else torch.index_select(
                    node_ebd_ext.reshape(-1, self.n_dim), 0, n_ext2e_index
                )
            )

        n_update_list: list[torch.Tensor] = [node_ebd]
        e_update_list: list[torch.Tensor] = [edge_ebd]
        a_update_list: list[torch.Tensor] = [angle_ebd]

        # node self mlp
        with torch.cuda.nvtx.range("node self update"):
            node_self_mlp = self.act(self.node_self_mlp(node_ebd))
            n_update_list.append(node_self_mlp)

        # node sym (grrg + drrd)
        torch.cuda.nvtx.range_push("geometry symmetrization")
        node_sym_list: list[torch.Tensor] = []
        node_sym_list.append(
            self.symmetrization_op(
                edge_ebd,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
            )
            if not self.use_dynamic_sel
            else self.symmetrization_op_dynamic(
                edge_ebd,
                h2,
                sw,
                owner=n2e_index,
                num_owner=nb * nloc,
                nb=nb,
                nloc=nloc,
                scale_factor=self.dynamic_e_sel ** (-0.5),
                axis_neuron=self.axis_neuron,
            )
        )
        node_sym_list.append(
            self.symmetrization_op(
                nei_node_ebd,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
            )
            if not self.use_dynamic_sel
            else self.symmetrization_op_dynamic(
                nei_node_ebd,
                h2,
                sw,
                owner=n2e_index,
                num_owner=nb * nloc,
                nb=nb,
                nloc=nloc,
                scale_factor=self.dynamic_e_sel ** (-0.5),
                axis_neuron=self.axis_neuron,
            )
        )
        node_sym = self.act(self.node_sym_linear(torch.cat(node_sym_list, dim=-1)))
        n_update_list.append(node_sym)
        torch.cuda.nvtx.range_pop()

        if not self.optim_update:
            if not self.use_dynamic_sel:
                # nb x nloc x nnei x (n_dim * 2 + e_dim)
                edge_info = torch.cat(
                    [
                        torch.tile(node_ebd.unsqueeze(-2), [1, 1, self.nnei, 1]),
                        nei_node_ebd,
                        edge_ebd,
                    ],
                    dim=-1,
                )
            else:
                # n_edge x (n_dim * 2 + e_dim)
                edge_info = torch.cat(
                    [
                        torch.index_select(
                            node_ebd.reshape(-1, self.n_dim), 0, n2e_index
                        ),
                        nei_node_ebd,
                        edge_ebd,
                    ],
                    dim=-1,
                )
        else:
            edge_info = None

        # node edge message
        # nb x nloc x nnei x (h * n_dim)
        torch.cuda.nvtx.range_push("G1 message pass")
        if not self.optim_update:
            assert edge_info is not None
            node_edge_update = self.act(
                self.node_edge_linear(edge_info)
            ) * sw.unsqueeze(-1)
        else:
            node_edge_update = self.act(
                self.optim_edge_update(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    nlist,
                    "node",
                )
                if not self.use_dynamic_sel
                else self.optim_edge_update_dynamic(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    n2e_index,
                    n_ext2e_index,
                    "node",
                )
            ) * sw.unsqueeze(-1)
        node_edge_update = (
            (torch.sum(node_edge_update, dim=-2) / self.nnei)
            if not self.use_dynamic_sel
            else (
                aggregate(
                    node_edge_update,
                    n2e_index,
                    average=False,
                    num_owner=nb * nloc,
                ).reshape(nb, nloc, node_edge_update.shape[-1])
                / self.dynamic_e_sel
            )
        )
        torch.cuda.nvtx.range_pop()

        if self.n_multi_edge_message > 1:
            # nb x nloc x h x n_dim
            node_edge_update_mul_head = node_edge_update.view(
                nb, nloc, self.n_multi_edge_message, self.n_dim
            )
            for head_index in range(self.n_multi_edge_message):
                n_update_list.append(node_edge_update_mul_head[..., head_index, :])
        else:
            n_update_list.append(node_edge_update)
        # update node_ebd
        with torch.cuda.nvtx.range("node update"):
            n_updated = self.list_update(n_update_list, "node")

        # edge self message
        torch.cuda.nvtx.range_push("atom to bond")
        if not self.optim_update:
            assert edge_info is not None
            edge_self_update = self.act(self.edge_self_linear(edge_info))
        else:
            edge_self_update = self.act(
                self.optim_edge_update(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    nlist,
                    "edge",
                )
                if not self.use_dynamic_sel
                else self.optim_edge_update_dynamic(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    n2e_index,
                    n_ext2e_index,
                    "edge",
                )
            )
        e_update_list.append(edge_self_update)
        torch.cuda.nvtx.range_pop()

        if self.update_angle:
            assert self.angle_self_linear is not None
            assert self.edge_angle_linear1 is not None
            assert self.edge_angle_linear2 is not None
            # get angle info
            if self.a_compress_rate != 0:
                if not self.a_compress_use_split:
                    assert self.a_compress_n_linear is not None
                    assert self.a_compress_e_linear is not None
                    node_ebd_for_angle = self.a_compress_n_linear(node_ebd)
                    edge_ebd_for_angle = self.a_compress_e_linear(edge_ebd)
                else:
                    # use the first a_compress_dim dim for node and edge
                    node_ebd_for_angle = node_ebd[..., : self.n_a_compress_dim]
                    edge_ebd_for_angle = edge_ebd[..., : self.e_a_compress_dim]
            else:
                node_ebd_for_angle = node_ebd
                edge_ebd_for_angle = edge_ebd

            if not self.use_dynamic_sel:
                # nb x nloc x a_nnei x e_dim
                edge_ebd_for_angle = edge_ebd_for_angle[..., : self.a_sel, :]
                # nb x nloc x a_nnei x e_dim
                edge_ebd_for_angle = torch.where(
                    a_nlist_mask.unsqueeze(-1), edge_ebd_for_angle, 0.0
                )
            if not self.optim_update:
                # nb x nloc x a_nnei x a_nnei x n_dim [OR] n_angle x n_dim
                node_for_angle_info = (
                    torch.tile(
                        node_ebd_for_angle.unsqueeze(2).unsqueeze(2),
                        (1, 1, self.a_sel, self.a_sel, 1),
                    )
                    if not self.use_dynamic_sel
                    else torch.index_select(
                        node_ebd_for_angle.reshape(-1, self.n_a_compress_dim),
                        0,
                        n2a_index,
                    )
                )

                # nb x nloc x (a_nnei) x a_nnei x e_dim [OR] n_angle x e_dim
                edge_for_angle_k = (
                    torch.tile(
                        edge_ebd_for_angle.unsqueeze(2), (1, 1, self.a_sel, 1, 1)
                    )
                    if not self.use_dynamic_sel
                    else torch.index_select(edge_ebd_for_angle, 0, eik2a_index)
                )
                # nb x nloc x a_nnei x (a_nnei) x e_dim [OR] n_angle x e_dim
                edge_for_angle_j = (
                    torch.tile(
                        edge_ebd_for_angle.unsqueeze(3), (1, 1, 1, self.a_sel, 1)
                    )
                    if not self.use_dynamic_sel
                    else torch.index_select(edge_ebd_for_angle, 0, eij2a_index)
                )
                # nb x nloc x a_nnei x a_nnei x (e_dim + e_dim) [OR] n_angle x (e_dim + e_dim)
                edge_for_angle_info = torch.cat(
                    [edge_for_angle_k, edge_for_angle_j], dim=-1
                )
                angle_info_list = [angle_ebd]
                angle_info_list.append(node_for_angle_info)
                angle_info_list.append(edge_for_angle_info)
                # nb x nloc x a_nnei x a_nnei x (a + n_dim + e_dim*2) or (a + a/c + a/c)
                # [OR]
                # n_angle x (a + n_dim + e_dim*2) or (a + a/c + a/c)
                angle_info = torch.cat(angle_info_list, dim=-1)
            else:
                angle_info = None

            # edge angle message
            # nb x nloc x a_nnei x a_nnei x e_dim [OR] n_angle x e_dim
            torch.cuda.nvtx.range_push("G2 message pass")
            if not self.optim_update:
                assert angle_info is not None
                edge_angle_update = self.act(self.edge_angle_linear1(angle_info))
            else:
                edge_angle_update = self.act(
                    self.optim_angle_update(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        "edge",
                    )
                    if not self.use_dynamic_sel
                    else self.optim_angle_update_dynamic(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        n2a_index,
                        eij2a_index,
                        eik2a_index,
                        "edge",
                    )
                )
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("edge update")
            if not self.use_dynamic_sel:
                # nb x nloc x a_nnei x a_nnei x e_dim
                weighted_edge_angle_update = (
                    a_sw.unsqueeze(-1).unsqueeze(-1)
                    * a_sw.unsqueeze(-2).unsqueeze(-1)
                    * edge_angle_update
                )
                # nb x nloc x a_nnei x e_dim
                reduced_edge_angle_update = torch.sum(
                    weighted_edge_angle_update, dim=-2
                ) / (self.a_sel**0.5)
                # nb x nloc x nnei x e_dim
                padding_edge_angle_update = torch.concat(
                    [
                        reduced_edge_angle_update,
                        torch.zeros(
                            [nb, nloc, self.nnei - self.a_sel, self.e_dim],
                            dtype=edge_ebd.dtype,
                            device=edge_ebd.device,
                        ),
                    ],
                    dim=2,
                )
            else:
                # n_angle x e_dim
                weighted_edge_angle_update = edge_angle_update * a_sw.unsqueeze(-1)
                # n_edge x e_dim
                padding_edge_angle_update = aggregate(
                    weighted_edge_angle_update,
                    eij2a_index,
                    average=False,
                    num_owner=n_edge,
                ) / (self.dynamic_a_sel**0.5)
            if not self.smooth_edge_update:
                # will be deprecated in the future
                # not support dynamic index, will pass anyway
                if self.use_dynamic_sel:
                    raise NotImplementedError(
                        "smooth_edge_update must be True when use_dynamic_sel is True!"
                    )
                full_mask = torch.concat(
                    [
                        a_nlist_mask,
                        torch.zeros(
                            [nb, nloc, self.nnei - self.a_sel],
                            dtype=a_nlist_mask.dtype,
                            device=a_nlist_mask.device,
                        ),
                    ],
                    dim=-1,
                )
                padding_edge_angle_update = torch.where(
                    full_mask.unsqueeze(-1), padding_edge_angle_update, edge_ebd
                )
            e_update_list.append(
                self.act(self.edge_angle_linear2(padding_edge_angle_update))
            )
            # update edge_ebd
            e_updated = self.list_update(e_update_list, "edge")
            torch.cuda.nvtx.range_pop()

            # angle self message
            # nb x nloc x a_nnei x a_nnei x dim_a
            torch.cuda.nvtx.range_push("angle update")
            if not self.optim_update:
                assert angle_info is not None
                angle_self_update = self.act(self.angle_self_linear(angle_info))
            else:
                angle_self_update = self.act(
                    self.optim_angle_update(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        "angle",
                    )
                    if not self.use_dynamic_sel
                    else self.optim_angle_update_dynamic(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        n2a_index,
                        eij2a_index,
                        eik2a_index,
                        "angle",
                    )
                )
            a_update_list.append(angle_self_update)
        else:
            # update edge_ebd
            torch.cuda.nvtx.range_push("angle update")
            e_updated = self.list_update(e_update_list, "edge")
        
        # update angle_ebd
        a_updated = self.list_update(a_update_list, "angle")
        torch.cuda.nvtx.range_pop()
        return n_updated, e_updated, a_updated

    @torch.jit.export
    def list_update_res_avg(
        self,
        update_list: list[torch.Tensor],
    ) -> torch.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        for ii in range(1, nitem):
            uu = uu + update_list[ii]
        return uu / (float(nitem) ** 0.5)

    @torch.jit.export
    def list_update_res_incr(self, update_list: list[torch.Tensor]) -> torch.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        scale = 1.0 / (float(nitem - 1) ** 0.5) if nitem > 1 else 0.0
        for ii in range(1, nitem):
            uu = uu + scale * update_list[ii]
        return uu

    @torch.jit.export
    def list_update_res_residual(
        self, update_list: list[torch.Tensor], update_name: str = "node"
    ) -> torch.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        # make jit happy
        if update_name == "node":
            for ii, vv in enumerate(self.n_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "edge":
            for ii, vv in enumerate(self.e_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "angle":
            for ii, vv in enumerate(self.a_residual):
                uu = uu + vv * update_list[ii + 1]
        else:
            raise NotImplementedError
        return uu

    @torch.jit.export
    def fuse_list_update_res_residual(
        self, update_list: list[torch.Tensor], update_name: str = "node"
    ) -> torch.Tensor:
        nitem = len(update_list)
        if nitem == 1:
            return update_list[0]

        uu = update_list[0]

        if update_name == "node":
            residuals = self.n_residual
        elif update_name == "edge":
            residuals = self.e_residual
        elif update_name == "angle":
            residuals = self.a_residual
        else:
            raise NotImplementedError

        # 使用 addcmul 融合算子：计算 uu = uu + update_list * vv
        # 完全避免了 vv * update_list 带来的中间张量显存开销，每个 item 仅 1 个 Kernel
        for ii, vv in enumerate(residuals):
            if ii + 1 < nitem:
                uu = torch.addcmul(uu, update_list[ii + 1], vv)

        return uu

    @torch.jit.export
    def list_update(
        self, update_list: list[torch.Tensor], update_name: str = "node"
    ) -> torch.Tensor:
        if self.update_style == "res_avg":
            return self.list_update_res_avg(update_list)
        elif self.update_style == "res_incr":
            return self.list_update_res_incr(update_list)
        elif self.update_style == "res_residual":
            # return self.fuse_list_update_res_residual(update_list, update_name=update_name)
            return self.list_update_res_residual(update_list, update_name=update_name)
        else:
            raise RuntimeError(f"unknown update style {self.update_style}")

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        data = {
            "@class": "RepFlowLayer",
            "@version": 2,
            "e_rcut": self.e_rcut,
            "e_rcut_smth": self.e_rcut_smth,
            "e_sel": self.e_sel,
            "a_rcut": self.a_rcut,
            "a_rcut_smth": self.a_rcut_smth,
            "a_sel": self.a_sel,
            "ntypes": self.ntypes,
            "n_dim": self.n_dim,
            "e_dim": self.e_dim,
            "a_dim": self.a_dim,
            "a_compress_rate": self.a_compress_rate,
            "a_compress_e_rate": self.a_compress_e_rate,
            "a_compress_use_split": self.a_compress_use_split,
            "n_multi_edge_message": self.n_multi_edge_message,
            "axis_neuron": self.axis_neuron,
            "activation_function": self.activation_function,
            "update_angle": self.update_angle,
            "update_style": self.update_style,
            "update_residual": self.update_residual,
            "update_residual_init": self.update_residual_init,
            "precision": self.precision,
            "optim_update": self.optim_update,
            "smooth_edge_update": self.smooth_edge_update,
            "use_dynamic_sel": self.use_dynamic_sel,
            "sel_reduce_factor": self.sel_reduce_factor,
            "node_self_mlp": self.node_self_mlp.serialize(),
            "node_sym_linear": self.node_sym_linear.serialize(),
            "node_edge_linear": self.node_edge_linear.serialize(),
            "edge_self_linear": self.edge_self_linear.serialize(),
        }
        if self.update_angle:
            data.update(
                {
                    "edge_angle_linear1": self.edge_angle_linear1.serialize(),
                    "edge_angle_linear2": self.edge_angle_linear2.serialize(),
                    "angle_self_linear": self.angle_self_linear.serialize(),
                }
            )
            if self.a_compress_rate != 0 and not self.a_compress_use_split:
                data.update(
                    {
                        "a_compress_n_linear": self.a_compress_n_linear.serialize(),
                        "a_compress_e_linear": self.a_compress_e_linear.serialize(),
                    }
                )
        if self.update_style == "res_residual":
            data.update(
                {
                    "@variables": {
                        "n_residual": [to_numpy_array(t) for t in self.n_residual],
                        "e_residual": [to_numpy_array(t) for t in self.e_residual],
                        "a_residual": [to_numpy_array(t) for t in self.a_residual],
                    }
                }
            )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "RepFlowLayer":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 2, 1)
        data.pop("@class")
        update_angle = data["update_angle"]
        a_compress_rate = data["a_compress_rate"]
        a_compress_use_split = data["a_compress_use_split"]
        node_self_mlp = data.pop("node_self_mlp")
        node_sym_linear = data.pop("node_sym_linear")
        node_edge_linear = data.pop("node_edge_linear")
        edge_self_linear = data.pop("edge_self_linear")
        edge_angle_linear1 = data.pop("edge_angle_linear1", None)
        edge_angle_linear2 = data.pop("edge_angle_linear2", None)
        angle_self_linear = data.pop("angle_self_linear", None)
        a_compress_n_linear = data.pop("a_compress_n_linear", None)
        a_compress_e_linear = data.pop("a_compress_e_linear", None)
        update_style = data["update_style"]
        variables = data.pop("@variables", {})
        n_residual = variables.get("n_residual", data.pop("n_residual", []))
        e_residual = variables.get("e_residual", data.pop("e_residual", []))
        a_residual = variables.get("a_residual", data.pop("a_residual", []))

        obj = cls(**data)
        obj.node_self_mlp = MLPLayer.deserialize(node_self_mlp)
        obj.node_sym_linear = MLPLayer.deserialize(node_sym_linear)
        obj.node_edge_linear = MLPLayer.deserialize(node_edge_linear)
        obj.edge_self_linear = MLPLayer.deserialize(edge_self_linear)

        if update_angle:
            assert isinstance(edge_angle_linear1, dict)
            assert isinstance(edge_angle_linear2, dict)
            assert isinstance(angle_self_linear, dict)
            obj.edge_angle_linear1 = MLPLayer.deserialize(edge_angle_linear1)
            obj.edge_angle_linear2 = MLPLayer.deserialize(edge_angle_linear2)
            obj.angle_self_linear = MLPLayer.deserialize(angle_self_linear)
            if a_compress_rate != 0 and not a_compress_use_split:
                assert isinstance(a_compress_n_linear, dict)
                assert isinstance(a_compress_e_linear, dict)
                obj.a_compress_n_linear = MLPLayer.deserialize(a_compress_n_linear)
                obj.a_compress_e_linear = MLPLayer.deserialize(a_compress_e_linear)

        if update_style == "res_residual":
            for ii, t in enumerate(obj.n_residual):
                t.data = to_torch_tensor(n_residual[ii])
            for ii, t in enumerate(obj.e_residual):
                t.data = to_torch_tensor(e_residual[ii])
            for ii, t in enumerate(obj.a_residual):
                t.data = to_torch_tensor(a_residual[ii])
        return obj

