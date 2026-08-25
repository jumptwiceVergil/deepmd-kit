import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def fused_cal_hg_dynamic_forward(
    M,  # n_edge
    N,  # 3 * e_dim
    E,  # e_dim
    NO, # num_owner
    EDGES_PER_OWNER = 11,
    dtype = "float32",
    accum_dtype = "float32",
    BLOCK_N: int = 64,
):
    @T.prim_func
    def hg_kernel(
        flat_edge_ebd: T.Buffer((M, E), dtype),
        flat_sw: T.Buffer((M,), dtype),
        flat_h2: T.Buffer((M, 3), dtype),
        scale_factor: T.float32,
        out: T.Buffer((NO, N), accum_dtype),
    ):
        with T.Kernel(NO, T.ceildiv(N, BLOCK_N), threads=64) as (bx, by):
            meta_shared = T.alloc_shared((EDGES_PER_OWNER, 4), dtype)

            # 加载 sw
            for r in T.Parallel(EDGES_PER_OWNER):
                edge_idx = bx * EDGES_PER_OWNER + r
                meta_shared[r, 0] = flat_sw[edge_idx]

            # 加载 h2
            for r, k in T.Parallel(EDGES_PER_OWNER, 3):
                edge_idx = bx * EDGES_PER_OWNER + r
                meta_shared[r, k + 1] = flat_h2[edge_idx, k]
            
            acc = T.alloc_fragment((BLOCK_N,), accum_dtype)

            T.clear(acc)

            for j in T.Parallel(BLOCK_N):
                col = by * BLOCK_N + j

                if col < N:
                    h2_idx = col // E
                    e_idx = col % E
                    v0 = flat_edge_ebd[bx * EDGES_PER_OWNER + 0, e_idx] * meta_shared[0, 0] * meta_shared[0, h2_idx + 1]
                    v1 = flat_edge_ebd[bx * EDGES_PER_OWNER + 1, e_idx] * meta_shared[1, 0] * meta_shared[1, h2_idx + 1]
                    v2 = flat_edge_ebd[bx * EDGES_PER_OWNER + 2, e_idx] * meta_shared[2, 0] * meta_shared[2, h2_idx + 1]
                    v3 = flat_edge_ebd[bx * EDGES_PER_OWNER + 3, e_idx] * meta_shared[3, 0] * meta_shared[3, h2_idx + 1]
                    v4 = flat_edge_ebd[bx * EDGES_PER_OWNER + 4, e_idx] * meta_shared[4, 0] * meta_shared[4, h2_idx + 1]
                    v5 = flat_edge_ebd[bx * EDGES_PER_OWNER + 5, e_idx] * meta_shared[5, 0] * meta_shared[5, h2_idx + 1]
                    v6 = flat_edge_ebd[bx * EDGES_PER_OWNER + 6, e_idx] * meta_shared[6, 0] * meta_shared[6, h2_idx + 1]
                    v7 = flat_edge_ebd[bx * EDGES_PER_OWNER + 7, e_idx] * meta_shared[7, 0] * meta_shared[7, h2_idx + 1]
                    v8 = flat_edge_ebd[bx * EDGES_PER_OWNER + 8, e_idx] * meta_shared[8, 0] * meta_shared[8, h2_idx + 1]
                    v9 = flat_edge_ebd[bx * EDGES_PER_OWNER + 9, e_idx] * meta_shared[9, 0] * meta_shared[9, h2_idx + 1]
                    v10 = flat_edge_ebd[bx * EDGES_PER_OWNER + 10, e_idx] * meta_shared[10, 0] * meta_shared[10, h2_idx + 1]
                    
                    acc[j] = (v0 + v1 + v2 + v3 + v4
                        + v5 + v6 + v7 + v8 + v9 + v10
                    ) * scale_factor
            
            for j in T.Parallel(BLOCK_N):
                col = by * BLOCK_N + j
                if col < N:
                    out[bx, col] = acc[j]

    return hg_kernel

@tilelang.jit
def fused_call_grrg_forward(
    NB,     # nb
    NLOC,   # nloc
    E,      # e_dim
    AXIS,   # axis_neuron
    dtype="float32",
    accum_dtype="float32",
    BLOCK_M=4,
    BLOCK_N=32,
):
    @T.prim_func
    def grrg_kernel(
        h2g2: T.Buffer((NB, NLOC, 3, E), dtype),
        out: T.Buffer((NB, NLOC, AXIS * E), accum_dtype),
    ):
        NUM_TILE_M = T.ceildiv(AXIS, BLOCK_M)
        NUM_TILE_N = T.ceildiv(E, BLOCK_N)
        with T.Kernel(NB, NLOC, NUM_TILE_M * NUM_TILE_N, threads=128) as (bx, by, bz):
            tile_m = bz // NUM_TILE_N
            tile_n = bz % NUM_TILE_N

            for a, e in T.Parallel(BLOCK_M, BLOCK_N):
                axis_idx = tile_m * BLOCK_M + a
                e_idx = tile_n * BLOCK_N + e
                if axis_idx < AXIS and e_idx < E:
                    v0 = h2g2[bx, by, 0, axis_idx] * h2g2[bx, by, 0, e_idx]
                    v1 = h2g2[bx, by, 1, axis_idx] * h2g2[bx, by, 1, e_idx]
                    v2 = h2g2[bx, by, 2, axis_idx] * h2g2[bx, by, 2, e_idx]

                    acc = (v0 + v1 + v2) / (3.0**1)
                    out[bx, by, axis_idx * E + e_idx] = acc

    return grrg_kernel

class FusedSymmetrizationOpDynamic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        n_edge, e_dim = flat_edge_ebd.shape
        # # n_edge x e_dim
        # flat_edge_ebd = flat_edge_ebd * flat_sw.unsqueeze(-1)
        # # n_edge x 3 x e_dim
        # flat_h2g2 = (flat_h2.unsqueeze(-1) * flat_edge_ebd.unsqueeze(-2)).reshape(
        #     -1, 3 * e_dim
        # )
        # # nb x nloc x 3 x e_dim
        # h2g2 = (
        #     aggregate(flat_h2g2, owner, average=False, num_owner=num_owner).reshape(
        #         nb, nloc, 3, e_dim
        #     )
        #     * scale_factor
        # )

        h2g2 = torch.empty(
            (num_owner, 3*e_dim),
            device=flat_edge_ebd.device,
            dtype=flat_edge_ebd.dtype,
        )

        hg_kernel = fused_cal_hg_dynamic_forward(M=n_edge, N=3*e_dim, E=e_dim, NO=num_owner)
        hg_kernel(flat_edge_ebd, flat_sw, flat_h2, scale_factor, h2g2)
        h2g2 = h2g2.reshape(nb, nloc, 3, e_dim)

        # # nb x nloc x 3 x e_dim
        # nb, nloc, _, e_dim = h2g2.shape
        # # nb x nloc x 3 x axis
        # h2g2m = h2g2[..., :axis_neuron]
        # # nb x nloc x axis x e_dim
        # g1_13 = torch.matmul(torch.transpose(h2g2m, -1, -2), h2g2) / (3.0**1)
        # # nb x nloc x (axis x e_dim)
        # grrg = g1_13.view(nb, nloc, axis_neuron * e_dim)
        
        grrg = torch.empty(
            (nb, nloc, axis_neuron * e_dim),
            device=h2g2.device,
            dtype=h2g2.dtype,
        )

        grrg_kernel = fused_call_grrg_forward(NB=nb, NLOC=nloc, E=e_dim, AXIS=axis_neuron)
        grrg_kernel(h2g2, grrg)

        ctx.save_for_backward(
            flat_edge_ebd,
            flat_h2,
            flat_sw,
            owner,
            h2g2,
        )
        
        ctx.nb = nb
        ctx.nloc = nloc
        ctx.num_owner = num_owner
        ctx.scale_factor = scale_factor
        ctx.axis_neuron = axis_neuron

        return grrg
    
    @staticmethod
    def backward(
        ctx,
        grad_grrg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, float, int]:  
        # [TODO] reduce the ctx saved tensor
        (
            flat_edge_ebd,
            flat_h2,
            flat_sw,
            owner,
            h2g2,
        ) = ctx.saved_tensors

        nb = ctx.nb
        nloc = ctx.nloc
        num_owner = ctx.num_owner
        scale_factor = ctx.scale_factor
        axis_neuron = ctx.axis_neuron 
        assert nb == 1, "only support nb=1."

        grad_flat_edge_ebd, grad_h2, grad_flat_sw = FusedSymmetrizationOpDynamicBackward.apply(
                grad_grrg, flat_edge_ebd, flat_h2, flat_sw,
                owner, h2g2, nb, nloc, num_owner, scale_factor, axis_neuron
        )
        
        return (
            grad_flat_edge_ebd,   # 0
            grad_h2,              # 1
            grad_flat_sw,         # 2
            None,                 # 3 owner
            None,                 # 4 num_owner
            None,                 # 5 nb
            None,                 # 6 nloc
            None,                 # 7 scale_factor
            None,                 # 8 axis_neuron
        )

class FusedSymmetrizationOpDynamicBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_grrg: torch.Tensor,
        flat_edge_ebd: torch.Tensor,
        flat_h2: torch.Tensor,
        flat_sw: torch.Tensor,
        owner: torch.Tensor, 
        h2g2: torch.Tensor,
        nb: int,
        nloc: int,
        num_owner: int,
        scale_factor: float,
        axis_neuron: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M, E = flat_edge_ebd.shape

        grad_g1 = grad_grrg.reshape(nb, nloc, axis_neuron, E)
        h2g2m = h2g2[..., :axis_neuron]

        grad_left = torch.matmul(grad_g1, h2g2.transpose(-1, -2))
        grad_left = grad_left.transpose(-1, -2)

        grad_right = torch.matmul(h2g2m, grad_g1)

        grad_h2g2 = grad_right.clone()
        grad_h2g2[..., :axis_neuron] += grad_left
        grad_h2g2 /= 3.0
        grad_h2g2 *= scale_factor
        owner = owner.long()
        grad_flat_h2g2 = grad_h2g2[0, owner, :, :]

        edge_scaled = flat_edge_ebd * flat_sw.unsqueeze(-1)
        grad_h2 = (grad_flat_h2g2 * edge_scaled.unsqueeze(1)).sum(dim=-1)

        grad_edge_scaled = (grad_flat_h2g2 * flat_h2.unsqueeze(-1)).sum(dim=1)
        grad_flat_edge_ebd = grad_edge_scaled * flat_sw.unsqueeze(-1)

        grad_flat_sw = (grad_edge_scaled * flat_edge_ebd).sum(dim=-1)

        ctx.save_for_backward(
            grad_grrg,
            flat_edge_ebd,
            flat_h2,
            flat_sw,
            owner,
            h2g2,
            grad_flat_h2g2,
        )

        ctx.nb = nb
        ctx.nloc = nloc
        ctx.num_owner = num_owner
        ctx.scale_factor = scale_factor
        ctx.axis_neuron = axis_neuron

        return (
            grad_flat_edge_ebd,   # 0
            grad_h2,              # 1
            grad_flat_sw,         # 2
        )

    @staticmethod
    def backward(
        ctx,
        grad_grad_edge: torch.Tensor,
        grad_grad_h2: torch.Tensor,
        grad_grad_sw: torch.Tensor,
    ):
        (
            grad_grrg,
            flat_edge_ebd,
            flat_h2,
            flat_sw,
            owner,
            h2g2,
            grad_flat_h2g2,
        ) = ctx.saved_tensors

        nb = ctx.nb
        nloc = ctx.nloc
        num_owner = ctx.num_owner
        scale_factor = ctx.scale_factor
        axis_neuron = ctx.axis_neuron

        M, E = flat_edge_ebd.shape
        A = axis_neuron
        G = grad_grrg.reshape(nb, nloc, A, E)
        H = h2g2
        B = H.shape[-2]
        Hm = H[..., :A]
        c = scale_factor / 3.0

        R = grad_flat_h2g2
        grad_R_from_h2 = (
            grad_grad_h2.unsqueeze(-1)
            * flat_edge_ebd.unsqueeze(1)
            * flat_sw.unsqueeze(-1).unsqueeze(1)
        )
        grad_R_from_edge = (
            grad_grad_edge.unsqueeze(1)
            * flat_h2.unsqueeze(-1)
            * flat_sw.unsqueeze(-1).unsqueeze(1)
        )
        grad_R_from_sw = (
            grad_grad_sw.unsqueeze(-1).unsqueeze(-1)
            * flat_h2.unsqueeze(-1)
            * flat_edge_ebd.unsqueeze(1)
        )
        grad_R = grad_R_from_h2 + grad_R_from_edge + grad_R_from_sw

        grad_Q = torch.zeros(nb, nloc, B, E, dtype=grad_R.dtype, device=grad_R.device)
        grad_Q[0].index_add_(0, owner.long(), grad_R)
        grad_G_from_q1 = torch.matmul(Hm.transpose(-1, -2), grad_Q)
        grad_Q_q2 = grad_Q[..., :A]
        grad_G_from_q2 = torch.matmul(grad_Q_q2.transpose(-1, -2), H)
        grad_G = c * (grad_G_from_q1 + grad_G_from_q2)

        grad_H = torch.zeros_like(H)
        grad_H_from_q1 = torch.matmul(grad_Q, G.transpose(-1, -2))
        grad_H[..., :A] += c * grad_H_from_q1
        grad_H_from_q2 = torch.matmul(grad_Q_q2, G)
        grad_H += c * grad_H_from_q2

        grad_edge_from_h2 = (grad_grad_h2.unsqueeze(-1) * R).sum(dim=1)
        grad_edge_from_h2 *= flat_sw.unsqueeze(-1)

        grad_edge_scaled = (R * flat_h2.unsqueeze(-1)).sum(dim=1)
        grad_edge_from_edge = grad_grad_edge * flat_sw.unsqueeze(-1) * grad_edge_scaled
        grad_edge_from_sw = grad_grad_sw.unsqueeze(-1) * grad_edge_scaled
        grad_flat_edge_ebd = grad_edge_from_h2 + grad_edge_from_edge + grad_edge_from_sw

        grad_h2_from_edge = (
            grad_grad_edge.unsqueeze(1)
            * flat_sw.unsqueeze(-1).unsqueeze(1)
            * R
        ).sum(dim=-1)
        grad_h2_from_sw = (
            grad_grad_sw.unsqueeze(-1).unsqueeze(-1)
            * R
            * flat_edge_ebd.unsqueeze(1)
        ).sum(dim=-1)
        grad_flat_h2 = grad_h2_from_edge + grad_h2_from_sw

        grad_sw_from_h2 = (
            grad_grad_h2 * (R * flat_edge_ebd.unsqueeze(1)).sum(dim=-1)
        ).sum(dim=-1)
        grad_sw_from_edge = (grad_grad_edge * grad_edge_scaled).sum(dim=-1)
        grad_flat_sw = grad_sw_from_h2 + grad_sw_from_edge

        grad_grad_grrg = grad_G.reshape_as(grad_grrg)

        return (
            grad_grad_grrg,       # 0 grad_grrg
            grad_flat_edge_ebd,   # 1 flat_edge_ebd
            grad_flat_h2,         # 2 flat_h2
            grad_flat_sw,         # 3 flat_sw
            None,                 # 4 owner
            grad_H,               # 5 h2g2
            None,                 # 6 nb
            None,                 # 7 nloc
            None,                 # 8 num_owner
            None,                 # 9 scale_factor
            None,                 # 10 axis_neuron
        )

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
        node_ebd: torch.Tensor,
        node_ebd_ext: torch.Tensor,
        flat_edge_ebd: torch.Tensor,
        n2e_index: torch.Tensor,
        n_ext2e_index: torch.Tensor,
        node: torch.Tensor,
        node_ext: torch.Tensor,
        edge: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:

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
    def backward(
        ctx,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, \
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        output = FusedEdgeUpdateFunctionBackward.apply(grad_out, node_ebd, node_ebd_ext, flat_edge_ebd,
                n2e_index, n_ext2e_index, node_weight, node_ext_weight, edge_weight)

        grad_node, grad_node_ext, grad_edge_ebd, _, _, grad_node_weight, \
            grad_node_ext_weight, grad_edge_weight, grad_bias = output
        
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

class FusedEdgeUpdateFunctionBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_out: torch.Tensor,
        node_ebd: torch.Tensor,
        node_ebd_ext: torch.Tensor,
        flat_edge_ebd: torch.Tensor,
        n2e_index: torch.Tensor,
        n_ext2e_index: torch.Tensor,
        node_weight: torch.Tensor,
        node_ext_weight: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, \
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # bias
        grad_bias = grad_out.sum(dim=0)

        ctx.save_for_backward(
            grad_out,
            node_ebd,
            node_ebd_ext,
            flat_edge_ebd,
            n2e_index,
            n_ext2e_index,
            node_weight,
            node_ext_weight,
            edge_weight,
        )

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

    @staticmethod
    def backward(
        ctx,
        grad_grad_node: torch.Tensor,
        grad_grad_node_ext: torch.Tensor,
        grad_grad_edge_ebd: torch.Tensor,
        grad_grad_n2e_index: torch.Tensor,
        grad_grad_n_ext2e_index: torch.Tensor,
        grad_grad_node_weight: torch.Tensor,
        grad_grad_node_ext_weight: torch.Tensor,
        grad_grad_edge_weight: torch.Tensor,
        grad_grad_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, \
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            grad_out,
            node_ebd,
            node_ebd_ext,
            flat_edge_ebd,
            n2e_index,
            n_ext2e_index,
            node_weight,
            node_ext_weight,
            edge_weight,
        ) = ctx.saved_tensors

        E, K = grad_out.shape
        N, D = node_ebd.shape

        gathered_node = node_ebd[n2e_index]
        gathered_node_ext = node_ebd_ext[n_ext2e_index]
        grad_grad_out = torch.zeros_like(grad_out)

        if grad_grad_node is not None:
            grad_gathered_node = grad_grad_node[n2e_index]
            grad_grad_out = grad_grad_out + grad_gathered_node @ node_weight
        
        if grad_grad_node_weight is not None:
            grad_grad_out = grad_grad_out + gathered_node @ grad_grad_node_weight
        
        if grad_grad_node_ext is not None:
            grad_gathered_node_ext = grad_grad_node_ext[n_ext2e_index]
            grad_grad_out = grad_grad_out + grad_gathered_node_ext @ node_ext_weight
        
        if grad_grad_node_ext_weight is not None:
            grad_grad_out = grad_grad_out + gathered_node_ext @ grad_grad_node_ext_weight

        if grad_grad_edge_ebd is not None:
            grad_grad_out = grad_grad_out + grad_grad_edge_ebd @ edge_weight

        if grad_grad_edge_weight is not None:
            grad_grad_out = grad_grad_out + flat_edge_ebd @ grad_grad_edge_weight

        if grad_grad_bias is not None:
            grad_grad_out = grad_grad_out + grad_grad_bias.unsqueeze(0)
        
        grad_node_ebd = torch.zeros_like(node_ebd)
        if grad_grad_node_weight is not None:
            grad_gathered_node_from_weight = grad_out @ grad_grad_node_weight.T
            grad_node_ebd.index_add_(0, n2e_index, grad_gathered_node_from_weight)

        grad_node_ebd_ext = torch.zeros_like(node_ebd_ext)
        if grad_grad_node_ext_weight is not None:
            grad_gathered_node_ext_from_weight = grad_out @ grad_grad_node_ext_weight.T
            grad_node_ebd_ext.index_add_(0, n_ext2e_index, grad_gathered_node_ext_from_weight)

        grad_flat_edge_ebd = torch.zeros_like(flat_edge_ebd)
        if grad_grad_edge_weight is not None:
            grad_flat_edge_ebd = grad_out @ grad_grad_edge_weight.T

        grad_node_weight = torch.zeros_like(node_weight)
        if grad_grad_node is not None:
            grad_gathered_node = grad_grad_node[n2e_index]
            grad_node_weight = grad_gathered_node.T @ grad_out

        grad_node_ext_weight = torch.zeros_like(node_ext_weight)

        if grad_grad_node_ext is not None:
            grad_gathered_node_ext = grad_grad_node_ext[n_ext2e_index]
            grad_node_ext_weight = grad_gathered_node_ext.T @ grad_out

        grad_edge_weight = torch.zeros_like(edge_weight)
        if grad_grad_edge_ebd is not None:
            grad_edge_weight = grad_grad_edge_ebd.T @ grad_out

        return (
            grad_grad_out,           # 0 grad_out
            grad_node_ebd,           # 1 node_ebd
            grad_node_ebd_ext,       # 2 node_ebd_ext
            grad_flat_edge_ebd,      # 3 flat_edge_ebd
            None,                    # 4 n2e_index
            None,                    # 5 n_ext2e_index
            grad_node_weight,        # 6 node_weight
            grad_node_ext_weight,    # 7 node_ext_weight
            grad_edge_weight,        # 8 edge_weight
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
        flat_angle_ebd: torch.Tensor,
        node_ebd: torch.Tensor,
        flat_edge_ebd: torch.Tensor,
        n2a_index: torch.Tensor,
        eij2a_index: torch.Tensor,
        eik2a_index: torch.Tensor,
        sub_angle: torch.Tensor,
        sub_node: torch.Tensor,
        sub_edge_ik: torch.Tensor,
        sub_edge_ij: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
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
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, \
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        node_ebd_shape = ctx.node_ebd_shape

        output = FusedAngleUpdateFunctionBackward.apply(grad_output, flat_angle_ebd, flat_node_ebd,
            flat_edge_ebd, n2a_index, eij2a_index, eik2a_index,
            sub_angle, sub_node, sub_edge_ik, sub_edge_ij, node_ebd_shape)
        
        (
            grad_flat_angle_ebd,
            grad_node_ebd,
            grad_flat_edge_ebd,
            grad_sub_angle,
            grad_sub_node,
            grad_sub_edge_ik,
            grad_sub_edge_ij,
            grad_bias,
        ) = output

        
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

class FusedAngleUpdateFunctionBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_output: torch.Tensor,
        flat_angle_ebd: torch.Tensor,
        flat_node_ebd: torch.Tensor,
        flat_edge_ebd: torch.Tensor,
        n2a_index: torch.Tensor,
        eij2a_index: torch.Tensor,
        eik2a_index: torch.Tensor,
        sub_angle: torch.Tensor,
        sub_node: torch.Tensor,
        sub_edge_ik: torch.Tensor,
        sub_edge_ij: torch.Tensor,
        node_ebd_shape,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, \
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_output = grad_output.contiguous()

        grad_flat_angle_ebd = torch.matmul(grad_output, sub_angle.transpose(0, 1))
        grad_gathered_node = torch.matmul(grad_output, sub_node.transpose(0, 1))
        
        grad_flat_node_ebd = torch.zeros(flat_node_ebd.shape, device=grad_output.device, dtype=grad_output.dtype)
        grad_flat_node_ebd.index_add_(0, n2a_index, grad_gathered_node)
        grad_node_ebd = grad_flat_node_ebd.reshape(node_ebd_shape)
        
        grad_gathered_edge_ik = torch.matmul(grad_output, sub_edge_ik.transpose(0, 1))
        grad_gathered_edge_ij = torch.matmul(grad_output, sub_edge_ij.transpose(0, 1))
        grad_flat_edge_ebd = torch.zeros(flat_edge_ebd.shape, device=grad_output.device, dtype=grad_output.dtype)
        grad_flat_edge_ebd.index_add_(0, eik2a_index, grad_gathered_edge_ik)
        grad_flat_edge_ebd.index_add_(0, eij2a_index, grad_gathered_edge_ij)
        grad_sub_angle = torch.matmul(flat_angle_ebd.transpose(0, 1), grad_output)

        gathered_node = torch.index_select(flat_node_ebd, 0, n2a_index)
        grad_sub_node = torch.matmul(gathered_node.transpose(0, 1), grad_output)

        gathered_edge_ik = torch.index_select(flat_edge_ebd, 0, eik2a_index)
        grad_sub_edge_ik = torch.matmul(gathered_edge_ik.transpose(0, 1), grad_output)
        gathered_edge_ij = torch.index_select(flat_edge_ebd, 0, eij2a_index)
        grad_sub_edge_ij = torch.matmul(gathered_edge_ij.transpose(0, 1), grad_output)

        grad_bias = grad_output.sum(dim=0)

        ctx.save_for_backward(
            grad_output,
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
        
        return (
            grad_flat_angle_ebd,
            grad_node_ebd,
            grad_flat_edge_ebd,
            grad_sub_angle,
            grad_sub_node,
            grad_sub_edge_ik,
            grad_sub_edge_ij,
            grad_bias,
        )

    @staticmethod
    def backward(
        ctx,
        grad_grad_flat_angle_ebd,
        grad_grad_node_ebd,
        grad_grad_flat_edge_ebd,
        grad_grad_sub_angle,
        grad_grad_sub_node,
        grad_grad_sub_edge_ik,
        grad_grad_sub_edge_ij,
        grad_grad_bias,
    ):
        (
            grad_output,
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

        grad_grad_output = torch.zeros_like(grad_output)
        grad_flat_angle_ebd = torch.zeros_like(flat_angle_ebd)
        grad_flat_node_ebd = torch.zeros_like(flat_node_ebd)
        grad_flat_edge_ebd = torch.zeros_like(flat_edge_ebd)
        grad_sub_angle = torch.zeros_like(sub_angle)
        grad_sub_node = torch.zeros_like(sub_node)
        grad_sub_edge_ik = torch.zeros_like(sub_edge_ik)
        grad_sub_edge_ij = torch.zeros_like(sub_edge_ij)
        
        # angle
        if grad_grad_flat_angle_ebd is not None:
            grad_grad_output += torch.matmul(grad_grad_flat_angle_ebd, sub_angle)
            grad_sub_angle += torch.matmul(grad_grad_flat_angle_ebd.transpose(0, 1), grad_output)

        if grad_grad_sub_angle is not None:
            grad_grad_output += torch.matmul(flat_angle_ebd, grad_grad_sub_angle)
            grad_flat_angle_ebd += torch.matmul(grad_output, grad_grad_sub_angle.transpose(0, 1))
        
        # node
        gathered_node = torch.index_select(flat_node_ebd, 0, n2a_index)

        if grad_grad_node_ebd is not None:
            grad_grad_flat_node_ebd = grad_grad_node_ebd.reshape(flat_node_ebd.shape)
            gathered_grad_node = torch.index_select(grad_grad_flat_node_ebd, 0, n2a_index)
            grad_grad_output += torch.matmul(gathered_grad_node, sub_node)

            grad_sub_node += torch.matmul(gathered_grad_node.transpose(0, 1), grad_output)

        if grad_grad_sub_node is not None:
            grad_grad_output += torch.matmul(gathered_node, grad_grad_sub_node)
            grad_gathered_node = torch.matmul(grad_output, grad_grad_sub_node.transpose(0, 1))
            grad_flat_node_ebd.index_add_(0, n2a_index, grad_gathered_node)
        
        # edge_ik
        gathered_edge_ik = torch.index_select(flat_edge_ebd, 0, eik2a_index)

        if grad_grad_flat_edge_ebd is not None:
            gathered_grad_edge_ik = torch.index_select(grad_grad_flat_edge_ebd, 0, eik2a_index)
            grad_grad_output += torch.matmul(gathered_grad_edge_ik, sub_edge_ik)
            grad_sub_edge_ik += torch.matmul(gathered_grad_edge_ik.transpose(0, 1), grad_output)

        if grad_grad_sub_edge_ik is not None:
            grad_grad_output += torch.matmul(gathered_edge_ik, grad_grad_sub_edge_ik)
            grad_gathered_edge_ik = torch.matmul(grad_output, grad_grad_sub_edge_ik.transpose(0, 1))
            grad_flat_edge_ebd.index_add_(0, eik2a_index, grad_gathered_edge_ik)
        
        # edge_ij
        gathered_edge_ij = torch.index_select(flat_edge_ebd, 0, eij2a_index)

        if grad_grad_flat_edge_ebd is not None:
            gathered_grad_edge_ij = torch.index_select(grad_grad_flat_edge_ebd, 0, eij2a_index)
            grad_grad_output += torch.matmul(gathered_grad_edge_ij, sub_edge_ij)
            grad_sub_edge_ij += torch.matmul(gathered_grad_edge_ij.transpose(0, 1), grad_output)

        if grad_grad_sub_edge_ij is not None:
            grad_grad_output += torch.matmul(gathered_edge_ij, grad_grad_sub_edge_ij)
            grad_gathered_edge_ij = torch.matmul(grad_output, grad_grad_sub_edge_ij.transpose(0, 1))
            grad_flat_edge_ebd.index_add_(0, eij2a_index, grad_gathered_edge_ij)

        if grad_grad_bias is not None:
            grad_grad_output += grad_grad_bias.unsqueeze(0)

        return (
            grad_grad_output,       # 0  grad_output
            grad_flat_angle_ebd,    # 1  flat_angle_ebd
            grad_flat_node_ebd,     # 2  flat_node_ebd
            grad_flat_edge_ebd,     # 3  flat_edge_ebd
            None,                   # 4  n2a_index
            None,                   # 5  eij2a_index
            None,                   # 6  eik2a_index
            grad_sub_angle,         # 7  sub_angle
            grad_sub_node,          # 8  sub_node
            grad_sub_edge_ik,       # 9  sub_edge_ik
            grad_sub_edge_ij,       # 10 sub_edge_ij
            None,                   # 11 node_ebd_shape
        )
