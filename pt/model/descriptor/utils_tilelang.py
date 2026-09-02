import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def fused_cal_hg_dynamic_forward(
    M,  # n_edge
    N,  # 3 * e_dim
    E,  # e_dim
    NO, # num_owner
    dtype = "float32",
    accum_dtype = "float32",
    BLOCK_N: int = 64,
):
    assert M % NO == 0
    EDGES_PER_OWNER = M // NO

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

@tilelang.jit
def fused_call_hg_dynamic_backward(
    M,
    E,
    NB,
    NLOC,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_D: T.int32 = 64,
):
    @T.prim_func
    def hg_backward(
        grad_h2g2: T.Buffer((NB, NLOC, 3, E), dtype),
        flat_edge_ebd: T.Buffer((M, E), dtype),
        flat_h2: T.Buffer((M, 3), dtype),
        flat_sw: T.Buffer((M,), dtype),
        owner: T.Buffer((M,), "int64"),
        grad_flat_h2g2: T.Buffer((M, 3, E), accum_dtype),
        grad_flat_edge_ebd: T.Buffer((M, E), accum_dtype),
        grad_h2: T.Buffer((M, 3), accum_dtype),
        grad_flat_sw: T.Buffer((M,), accum_dtype),
    ):
        with T.Kernel(M, threads=BLOCK_D) as (bx,):
            m = bx
            tx = T.get_thread_binding()

            owner_m = owner[m]
            sw = flat_sw[m]

            h0 = flat_h2[m, 0]
            h1 = flat_h2[m, 1]
            h2 = flat_h2[m, 2]

            acc_h0 = T.alloc_var(T.float32)
            acc_h1 = T.alloc_var(T.float32)
            acc_h2 = T.alloc_var(T.float32)
            acc_sw = T.alloc_var(T.float32)

            acc_h0 = 0.0
            acc_h1 = 0.0
            acc_h2 = 0.0
            acc_sw = 0.0

            for d in T.serial(tx, E, BLOCK_D):
                g0 = grad_h2g2[0, owner_m, 0, d]
                g1 = grad_h2g2[0, owner_m, 1, d]
                g2 = grad_h2g2[0, owner_m, 2, d]

                edge = flat_edge_ebd[m, d]

                grad_flat_h2g2[m, 0, d] = g0
                grad_flat_h2g2[m, 1, d] = g1
                grad_flat_h2g2[m, 2, d] = g2

                acc_h0 += g0 * edge
                acc_h1 += g1 * edge
                acc_h2 += g2 * edge

                q = g0 * h0 + g1 * h1 + g2 * h2
                grad_flat_edge_ebd[m, d] = q * sw
                acc_sw += q * edge

            red_h0 = T.alloc_shared((1,), T.float32)
            red_h1 = T.alloc_shared((1,), T.float32)
            red_h2 = T.alloc_shared((1,), T.float32)
            red_sw = T.alloc_shared((1,), T.float32)

            sh_acc_h0 = T.alloc_shared((BLOCK_D,), T.float32)
            sh_acc_h1 = T.alloc_shared((BLOCK_D,), T.float32)
            sh_acc_h2 = T.alloc_shared((BLOCK_D,), T.float32)
            sh_acc_sw = T.alloc_shared((BLOCK_D,), T.float32)

            sh_acc_h0[tx] = acc_h0
            sh_acc_h1[tx] = acc_h1
            sh_acc_h2[tx] = acc_h2
            sh_acc_sw[tx] = acc_sw

            T.sync_threads()

            T.reduce_sum(sh_acc_h0, red_h0, dim=0)
            T.reduce_sum(sh_acc_h1, red_h1, dim=0)
            T.reduce_sum(sh_acc_h2, red_h2, dim=0)
            T.reduce_sum(sh_acc_sw, red_sw, dim=0)

            if tx == 0:
                grad_h2[m, 0] = red_h0[0] * sw
                grad_h2[m, 1] = red_h1[0] * sw
                grad_h2[m, 2] = red_h2[0] * sw
                grad_flat_sw[m] = red_sw[0]

    return hg_backward

@tilelang.jit
def fused_call_grrg_backward(
    NB,
    NLOC,
    E,
    A,
    dtype="float32",
    THREADS=128,
):
    @T.prim_func
    def grrg_backward(
        grad_grrg: T.Buffer((NB, NLOC, A * E), dtype),
        h2g2: T.Buffer((NB, NLOC, 3, E), dtype),
        grad_h2g2: T.Buffer((NB, NLOC, 3, E), dtype),
        scale_factor: T.float32,
    ):
        with T.Kernel(NB * NLOC, threads=THREADS) as (bx,):
            idx = bx
            nb_idx = idx // NLOC
            nloc_idx = idx % NLOC

            sh_G = T.alloc_shared((A, E), dtype)
            sh_H = T.alloc_shared((3, E), dtype)

            sh_right_tmp = T.alloc_shared((3, E, A), dtype)
            sh_right = T.alloc_shared((3, E), dtype)
            sh_left_tmp = T.alloc_shared((3, A, E), dtype)
            sh_left = T.alloc_shared((3, A), dtype)

            total_G = A * E
            for linear in T.Parallel(total_G):
                a = linear // E
                e = linear % E

                sh_G[a, e] = grad_grrg[nb_idx, nloc_idx, a * E + e]

            total_H = 3 * E
            for linear in T.Parallel(total_H):
                b = linear // E
                e = linear % E

                sh_H[b, e] = h2g2[nb_idx, nloc_idx, b, e]

            T.sync_threads()

            total_right_tmp = 3 * E * A
            for linear in T.Parallel(total_right_tmp):
                tmp = linear
                b = tmp // (E * A)
                rem = tmp % (E * A)
                e = rem // A
                a = rem % A

                sh_right_tmp[b, e, a] = sh_H[b, a] * sh_G[a, e]

            total_left_tmp = 3 * A * E
            for linear in T.Parallel(total_left_tmp):
                tmp = linear
                b = tmp // (A * E)
                rem = tmp % (A * E)
                a = rem // E
                k = rem % E

                sh_left_tmp[b, a, k] = sh_H[b, k] * sh_G[a, k]

            T.sync_threads()

            T.reduce_sum(sh_right_tmp, sh_right, dim=2)
            T.reduce_sum(sh_left_tmp, sh_left, dim=2)

            T.sync_threads()

            scale = scale_factor / 3.0

            total_out = 3 * E
            for linear in T.Parallel(total_out):
                b = linear // E
                e = linear % E
                if e < A:
                    grad_h2g2[nb_idx, nloc_idx, b, e] = (sh_right[b, e] + sh_left[b, e]) * scale
                else:
                    grad_h2g2[nb_idx, nloc_idx, b, e] = sh_right[b, e] * scale

    return grrg_backward

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
        
        assert flat_edge_ebd.shape[-1] == h2g2.shape[-1]
        assert grad_grrg.shape[-1] == axis_neuron * E
        
        grad_grrg = grad_grrg.contiguous()
        h2g2 = h2g2.contiguous()

        grad_h2g2 = torch.empty(
            (nb, nloc, 3, E),
            dtype=grad_grrg.dtype,
            device=grad_grrg.device,
        )

        grrg_backward_kernel = fused_call_grrg_backward(
            NB=nb,
            NLOC=nloc,
            E=E,
            A=axis_neuron,
        )

        grrg_backward_kernel(
            grad_grrg,
            h2g2,
            grad_h2g2,
            float(scale_factor),
        )

        # grad_g1 = grad_grrg.reshape(nb, nloc, axis_neuron, E)
        # h2g2m = h2g2[..., :axis_neuron]

        # grad_left = torch.matmul(grad_g1, h2g2.transpose(-1, -2))
        # grad_left = grad_left.transpose(-1, -2)

        # grad_right = torch.matmul(h2g2m, grad_g1)
        
        # grad_h2g2 = grad_right.clone()
        # grad_h2g2[..., :axis_neuron] += grad_left
        # grad_h2g2 /= 3.0
        # grad_h2g2 *= scale_factor

        owner = owner.long()
        grad_flat_h2g2 = torch.empty(
            (M, 3, E),
            dtype=grad_h2g2.dtype,
            device=grad_h2g2.device,
        )
        grad_flat_edge_ebd = torch.empty_like(flat_edge_ebd)
        grad_h2 = torch.empty_like(flat_h2)
        grad_flat_sw = torch.empty_like(flat_sw)

        hg_backward_kernel = fused_call_hg_dynamic_backward(M=M, E=E, NB=nb, NLOC=nloc)
        hg_backward_kernel(
            grad_h2g2,
            flat_edge_ebd,
            flat_h2,
            flat_sw,
            owner,
            grad_flat_h2g2,
            grad_flat_edge_ebd,
            grad_h2,
            grad_flat_sw,
        )

        # grad_flat_h2g2 = grad_h2g2[0, owner, :, :]

        # edge_scaled = flat_edge_ebd * flat_sw.unsqueeze(-1)
        # grad_h2 = (grad_flat_h2g2 * edge_scaled.unsqueeze(1)).sum(dim=-1)

        # grad_edge_scaled = (grad_flat_h2g2 * flat_h2.unsqueeze(-1)).sum(dim=1)
        # grad_flat_edge_ebd = grad_edge_scaled * flat_sw.unsqueeze(-1)

        # grad_flat_sw = (grad_edge_scaled * flat_edge_ebd).sum(dim=-1)

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

        edge_forward_kernel = fused_edge_update_forward(
            N_EDGES=n_edges,
            N_NODES_LOC=n_nodes_loc,
            N_NODES_EXT=n_nodes_ext,
            NODE_DIM=node_dim,
            EDGE_DIM=edge_dim,
            OUT_DIM=out_dim,
        )

        out = torch.empty((n_edges, out_dim), device=node_ebd.device, dtype=node_ebd.dtype,)
        # sub_node_update = torch.empty((n_edges, out_dim), device=node_ebd.device, dtype=node_ebd.dtype)

        edge_forward_kernel(node_ebd, node_ebd_ext, flat_edge_ebd, n2e_index,
            n_ext2e_index, node, node_ext, edge, bias, out, # sub_node_update)

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

@tilelang.jit
def fused_node_weight_backward_v1(
    E,
    N,
    K,
    D,
    dtype="float32",
    accum_dtype="float32",
):
    @T.prim_func
    def node_weight_backward(
        grad_out: T.Buffer((E, K), dtype),
        node_ebd: T.Buffer((N, D), dtype),
        n2e_index: T.Buffer((E,), "int64"),
        grad_node_weight: T.Buffer((D, K), accum_dtype),
    ):

        with T.Kernel(D, K, threads=128) as (bx, by):
            d = bx
            k = by

            acc = T.alloc_fragment((1,), accum_dtype)
            T.clear(acc)

            for e in T.serial(E):
                node_id = n2e_index[e]
                node_val = node_ebd[node_id, d]
                grad_val = grad_out[e, k]
                acc[0] += node_val * grad_val

            grad_node_weight[d, k] = acc[0]

    return node_weight_backward

@tilelang.jit
def fused_node_weight_backward_v2(
    E,
    N,
    K,
    D,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_D=32,
    BLOCK_K=32,
):
    @T.prim_func
    def node_weight_backward(
        grad_out: T.Buffer((E, K), dtype),
        node_ebd: T.Buffer((N, D), dtype),
        n2e_index: T.Buffer((E,), "int64"),
        grad_node_weight: T.Buffer((D, K), accum_dtype),
    ):

        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=128) as (bx, by):
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc)

            for e in T.serial(E):
                node_id = n2e_index[e]

                for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                    d = bx * BLOCK_D + di
                    k = by * BLOCK_K + ki

                    if d < D and k < K:
                        node_val = node_ebd[node_id, d]
                        grad_val = grad_out[e, k]
                        acc[di, ki] += node_val * grad_val

            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    grad_node_weight[d, k] = acc[di, ki]

    return node_weight_backward

@tilelang.jit
def fused_node_weight_backward_v3(
    E,
    N,
    K,
    D,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_D=32,
    BLOCK_K=32,
):
    assert E % N == 0
    EDGES_PER_NODE = E // N

    @T.prim_func
    def node_weight_backward(
        grad_out: T.Buffer((E, K), dtype),
        node_ebd: T.Buffer((N, D), dtype),
        n2e_index: T.Buffer((E,), "int64"),
        grad_node_weight: T.Buffer((D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=128) as (bx, by):
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc)

            for n in T.serial(N):
                edge_start = n * EDGES_PER_NODE
                for j in T.serial(EDGES_PER_NODE):
                    e = edge_start + j
                    for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                        d = bx * BLOCK_D + di
                        k = by * BLOCK_K + ki

                        if d < D and k < K:
                            node_val = node_ebd[n, d]
                            grad_val = grad_out[e, k]
                            acc[di, ki] += node_val * grad_val

            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    grad_node_weight[d, k] = acc[di, ki]

    return node_weight_backward

@tilelang.jit
def fused_node_weight_backward_v4(
    E,
    N,
    K,
    D,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_D=32,
    BLOCK_K=32,
):
    assert E % N == 0
    EDGES_PER_NODE = E // N

    @T.prim_func
    def node_weight_backward(
        grad_out: T.Buffer((E, K), dtype),
        node_ebd: T.Buffer((N, D), dtype),
        n2e_index: T.Buffer((E,), "int64"),
        grad_node_weight: T.Buffer((D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=128) as (bx, by):
            node_shared = T.alloc_shared((BLOCK_D,), dtype)
            grad_shared = T.alloc_shared((EDGES_PER_NODE, BLOCK_K), dtype)

            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc)

            d_start = bx * BLOCK_D
            k_start = by * BLOCK_K

            for n in T.serial(N):
                for di in T.Parallel(BLOCK_D):
                    d = d_start + di
                    if d < D:
                        node_shared[di] = node_ebd[n, d]
                    else:
                        node_shared[di] = 0.0

                T.sync_threads()

                for j, ki in T.Parallel(EDGES_PER_NODE, BLOCK_K):
                    e = n * EDGES_PER_NODE + j
                    k = k_start + ki
                    if k < K:
                        grad_shared[j, ki] = grad_out[e, k]
                    else:
                        grad_shared[j, ki] = 0.0

                T.sync_threads()

                for j in T.serial(EDGES_PER_NODE):
                    for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                        d = d_start + di
                        k = k_start + ki
                        if d < D and k < K:
                            acc[di, ki] += node_shared[di] * grad_shared[j, ki]

                T.sync_threads()

            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = d_start + di
                k = k_start + ki
                if d < D and k < K:
                    grad_node_weight[d, k] = acc[di, ki]

    return node_weight_backward

@tilelang.jit
def fused_node_backward_v1(
    E,
    N,
    K,
    D,
    dtype="float32",
    accum_dtype="float32",
):
    @T.prim_func
    def node_backward(
        grad_out: T.Buffer((E, K), dtype),
        n2e_index: T.Buffer((E,), "int64"),
        node_weight: T.Buffer((D, K), dtype),
        grad_node: T.Buffer((N, D), accum_dtype),
    ):

        with T.Kernel(N, D, threads=128) as (bx, by):
            node_id = bx
            d = by

            acc = T.alloc_fragment((1,), accum_dtype)
            T.clear(acc)

            for e in T.serial(E):
                current_node = n2e_index[e]
                if current_node == node_id:
                    for k in T.serial(K):
                        grad_val = grad_out[e, k]
                        weight_val = node_weight[d, k]
                        acc[0] += grad_val * weight_val

            grad_node[node_id, d] = acc[0]

    return node_backward

@tilelang.jit
def fused_node_backward_v2(
    E,
    N,
    K,
    D,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_K=32,
):
    assert E % N == 0
    EDGES_PER_NODE = E // N

    @T.prim_func
    def node_backward(
        grad_out: T.Buffer((E, K), dtype),
        n2e_index: T.Buffer((E,), "int64"),
        node_weight: T.Buffer((D, K), dtype),
        grad_node: T.Buffer((N, D), accum_dtype),
    ):
        with T.Kernel(N, T.ceildiv(D, 1), threads=128) as (bx, by):
            node_id = bx
            d = by

            acc = T.alloc_fragment((1,), accum_dtype)
            T.clear(acc)

            edge_start = node_id * EDGES_PER_NODE
            for j in T.serial(EDGES_PER_NODE):
                e = edge_start + j
                for k in T.serial(K):
                    grad_val = grad_out[e, k]
                    weight_val = node_weight[d, k]

                    acc[0] += grad_val * weight_val

            grad_node[node_id, d] = acc[0]

    return node_backward

@tilelang.jit
def fused_node_backward_v3(
    E,
    N,
    K,
    D,
    dtype="float32",
    accum_dtype="float32",
):
    assert E % N == 0
    EDGES_PER_NODE = E // N

    @T.prim_func
    def node_backward(
        grad_out: T.Buffer((E, K), dtype),
        n2e_index: T.Buffer((E,), "int64"),
        node_weight: T.Buffer((D, K), dtype),
        grad_node: T.Buffer((N, D), accum_dtype),
    ):
        with T.Kernel(N, D, threads=128) as (bx, by):
            node_id = bx
            d = by

            acc = T.alloc_fragment((1,), accum_dtype)
            T.clear(acc)

            edge_start = node_id * EDGES_PER_NODE
            for j in T.serial(EDGES_PER_NODE):
                e = edge_start + j
                for k in T.serial(K):
                    acc[0] += grad_out[e, k] * node_weight[d, k]

            grad_node[node_id, d] = acc[0]

    return node_backward

@tilelang.jit
def fused_node_backward_v4(
    E,
    N,
    K,
    D,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_K=32,
):
    assert E % N == 0
    EDGES_PER_NODE = E // N

    @T.prim_func
    def node_backward(
        grad_out: T.Buffer((E, K), dtype),
        n2e_index: T.Buffer((E,), "int64"),
        node_weight: T.Buffer((D, K), dtype),
        grad_node: T.Buffer((N, D), accum_dtype),
    ):
        with T.Kernel(N, D, threads=128) as (bx, by):
            node_id = bx
            d = by

            acc = T.alloc_fragment((1,), accum_dtype)
            T.clear(acc)

            edge_start = node_id * EDGES_PER_NODE
            for kt in T.serial(T.ceildiv(K, BLOCK_K)):
                k_base = kt * BLOCK_K
                for kk in T.serial(BLOCK_K):
                    k = k_base + kk
                    if k < K:
                        for j in T.serial(EDGES_PER_NODE):
                            e = edge_start + j
                            acc[0] += grad_out[e, k] * node_weight[d, k]

            grad_node[node_id, d] = acc[0]

    return node_backward

@tilelang.jit
def fused_node_backward_v5(
    E,
    N,
    K,
    D,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_D=32,
    BLOCK_K=32,
):
    assert E % N == 0
    EDGES_PER_NODE = E // N

    @T.prim_func
    def node_backward(
        grad_out: T.Buffer((E, K), dtype),
        n2e_index: T.Buffer((E,), "int64"),
        node_weight: T.Buffer((D, K), dtype),
        grad_node: T.Buffer((N, D), accum_dtype),
    ):
        with T.Kernel(N, T.ceildiv(D, BLOCK_D), threads=128) as (bx, by):
            node_id = bx
            d_base = by * BLOCK_D
            edge_start = node_id * EDGES_PER_NODE

            weight_shared = T.alloc_shared((BLOCK_D, BLOCK_K), dtype)
            grad_shared = T.alloc_shared((EDGES_PER_NODE, BLOCK_K), dtype)

            acc = T.alloc_fragment((BLOCK_D,), accum_dtype)
            T.clear(acc)

            for kt in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                k_base = kt * BLOCK_K
                for d_local, k_local in T.Parallel(BLOCK_D, BLOCK_K):
                    d_load = d_base + d_local
                    k_load = k_base + k_local
                    if d_load < D and k_load < K:
                        weight_shared[d_local, k_local] = node_weight[d_load, k_load]
                    else:
                        weight_shared[d_local, k_local] = 0.0

                for edge_local, k_local in T.Parallel(EDGES_PER_NODE, BLOCK_K):
                    e = edge_start + edge_local
                    k_load = k_base + k_local
                    if k_load < K:
                        grad_shared[edge_local, k_local] = grad_out[e, k_load]
                    else:
                        grad_shared[edge_local, k_local] = 0.0

                T.sync_threads()

                for d_local in T.Parallel(BLOCK_D):
                    d = d_base + d_local
                    if d < D:
                        for kk in T.serial(BLOCK_K):
                            k = k_base + kk
                            if k < K:
                                weight_val = weight_shared[d_local, kk]
                                for j in T.serial(EDGES_PER_NODE):
                                    acc[d_local] += grad_shared[j, kk] * weight_val

                T.sync_threads()

            for d_local in T.Parallel(BLOCK_D):
                d = d_base + d_local
                if d < D:
                    grad_node[node_id, d] = acc[d_local]

    return node_backward

@tilelang.jit
def fused_node_ext_weight_backward(
    E,
    N,
    K,
    D,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_M=32,
    BLOCK_D=32,
    BLOCK_K=32,
):
    @T.prim_func
    def node_ext_weight_backward(
        node_ebd_ext: T.Buffer((N, D), dtype),
        n_ext2e_index: T.Buffer((E,), "int64"),
        grad_out: T.Buffer((E, K), dtype),
        grad_node_ext_weight: T.Buffer((D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=128) as (bx, by):
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc)

            d_start = bx * BLOCK_D
            k_start = by * BLOCK_K

            for e in T.serial(E):
                n = n_ext2e_index[e]
                for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                    d = d_start + di
                    k = k_start + ki
                    if d < D and k < K:
                        acc[di, ki] += node_ebd_ext[n, d] * grad_out[e, k]

            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = d_start + di
                k = k_start + ki
                if d < D and k < K:
                    grad_node_ext_weight[d, k] = acc[di, ki]

    return node_ext_weight_backward

@tilelang.jit
def fused_node_ext_weight_backward_v2(
    E,
    N,
    K,
    D,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_M=32,
    BLOCK_D=32,
    BLOCK_K=32
):
    @T.prim_func
    def node_ext_weight_backward(
        node_ebd_ext: T.Buffer((N, D), dtype),
        n_ext2e_index: T.Buffer((E,), "int64"),
        grad_out: T.Buffer((E, K), dtype),
        grad_node_ext_weight: T.Buffer((D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=128) as (bx, by):
            node_shared = T.alloc_shared((N, BLOCK_D), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc)

            d_start = bx * BLOCK_D
            k_start = by * BLOCK_K
            for n, di in T.Parallel(N, BLOCK_D):
                d = d_start + di
                if d < D:
                    node_shared[n, di] = node_ebd_ext[n, d]
                else:
                    node_shared[n, di] = 0.0

            T.sync_threads()
            for e in T.serial(E):
                n = n_ext2e_index[e]
                for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                    d = d_start + di
                    k = k_start + ki
                    if d < D and k < K:
                        acc[di, ki] += node_shared[n, di] * grad_out[e, k]
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = d_start + di
                k = k_start + ki
                if d < D and k < K:
                    grad_node_ext_weight[d, k] = acc[di, ki]

    return node_ext_weight_backward

@tilelang.jit
def fused_node_ext_weight_backward_v3(
    E,
    N,
    K,
    D,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_M=32,
    BLOCK_D=32,
    BLOCK_K=32
):
    @T.prim_func
    def node_ext_weight_backward(
        node_ebd_ext: T.Buffer((N, D), dtype),
        n_ext2e_index: T.Buffer((E,), "int64"),
        grad_out: T.Buffer((E, K), dtype),
        grad_node_ext_weight: T.Buffer((D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=128) as (bx, by):
            node_shared = T.alloc_shared((N, BLOCK_D), dtype)
            grad_shared = T.alloc_shared((E, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc)

            d_start = bx * BLOCK_D
            k_start = by * BLOCK_K
            for n, di in T.Parallel(N, BLOCK_D):
                d = d_start + di
                if d < D:
                    node_shared[n, di] = node_ebd_ext[n, d]
                else:
                    node_shared[n, di] = 0.0
            
            for e, ki in T.Parallel(E, BLOCK_K):
                k = k_start + ki
                if k < K:
                    grad_shared[e, ki] = grad_out[e, k]
                else:
                    grad_shared[e, ki] = 0.0

            T.sync_threads()
            for e in T.serial(E):
                n = n_ext2e_index[e]
                for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                    d = d_start + di
                    k = k_start + ki
                    if d < D and k < K:
                        acc[di, ki] += node_shared[n, di] * grad_shared[e, ki]
            
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = d_start + di
                k = k_start + ki
                if d < D and k < K:
                    grad_node_ext_weight[d, k] = acc[di, ki]

    return node_ext_weight_backward

@tilelang.jit
def fused_node_ext_weight_backward_v4(
    E,
    N,
    K,
    D,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_D=32,
    BLOCK_K=32,
):
    E_PAD = T.ceildiv(E, 8) * 8

    @T.prim_func
    def node_ext_weight_backward(
        node_ebd_ext: T.Buffer((N, D), dtype),
        n_ext2e_index: T.Buffer((E,), "int64"),
        grad_out: T.Buffer((E, K), dtype),
        grad_node_ext_weight: T.Buffer((D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=128) as (bx, by):
            d_start = bx * BLOCK_D
            k_start = by * BLOCK_K
            
            node_shared = T.alloc_shared((BLOCK_D, E_PAD), dtype)
            grad_shared = T.alloc_shared((E_PAD, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc)

            for di, e in T.Parallel(BLOCK_D, E_PAD):
                d = d_start + di
                if e < E and d < D:
                    n = n_ext2e_index[e]
                    node_shared[di, e] = node_ebd_ext[n, d]
                else:
                    node_shared[di, e] = 0.0

            for e, ki in T.Parallel(E_PAD, BLOCK_K):
                k = k_start + ki
                if e < E and k < K:
                    grad_shared[e, ki] = grad_out[e, k]
                else:
                    grad_shared[e, ki] = 0.0

            T.sync_threads()

            T.gemm(node_shared, grad_shared, acc)
            
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = d_start + di
                k = k_start + ki
                if d < D and k < K:
                    grad_node_ext_weight[d, k] = acc[di, ki]

    return node_ext_weight_backward

@tilelang.jit
def fused_node_ext_backward_v1(
    E,
    K,
    D,
    N,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_E=32,
    BLOCK_D=32,
    BLOCK_K=32,
):
    @T.prim_func
    def node_ext_backward(
        grad_out: T.Buffer((E, K), dtype),
        node_ext_weight: T.Buffer((D, K), dtype),
        n_ext2e_index: T.Buffer((E,), "int64"),
        grad_node_ext: T.Buffer((N, D), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(E, BLOCK_E), T.ceildiv(D, BLOCK_D), threads=128) as (bx, by):
            grad_out_shared = T.alloc_shared((BLOCK_E, BLOCK_K), dtype)
            node_ext_shared = T.alloc_shared((BLOCK_K, BLOCK_D), dtype)
            gemm_out = T.alloc_fragment((BLOCK_E, BLOCK_D), accum_dtype)
            T.clear(gemm_out)

            for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                for e, k in T.Parallel(BLOCK_E, BLOCK_K):
                    e_idx = bx * BLOCK_E + e
                    k_idx = ko * BLOCK_K + k
                    if e_idx < E and k_idx < K:
                        grad_out_shared[e, k] = grad_out[e_idx, k_idx]
                    else:
                        grad_out_shared[e, k] = 0
                
                for d, k in T.Parallel(BLOCK_D, BLOCK_K):
                    d_idx = by * BLOCK_D + d
                    k_idx = ko * BLOCK_K + k
                    if d_idx < D and k_idx < K:
                        node_ext_shared[k, d] = node_ext_weight[d_idx, k_idx]
                    else:
                        node_ext_shared[k, d] = 0
                
                T.gemm(grad_out_shared, node_ext_shared, gemm_out)
            
            for e, d in T.Parallel(BLOCK_E, BLOCK_D):
                e_idx = bx * BLOCK_E + e
                d_idx = by * BLOCK_D + d
                if e_idx < E and d_idx < D:
                    n_idx = n_ext2e_index[e_idx]
                    T.atomic_add(grad_node_ext[n_idx, d_idx], gemm_out[e, d])

    return node_ext_backward

@tilelang.jit
def fused_node_ext_backward_v2(
    E,
    K,
    D,
    N,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_E=32,
    BLOCK_D=32,
    BLOCK_K=32,
):
    @T.prim_func
    def node_ext_backward(
        grad_out: T.Buffer((E, K), dtype),
        node_ext_weight: T.Buffer((D, K), dtype),
        n_ext2e_index: T.Buffer((E,), "int64"),
        grad_node_ext: T.Buffer((N, D), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(E, BLOCK_E), T.ceildiv(D, BLOCK_D), threads=128) as (bx, by):
            grad_out_shared = T.alloc_shared((BLOCK_E, BLOCK_K), dtype)
            node_ext_shared = T.alloc_shared((BLOCK_K, BLOCK_D), dtype)

            gemm_out = T.alloc_fragment((BLOCK_E, BLOCK_D), accum_dtype)
            T.clear(gemm_out)

            for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                for e, k in T.Parallel(BLOCK_E, BLOCK_K):
                    e_idx = bx * BLOCK_E + e
                    k_idx = ko * BLOCK_K + k
                    if e_idx < E and k_idx < K:
                        grad_out_shared[e, k] = grad_out[e_idx, k_idx]
                    else:
                        grad_out_shared[e, k] = 0

                for d, k in T.Parallel(BLOCK_D, BLOCK_K):
                    d_idx = by * BLOCK_D + d
                    k_idx = ko * BLOCK_K + k
                    if d_idx < D and k_idx < K:
                        node_ext_shared[k, d] = node_ext_weight[d_idx, k_idx]
                    else:
                        node_ext_shared[k, d] = 0

                T.gemm(grad_out_shared, node_ext_shared, gemm_out)

            for e, d in T.Parallel(BLOCK_E, BLOCK_D):
                e_idx = bx * BLOCK_E + e
                d_idx = by * BLOCK_D + d
                if e_idx < E and d_idx < D:
                    n_idx = n_ext2e_index[e_idx]
                    T.atomic_add(grad_node_ext[n_idx, d_idx], gemm_out[e, d])

    return node_ext_backward

@tilelang.jit
def fused_node_ext_backward_v3(
    E,
    K,
    D,
    N,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_E=32,
    BLOCK_D=32,
    BLOCK_K=32,
):
    @T.prim_func
    def node_ext_backward(
        grad_out: T.Buffer((E, K), dtype),
        node_ext_weight: T.Buffer((D, K), dtype),
        n_ext2e_index: T.Buffer((E,), "int64"),
        grad_node_ext: T.Buffer((N, D), accum_dtype),
        grad_bias: T.Buffer((K,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(E, BLOCK_E), T.ceildiv(D, BLOCK_D), threads=128) as (bx, by):
            grad_out_shared = T.alloc_shared((BLOCK_E, BLOCK_K), dtype)
            node_ext_shared = T.alloc_shared((BLOCK_K, BLOCK_D), dtype)
            
            gemm_out = T.alloc_fragment((BLOCK_E, BLOCK_D), accum_dtype)
            bias_accum = T.alloc_fragment((BLOCK_K,), accum_dtype)
            T.clear(gemm_out)
            T.clear(bias_accum)

            for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                for e, k in T.Parallel(BLOCK_E, BLOCK_K):
                    e_idx = bx * BLOCK_E + e
                    k_idx = ko * BLOCK_K + k
                    if e_idx < E and k_idx < K:
                        grad_out_shared[e, k] = grad_out[e_idx, k_idx]
                    else:
                        grad_out_shared[e, k] = 0

                for d, k in T.Parallel(BLOCK_D, BLOCK_K):
                    d_idx = by * BLOCK_D + d
                    k_idx = ko * BLOCK_K + k
                    if d_idx < D and k_idx < K:
                        node_ext_shared[k, d] = node_ext_weight[d_idx, k_idx]
                    else:
                        node_ext_shared[k, d] = 0

                if by == 0:
                    for k in T.Parallel(BLOCK_K):
                        for e in T.serial(BLOCK_E):
                            bias_accum[k] += T.cast(grad_out_shared[e, k], accum_dtype)

                    for k in T.Parallel(BLOCK_K):
                        k_idx = ko * BLOCK_K + k
                        if k_idx < K:
                            T.atomic_add(grad_bias[k_idx], bias_accum[k])

                T.gemm(grad_out_shared, node_ext_shared, gemm_out)

            for e, d in T.Parallel(BLOCK_E, BLOCK_D):
                e_idx = bx * BLOCK_E + e
                d_idx = by * BLOCK_D + d
                if e_idx < E and d_idx < D:
                    n_idx = n_ext2e_index[e_idx]
                    T.atomic_add(grad_node_ext[n_idx, d_idx], gemm_out[e, d])

    return node_ext_backward

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
        E, K = grad_out.shape
        N_node, D_node = node_ebd.shape
        N_ext, D_ext = node_ebd_ext.shape
        E_edge, D_edge = flat_edge_ebd.shape

        grad_edge_ebd = grad_out @ edge_weight.T
        grad_edge_weight = flat_edge_ebd.T @ grad_out

        # gathered_node = node_ebd[n2e_index]
        # grad_node_weight = gathered_node.T @ grad_out
        grad_node_weight = torch.zeros(
            (D_node, K),
            device=grad_out.device,
            dtype=grad_out.dtype,
        )
        node_weight_kernel = fused_node_weight_backward_v4(E=E, N=N_node, K=K, D=D_node)
        node_weight_kernel(grad_out, node_ebd, n2e_index, grad_node_weight)

        # grad_gathered_node = grad_out @ node_weight.T
        # grad_node.index_add_(0, n2e_index, grad_gathered_node)
        grad_node = torch.zeros(
            (N_node, D_node),
            device=grad_out.device,
            dtype=grad_out.dtype,
        )
        node_kernel = fused_node_backward_v5(E, N_node, K, D_node)
        node_kernel(grad_out, n2e_index, node_weight, grad_node)
        
        # gathered_node_ext = node_ebd_ext[n_ext2e_index]
        # grad_node_ext_weight = gathered_node_ext.T @ grad_out
        grad_node_ext_weight = torch.zeros(
            (D_ext, K),
            device=grad_out.device,
            dtype=grad_out.dtype,
        )
        node_ext_weight_kernel = fused_node_ext_weight_backward_v4(E=E, N=N_ext, K=K, D=D_ext)
        node_ext_weight_kernel(node_ebd_ext, n_ext2e_index, grad_out, grad_node_ext_weight)
        
        grad_node_ext = torch.zeros(
            (N_ext, D_ext),
            device=grad_out.device,
            dtype=grad_out.dtype,
        )
        grad_bias = torch.zeros((K,), device=grad_out.device, dtype=grad_out.dtype)
        # grad_gathered_node_ext = grad_out @ node_ext_weight.T
        # grad_node_ext.index_add_(0, n_ext2e_index, grad_gathered_node_ext)
        # grad_bias = grad_out.sum(dim=0)

        node_ext_kernel = fused_node_ext_backward_v3(E=E, K=K, D=D_ext, N=N_ext)
        node_ext_kernel(grad_out, node_ext_weight, n_ext2e_index, grad_node_ext, grad_bias)

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
        
        grad_grad_out = torch.zeros_like(grad_out)

        if grad_grad_edge_ebd is not None:
            grad_grad_out = grad_grad_out + grad_grad_edge_ebd @ edge_weight

        if grad_grad_edge_weight is not None:
            grad_grad_out = grad_grad_out + flat_edge_ebd @ grad_grad_edge_weight

        gathered_node = node_ebd[n2e_index]
        if grad_grad_node_weight is not None:
            grad_grad_out = grad_grad_out + gathered_node @ grad_grad_node_weight

        if grad_grad_node is not None:
            gathered_grad_node = grad_grad_node[n2e_index]
            grad_grad_out = grad_grad_out + gathered_grad_node @ node_weight

        gathered_node_ext = node_ebd_ext[n_ext2e_index]
        if grad_grad_node_ext_weight is not None:
            grad_grad_out = grad_grad_out + gathered_node_ext @ grad_grad_node_ext_weight

        if grad_grad_node_ext is not None:
            gathered_grad_node_ext = grad_grad_node_ext[n_ext2e_index]
            grad_grad_out = grad_grad_out + gathered_grad_node_ext @ node_ext_weight

        if grad_grad_bias is not None:
            grad_grad_out = grad_grad_out + grad_grad_bias.unsqueeze(0).expand_as(grad_out)

        grad_grad_node_ebd = torch.zeros_like(node_ebd)

        if grad_grad_node_weight is not None:
            grad_gathered_node = grad_out @ grad_grad_node_weight.T

            grad_grad_node_ebd.index_add_(0, n2e_index, grad_gathered_node)

        grad_grad_node_ebd_ext = torch.zeros_like(node_ebd_ext)
        if grad_grad_node_ext_weight is not None:
            grad_gathered_node_ext = grad_out @ grad_grad_node_ext_weight.T
            grad_grad_node_ebd_ext.index_add_(0, n_ext2e_index, grad_gathered_node_ext)

        grad_grad_flat_edge_ebd = torch.zeros_like(flat_edge_ebd)
        if grad_grad_edge_weight is not None:
            grad_grad_flat_edge_ebd = grad_out @ grad_grad_edge_weight.T

        grad_grad_node_weight_input = torch.zeros_like(node_weight)
        if grad_grad_node is not None:
            gathered_grad_node = grad_grad_node[n2e_index]
            grad_grad_node_weight_input = gathered_grad_node.T @ grad_out

        grad_grad_node_ext_weight_input = torch.zeros_like(node_ext_weight)
        if grad_grad_node_ext is not None:
            gathered_grad_node_ext = grad_grad_node_ext[n_ext2e_index]
            grad_grad_node_ext_weight_input = gathered_grad_node_ext.T @ grad_out

        grad_grad_edge_weight_input = torch.zeros_like(edge_weight)
        if grad_grad_edge_ebd is not None:
            grad_grad_edge_weight_input = grad_grad_edge_ebd.T @ grad_out

        return (
            grad_grad_out,
            grad_grad_node_ebd,
            grad_grad_node_ebd_ext,
            grad_grad_flat_edge_ebd,
            None,
            None,
            grad_grad_node_weight_input,
            grad_grad_node_ext_weight_input,
            grad_grad_edge_weight_input,
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

@tilelang.jit
def fused_angle_node_backward(
    M,
    N,
    K,
    A,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_M=32,
    BLOCK_N=32,
    BLOCK_K=32,
):
    @T.prim_func
    def angle_node_backward(
        grad_output: T.Buffer((M, K), dtype),
        sub_node: T.Buffer((N, K), dtype),
        n2a_index: T.Buffer((M,), "int64"),
        grad_flat_node_ebd: T.Buffer((A, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(N, BLOCK_N), threads=128) as (bx, by):
            output_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            node_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
            gathered_node_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)

            gathered_node_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            T.clear(gathered_node_local)

            for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                    m = bx * BLOCK_M + i
                    kk = k * BLOCK_K + j
                    if m < M and kk < K:
                        output_shared[i, j] = grad_output[m, kk]
                    else:
                        output_shared[i, j] = 0

                for i, j in T.Parallel(BLOCK_K, BLOCK_N):
                    kk = k * BLOCK_K + i
                    n = by * BLOCK_N + j
                    if kk < K and n < N:
                        node_shared[i, j] = sub_node[n, kk]
                    else:
                        node_shared[i, j] = 0

                T.gemm(output_shared, node_shared, gathered_node_local)

            T.copy(gathered_node_local, gathered_node_shared)

            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                m = bx * BLOCK_M + i
                n = by * BLOCK_N + j
                if m < M and n < N:
                    node = n2a_index[m]
                    T.atomic_add(grad_flat_node_ebd[node, n], gathered_node_shared[i, j])

    return angle_node_backward

@tilelang.jit
def fused_angle_node_backward_v2(
    M,
    A,
    N,
    K,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_M=16,
    BLOCK_N=32,
    BLOCK_K=32,
):
    @T.prim_func
    def angle_node_backward(
        grad_output: T.Buffer((M, K), dtype),
        sub_node: T.Buffer((N, K), dtype),
        node_start: T.Buffer((A,), "int32"),
        node_count: T.Buffer((A,), "int32"),
        grad_flat_node_ebd: T.Buffer((A, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(A, BLOCK_M), T.ceildiv(N, BLOCK_N), threads=128) as (bx, by):
            reduced_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            node_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)

            output_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            T.clear(output_local)

            node_base = bx * BLOCK_M

            for k0 in T.serial(T.ceildiv(K, BLOCK_K)):
                for mi, kk in T.Parallel(BLOCK_M, BLOCK_K):
                    node = node_base + mi
                    k = k0 * BLOCK_K + kk
                    reduced_shared[mi, kk] = 0
                    if node < A and k < K:
                        start = node_start[node]
                        count = node_count[node]
                        for m in T.serial(count):
                            row = start + m
                            if row < M:
                                reduced_shared[mi, kk] += grad_output[row, k]

                T.sync_threads()

                for kk, nn in T.Parallel(BLOCK_K, BLOCK_N):
                    k = k0 * BLOCK_K + kk
                    n = by * BLOCK_N + nn
                    if k < K and n < N:
                        node_shared[kk, nn] = sub_node[n, k]
                    else:
                        node_shared[kk, nn] = 0

                T.sync_threads()

                T.gemm(reduced_shared, node_shared, output_local)

                T.sync_threads()

            for mi, nn in T.Parallel(BLOCK_M, BLOCK_N):
                node = node_base + mi
                n = by * BLOCK_N + nn
                if node < A and n < N:
                    grad_flat_node_ebd[node, n] = output_local[mi, nn]

    return angle_node_backward

@tilelang.jit
def fused_angle_node_backward_v3(
    M,
    A,
    N,
    K,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_M=16,
    BLOCK_N=32,
    BLOCK_K=32,
    REDUCE_THREADS=8,
):
    assert BLOCK_M * REDUCE_THREADS == 128

    @T.prim_func
    def angle_node_backward(
        grad_output: T.Buffer((M, K), dtype),
        sub_node: T.Buffer((N, K), dtype),
        node_start: T.Buffer((A,), "int32"),
        node_count: T.Buffer((A,), "int32"),
        grad_flat_node_ebd: T.Buffer((A, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(A, BLOCK_M), T.ceildiv(N, BLOCK_N), threads=128) as (bx, by):
            reduced_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            partial_shared = T.alloc_shared((BLOCK_M, REDUCE_THREADS, BLOCK_K), dtype)
            node_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
            output_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            T.clear(output_local)

            tx = T.get_thread_binding(0)

            node_slot = tx // REDUCE_THREADS
            lane = tx % REDUCE_THREADS

            node_base = bx * BLOCK_M
            for k0 in T.serial(T.ceildiv(K, BLOCK_K)):
                for kk in T.serial(BLOCK_K):
                    k = k0 * BLOCK_K + kk
                    acc = T.alloc_var(accum_dtype, init=0)
                    if node_slot < BLOCK_M:
                        node = node_base + node_slot
                        if node < A and k < K:
                            start = node_start[node]
                            count = node_count[node]
                            for r in T.serial(16):
                                row_offset = lane + r * REDUCE_THREADS
                                if row_offset < count:
                                    row = start + row_offset
                                    if row < M:
                                        acc += T.cast(grad_output[row, k], accum_dtype)
                    partial_shared[node_slot, lane, kk] = T.cast(acc, dtype)
                T.sync_threads()

                if node_slot < BLOCK_M and lane == 0:
                    node = node_base + node_slot
                    for kk in T.serial(BLOCK_K):
                        k = k0 * BLOCK_K + kk
                        acc = T.alloc_var(accum_dtype, init=0)
                        if node < A and k < K:
                            for r in T.serial(REDUCE_THREADS):
                                acc += T.cast(partial_shared[node_slot, r, kk], accum_dtype)
                        reduced_shared[node_slot, kk] = T.cast(acc, dtype)
                T.sync_threads()

                for kk, nn in T.Parallel(BLOCK_K, BLOCK_N):
                    k = k0 * BLOCK_K + kk
                    n = by * BLOCK_N + nn
                    if k < K and n < N:
                        node_shared[kk, nn] = sub_node[n, k]
                    else:
                        node_shared[kk, nn] = 0

                T.sync_threads()

                T.gemm(reduced_shared, node_shared, output_local)

                T.sync_threads()

            for mi, nn in T.Parallel(BLOCK_M, BLOCK_N):
                node = node_base + mi
                n = by * BLOCK_N + nn
                if node < A and n < N:
                    grad_flat_node_ebd[node, n] = output_local[mi, nn]

    return angle_node_backward

@tilelang.jit
def fused_angle_node_backward_v3_1(
    M,
    A,
    N,
    K,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_M=16,
    BLOCK_N=32,
    BLOCK_K=32,
    REDUCE_THREADS=8,
):
    assert BLOCK_M * REDUCE_THREADS == 128
    assert REDUCE_THREADS == 8

    @T.prim_func
    def angle_node_backward(
        grad_output: T.Buffer((M, K), dtype),
        sub_node: T.Buffer((N, K), dtype),
        node_start: T.Buffer((A,), "int32"),
        node_count: T.Buffer((A,), "int32"),
        grad_flat_node_ebd: T.Buffer((A, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(A, BLOCK_M), T.ceildiv(N, BLOCK_N), threads=128) as (bx, by):
            reduced_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            node_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)

            output_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            T.clear(output_local)

            tx = T.get_thread_binding(0)

            node_slot = tx // REDUCE_THREADS
            lane = tx % REDUCE_THREADS

            node_base = bx * BLOCK_M

            for k0 in T.serial(T.ceildiv(K, BLOCK_K)):
                for kk in T.serial(BLOCK_K):
                    k = k0 * BLOCK_K + kk
                    acc = T.alloc_var(accum_dtype, init=0)
                    if node_slot < BLOCK_M:
                        node = node_base + node_slot
                        if node < A and k < K:
                            start = node_start[node]
                            count = node_count[node]
                            for r in T.serial(16):
                                row_offset = lane + r * REDUCE_THREADS
                                if row_offset < count:
                                    row = start + row_offset
                                    if row < M:
                                        acc += T.cast(grad_output[row, k], accum_dtype)
                    acc += T.shfl_down(acc, 4, 8)
                    acc += T.shfl_down(acc, 2, 8)
                    acc += T.shfl_down(acc, 1, 8)

                    if node_slot < BLOCK_M and lane == 0:
                        reduced_shared[node_slot, kk] = T.cast(acc, dtype)

                T.sync_threads()

                for kk, nn in T.Parallel(BLOCK_K, BLOCK_N):
                    k = k0 * BLOCK_K + kk
                    n = by * BLOCK_N + nn
                    if k < K and n < N:
                        node_shared[kk, nn] = sub_node[n, k]
                    else:
                        node_shared[kk, nn] = 0

                T.sync_threads()
                
                T.gemm(reduced_shared, node_shared, output_local)

                T.sync_threads()

            for mi, nn in T.Parallel(BLOCK_M, BLOCK_N):
                node = node_base + mi
                n = by * BLOCK_N + nn
                if node < A and n < N:
                    grad_flat_node_ebd[node, n] = output_local[mi, nn]

    return angle_node_backward

def build_edge_backward_metadata(
    eik2a_index: torch.Tensor,
    eij2a_index: torch.Tensor,
):
    assert eik2a_index.dim() == 1
    assert eij2a_index.dim() == 1
    assert eik2a_index.numel() == eij2a_index.numel()

    target_ids_ik, _, count_ik = torch.unique(
        eik2a_index,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )

    target_ids_ij, _, count_ij = torch.unique(
        eij2a_index,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )

    if not torch.equal(target_ids_ik, target_ids_ij):
        raise RuntimeError(
            "eik2a_index and eij2a_index do not have the same target ids."
        )

    if not torch.equal(count_ik, count_ij):
        raise RuntimeError(
            "eik2a_index and eij2a_index do not have the same group sizes."
        )

    target_ids = target_ids_ik
    group_count = count_ik

    num_groups = target_ids.numel()
    max_group = int(group_count.max().item())

    order_ik = torch.argsort(eik2a_index, stable=True)

    sorted_group_ik = torch.repeat_interleave(
        torch.arange(num_groups, device=eik2a_index.device, dtype=torch.int64),
        group_count,
    )

    group_start = torch.cumsum(group_count, dim=0) - group_count

    occurrence_ik = (
        torch.arange(eik2a_index.numel(), device=eik2a_index.device, dtype=torch.int64)
        - torch.repeat_interleave(group_start, group_count)
    )

    eik_pos = torch.full((num_groups, max_group), -1, dtype=torch.int64, device=eik2a_index.device)

    eik_pos[sorted_group_ik, occurrence_ik] = order_ik

    order_ij = torch.argsort(eij2a_index, stable=True)

    sorted_group_ij = torch.repeat_interleave(
        torch.arange(num_groups, device=eij2a_index.device, dtype=torch.int64),
        group_count,
    )

    occurrence_ij = (
        torch.arange(eij2a_index.numel(), device=eij2a_index.device, dtype=torch.int64)
        - torch.repeat_interleave(group_start, group_count)
    )

    eij_pos = torch.full((num_groups, max_group), -1, dtype=torch.int64, device=eij2a_index.device)

    eij_pos[sorted_group_ij, occurrence_ij] = order_ij

    return (
        target_ids,
        eik_pos,
        eij_pos,
        group_count,
    )

@tilelang.jit
def fused_edge_ik_ij_backward_v1(
    G,
    N,
    E,
    K,
    D,
    max_group,
    BLOCK_K=32,
    BLOCK_D=32,
    threads=128,
    dtype="float32",
    accum_dtype="float32",
):
    @T.prim_func
    def edge_backward(
        grad_output: T.Tensor((E, D), dtype),
        sub_edge_ik: T.Tensor((K, D), dtype),
        sub_edge_ij: T.Tensor((K, D), dtype),
        target_ids: T.Tensor((G,), "int64"),
        eik_pos: T.Tensor((G, max_group), "int64"),
        eij_pos: T.Tensor((G, max_group), "int64"),
        group_count: T.Tensor((G,), "int64"),
        grad_flat_edge_ebd: T.Tensor((N, K), accum_dtype),
    ):
        with T.Kernel(G, threads=threads) as bx:
            grad_s_ik = T.alloc_shared((max_group, BLOCK_D), dtype)
            grad_s_ij = T.alloc_shared((max_group, BLOCK_D), dtype)
            sub_s_ik = T.alloc_shared((BLOCK_K, BLOCK_D), dtype)
            sub_s_ij = T.alloc_shared((BLOCK_K, BLOCK_D), dtype)
            acc = T.alloc_fragment((BLOCK_K,), accum_dtype)

            target = target_ids[bx]
            ng = group_count[bx]

            for ki in T.Parallel(BLOCK_K):
                if ki < K:
                    acc[ki] = grad_flat_edge_ebd[target, ki]
                else:
                    acc[ki] = 0.0

            for do in T.Pipelined(T.ceildiv(D, BLOCK_D), num_stages=2):
                for ki, di in T.Parallel(BLOCK_K, BLOCK_D):
                    d = do * BLOCK_D + di
                    if ki < K and d < D:
                        sub_s_ik[ki, di] = sub_edge_ik[ki, d]
                        sub_s_ij[ki, di] = sub_edge_ij[ki, d]
                    else:
                        sub_s_ik[ki, di] = 0.0
                        sub_s_ij[ki, di] = 0.0

                for gi, di in T.Parallel(max_group, BLOCK_D):
                    d = do * BLOCK_D + di
                    if gi < ng and d < D:
                        ik_e = eik_pos[bx, gi]
                        ij_e = eij_pos[bx, gi]
                        if ik_e >= 0:
                            grad_s_ik[gi, di] = grad_output[ik_e, d]
                        else:
                            grad_s_ik[gi, di] = 0.0

                        if ij_e >= 0:
                            grad_s_ij[gi, di] = grad_output[ij_e, d]
                        else:
                            grad_s_ij[gi, di] = 0.0
                    else:
                        grad_s_ik[gi, di] = 0.0
                        grad_s_ij[gi, di] = 0.0

                T.sync_threads()

                for ki in T.Parallel(BLOCK_K):
                    if ki < K:
                        for gi in T.serial(max_group):
                            if gi < ng:
                                for di in T.serial(BLOCK_D):
                                    d = do * BLOCK_D + di
                                    if d < D:
                                        acc[ki] += grad_s_ik[gi, di] * sub_s_ik[ki, di]

                                        acc[ki] += grad_s_ij[gi, di] * sub_s_ij[ki, di]

                T.sync_threads()

            for ki in T.Parallel(BLOCK_K):
                if ki < K:
                    grad_flat_edge_ebd[target, ki] = acc[ki]

    return edge_backward

@tilelang.jit
def fused_edge_ik_ij_backward_v2(
    G,
    N,
    E,
    K,
    D,
    max_group,
    BLOCK_M=32,
    BLOCK_K=32,
    BLOCK_D=32,
    threads=128,
    dtype="float32",
    accum_dtype="float32",
):
    @T.prim_func
    def edge_backward(
        grad_output: T.Tensor((E, D), dtype),
        sub_edge_ik: T.Tensor((K, D), dtype),
        sub_edge_ij: T.Tensor((K, D), dtype),
        target_ids: T.Tensor((G,), "int64"),
        eik_pos: T.Tensor((G, max_group), "int64"),
        eij_pos: T.Tensor((G, max_group), "int64"),
        group_count: T.Tensor((G,), "int64"),
        grad_flat_edge_ebd: T.Tensor((N, K), accum_dtype),
    ):
        with T.Kernel(G, threads=threads) as bx:
            grad_s_ik = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            grad_s_ij = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            sub_s_ik = T.alloc_shared((BLOCK_D, BLOCK_K), dtype)
            sub_s_ij = T.alloc_shared((BLOCK_D, BLOCK_K), dtype)

            acc_gemm = T.alloc_fragment((BLOCK_M, BLOCK_K), accum_dtype)
            acc_reduced = T.alloc_fragment((BLOCK_K,), accum_dtype)

            target = target_ids[bx]
            ng = group_count[bx]

            T.clear(acc_gemm)

            for do in T.Pipelined(T.ceildiv(D, BLOCK_D), num_stages=2):
                for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                    d = do * BLOCK_D + di
                    if d < D and ki < K:
                        sub_s_ik[di, ki] = sub_edge_ik[ki, d]
                        sub_s_ij[di, ki] = sub_edge_ij[ki, d]
                    else:
                        sub_s_ik[di, ki] = 0.0
                        sub_s_ij[di, ki] = 0.0

                for gi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    d = do * BLOCK_D + di
                    if gi < ng and d < D:
                        ik_e = eik_pos[bx, gi]
                        ij_e = eij_pos[bx, gi]
                        if ik_e >= 0:
                            grad_s_ik[gi, di] = grad_output[ik_e, d]
                        else:
                            grad_s_ik[gi, di] = 0.0

                        if ij_e >= 0:
                            grad_s_ij[gi, di] = grad_output[ij_e, d]
                        else:
                            grad_s_ij[gi, di] = 0.0
                    else:
                        grad_s_ik[gi, di] = 0.0
                        grad_s_ij[gi, di] = 0.0

                T.sync_threads()

                T.gemm(grad_s_ik, sub_s_ik, acc_gemm)
                T.gemm(grad_s_ij, sub_s_ij, acc_gemm)

                T.sync_threads()

            T.reduce_sum(acc_gemm, acc_reduced, dim=0)

            for ki in T.Parallel(BLOCK_K):
                if ki < K:
                    acc = T.alloc_var(accum_dtype)
                    acc = grad_flat_edge_ebd[target, ki]
                    acc += acc_reduced[ki]
                    grad_flat_edge_ebd[target, ki] = acc

    return edge_backward

@tilelang.jit
def fused_edge_ik_ij_backward_v3(
    G,
    N,
    E,
    K,
    D,
    max_group,
    BLOCK_M=32,
    BLOCK_K=32,
    BLOCK_D=32,
    threads=128,
    dtype="float32",
    accum_dtype="float32",
):
    @T.prim_func
    def edge_backward(
        grad_output: T.Tensor((E, D), dtype),
        sub_edge_ik: T.Tensor((K, D), dtype),
        sub_edge_ij: T.Tensor((K, D), dtype),
        target_ids: T.Tensor((G,), "int64"),
        eik_pos: T.Tensor((G, max_group), "int64"),
        eij_pos: T.Tensor((G, max_group), "int64"),
        group_count: T.Tensor((G,), "int64"),
        grad_flat_edge_ebd: T.Tensor((N, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(G, BLOCK_M), threads=threads) as bx:
            reduced_ik = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            reduced_ij = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            sub_s_ik = T.alloc_shared((BLOCK_D, BLOCK_K), dtype)
            sub_s_ij = T.alloc_shared((BLOCK_D, BLOCK_K), dtype)

            acc_ik = T.alloc_fragment((BLOCK_M, BLOCK_K), accum_dtype)
            acc_ij = T.alloc_fragment((BLOCK_M, BLOCK_K), accum_dtype)
            T.clear(acc_ik)
            T.clear(acc_ij)

            for do in T.Pipelined(T.ceildiv(D, BLOCK_D), num_stages=2):
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    g = bx * BLOCK_M + mi
                    d = do * BLOCK_D + di
                    value_ik = T.alloc_var(accum_dtype)
                    value_ij = T.alloc_var(accum_dtype)
                    value_ik = 0.0
                    value_ij = 0.0

                    if g < G and d < D:
                        ng = group_count[g]
                        for gi in T.serial(max_group):
                            if gi < ng:
                                ik_e = eik_pos[g, gi]
                                ij_e = eij_pos[g, gi]
                                if ik_e >= 0:
                                    value_ik += grad_output[ik_e, d]
                                if ij_e >= 0:
                                    value_ij += grad_output[ij_e, d]

                    reduced_ik[mi, di] = value_ik
                    reduced_ij[mi, di] = value_ij

                T.sync_threads()

                for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                    d = do * BLOCK_D + di
                    if d < D and ki < K:
                        sub_s_ik[di, ki] = sub_edge_ik[ki, d]
                        sub_s_ij[di, ki] = sub_edge_ij[ki, d]
                    else:
                        sub_s_ik[di, ki] = 0.0
                        sub_s_ij[di, ki] = 0.0

                T.sync_threads()

                T.gemm(reduced_ik, sub_s_ik, acc_ik)
                T.gemm(reduced_ij, sub_s_ij, acc_ij)

                T.sync_threads()

            for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                g = bx * BLOCK_M + mi
                if g < G and ki < K:
                    target = target_ids[g]
                    grad_flat_edge_ebd[target, ki] = acc_ik[mi, ki] + acc_ij[mi, ki]

    return edge_backward

@tilelang.jit
def fused_angle_sub_node_backward_v1(
    M,
    A,
    N,
    K,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_A=16,
    BLOCK_N=32,
    BLOCK_K=32,
    REDUCE_THREADS=8,
):
    assert BLOCK_A * REDUCE_THREADS == 128
    assert REDUCE_THREADS == 8

    @T.prim_func
    def sub_node_backward(
        flat_node_ebd: T.Buffer((A, N), dtype),
        grad_output: T.Buffer((M, K), dtype),
        node_start: T.Buffer((A,), "int32"),
        node_count: T.Buffer((A,), "int32"),
        grad_sub_node: T.Buffer((N, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(K, BLOCK_K), threads=128) as (bx, by):
            reduced_grad_shared = T.alloc_shared((BLOCK_A, BLOCK_K), dtype)
            node_shared = T.alloc_shared((BLOCK_N, BLOCK_A), dtype)
            
            output_local = T.alloc_fragment((BLOCK_N, BLOCK_K), accum_dtype)
            T.clear(output_local)

            tx = T.get_thread_binding(0)

            node_slot = tx // REDUCE_THREADS
            lane = tx % REDUCE_THREADS

            for kk in T.serial(BLOCK_K):
                k = by * BLOCK_K + kk
                acc = T.alloc_var(accum_dtype, init=0)
                if node_slot < A and k < K:
                    node = node_slot
                    start = node_start[node]
                    count = node_count[node]

                    for r in T.serial(16):
                        row_offset = lane + r * REDUCE_THREADS
                        if row_offset < count:
                            row = start + row_offset
                            if row < M:
                                acc += T.cast(grad_output[row, k], accum_dtype)

                acc += T.shfl_down(acc, 4, 8)
                acc += T.shfl_down(acc, 2, 8)
                acc += T.shfl_down(acc, 1, 8)

                if node_slot < A and lane == 0:
                    reduced_grad_shared[node_slot, kk] = T.cast(acc, dtype)

            T.sync_threads()

            for nn, aa in T.Parallel(BLOCK_N, BLOCK_A):
                n = bx * BLOCK_N + nn
                a = aa
                if n < N and a < A:
                    node_shared[nn, aa] = flat_node_ebd[a, n]
                else:
                    node_shared[nn, aa] = 0

            T.sync_threads()

            T.gemm(node_shared, reduced_grad_shared, output_local)

            T.sync_threads()

            for nn, kk in T.Parallel(BLOCK_N, BLOCK_K):
                n = bx * BLOCK_N + nn
                k = by * BLOCK_K + kk
                if n < N and k < K:
                    grad_sub_node[n, k] = output_local[nn, kk]

    return sub_node_backward

@tilelang.jit
def fused_edge_ik_ij_sub_backward_v1(
    G,
    N,
    E,
    K,
    D,
    max_group,
    BLOCK_M=32,
    BLOCK_K=32,
    BLOCK_D=32,
    threads=128,
    dtype="float32",
    accum_dtype="float32",
):
    @T.prim_func
    def edge_sub_backward(
        flat_edge_ebd: T.Tensor((N, K), dtype),
        grad_output: T.Tensor((E, D), dtype),
        target_ids: T.Tensor((G,), "int64"),
        eik_pos: T.Tensor((G, max_group), "int64"),
        eij_pos: T.Tensor((G, max_group), "int64"),
        group_count: T.Tensor((G,), "int64"),
        grad_sub_edge_ik: T.Tensor((K, D), accum_dtype),
        grad_sub_edge_ij: T.Tensor((K, D), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(K, BLOCK_K), T.ceildiv(D, BLOCK_D), threads=threads) as (bx, by):
            reduced_ik = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            reduced_ij = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            edge_shared = T.alloc_shared((BLOCK_K, BLOCK_M), dtype)

            acc_ik = T.alloc_fragment((BLOCK_K, BLOCK_D), accum_dtype)
            acc_ij = T.alloc_fragment((BLOCK_K, BLOCK_D), accum_dtype)
            T.clear(acc_ik)
            T.clear(acc_ij)

            for go in T.Pipelined(T.ceildiv(G, BLOCK_M), num_stages=2):
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    g = go * BLOCK_M + mi
                    d = by * BLOCK_D + di
                    value_ik = T.alloc_var(accum_dtype, init=0)
                    value_ij = T.alloc_var(accum_dtype, init=0)
                    if g < G and d < D:
                        ng = group_count[g]
                        for gi in T.serial(max_group):
                            if gi < ng:
                                ik_e = eik_pos[g, gi]
                                ij_e = eij_pos[g, gi]
                                if ik_e >= 0:
                                    value_ik += T.cast(grad_output[ik_e, d], accum_dtype)
                                if ij_e >= 0:
                                    value_ij += T.cast(grad_output[ij_e, d], accum_dtype)

                    reduced_ik[mi, di] = T.cast(value_ik, dtype)
                    reduced_ij[mi, di] = T.cast(value_ij, dtype)

                T.sync_threads()

                for ki, mi in T.Parallel(BLOCK_K, BLOCK_M):
                    k = bx * BLOCK_K + ki
                    g = go * BLOCK_M + mi
                    if g < G and k < K:
                        target = target_ids[g]
                        edge_shared[ki, mi] = flat_edge_ebd[target, k]
                    else:
                        edge_shared[ki, mi] = 0

                T.sync_threads()

                T.gemm(edge_shared, reduced_ik, acc_ik)
                T.gemm(edge_shared, reduced_ij, acc_ij)

                T.sync_threads()

            for ki, di in T.Parallel(BLOCK_K, BLOCK_D):
                k = bx * BLOCK_K + ki
                d = by * BLOCK_D + di
                if k < K and d < D:
                    grad_sub_edge_ik[k, d] = acc_ik[ki, di]
                    grad_sub_edge_ij[k, d] = acc_ij[ki, di]

    return edge_sub_backward

class FusedAngleUpdateFunctionBackward(torch.autograd.Function):
    _target_ids = None
    _eik_pos = None
    _eij_pos = None
    _group_count = None

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

        M, K = grad_output.shape
        A, N = flat_node_ebd.shape

        grad_flat_angle_ebd = torch.matmul(grad_output, sub_angle.transpose(0, 1))
        
        grad_flat_node_ebd = torch.zeros(flat_node_ebd.shape, device=grad_output.device, dtype=grad_output.dtype)
        # grad_gathered_node = torch.matmul(grad_output, sub_node.transpose(0, 1))
        # grad_flat_node_ebd.index_add_(0, n2a_index, grad_gathered_node)
        
        node_start = torch.tensor(
            [0, 121, 242, 363, 484, 605,
             726, 790, 854, 918, 982, 1046],
            dtype=torch.int32,
            device=grad_output.device,
        )

        node_count = torch.tensor(
            [121, 121, 121, 121, 121, 121,
             64, 64, 64, 64, 64, 64],
            dtype=torch.int32,
            device=grad_output.device,
        )
        node_backward = fused_angle_node_backward_v3_1(M=M, N=N, K=K, A=A)
        node_backward(grad_output, sub_node, node_start, node_count, grad_flat_node_ebd)

        grad_node_ebd = grad_flat_node_ebd.reshape(node_ebd_shape)
        
        grad_flat_edge_ebd = torch.zeros(flat_edge_ebd.shape, device=grad_output.device, dtype=grad_output.dtype)
        # grad_gathered_edge_ik = torch.matmul(grad_output, sub_edge_ik.transpose(0, 1))
        # grad_gathered_edge_ij = torch.matmul(grad_output, sub_edge_ij.transpose(0, 1))
        # grad_flat_edge_ebd.index_add_(0, eik2a_index, grad_gathered_edge_ik)
        # grad_flat_edge_ebd.index_add_(0, eij2a_index, grad_gathered_edge_ij)

        if FusedAngleUpdateFunctionBackward._target_ids is None:
            (
                FusedAngleUpdateFunctionBackward._target_ids,
                FusedAngleUpdateFunctionBackward._eik_pos,
                FusedAngleUpdateFunctionBackward._eij_pos,
                FusedAngleUpdateFunctionBackward._group_count,
            ) = build_edge_backward_metadata(
                eik2a_index,
                eij2a_index,
            )
        
        target_ids = FusedAngleUpdateFunctionBackward._target_ids
        eik_pos = FusedAngleUpdateFunctionBackward._eik_pos
        eij_pos = FusedAngleUpdateFunctionBackward._eij_pos
        group_count = FusedAngleUpdateFunctionBackward._group_count

        G = target_ids.numel()
        IK = sub_edge_ik.shape[0]
        N_EDGE, EK = flat_edge_ebd.shape
        max_group = eik_pos.shape[1]

        edge_backward = fused_edge_ik_ij_backward_v3(G=G, N=N_EDGE, E=M, K=IK, D=K, max_group=max_group)
        edge_backward(grad_output, sub_edge_ik, sub_edge_ij, target_ids, \
            eik_pos, eij_pos, group_count, grad_flat_edge_ebd)
    
        grad_sub_angle = torch.matmul(flat_angle_ebd.transpose(0, 1), grad_output)

        # gathered_node = torch.index_select(flat_node_ebd, 0, n2a_index)
        # grad_sub_node = torch.matmul(gathered_node.transpose(0, 1), grad_output)
        grad_sub_node = torch.zeros((N, K), device=grad_output.device, dtype=grad_output.dtype)
        sub_node_backward = fused_angle_sub_node_backward_v1(M=M, A=A, N=N, K=K)
        sub_node_backward(flat_node_ebd, grad_output, node_start, node_count, grad_sub_node)

        # gathered_edge_ik = torch.index_select(flat_edge_ebd, 0, eik2a_index)
        # grad_sub_edge_ik = torch.matmul(gathered_edge_ik.transpose(0, 1), grad_output)
        
        # gathered_edge_ij = torch.index_select(flat_edge_ebd, 0, eij2a_index)
        # grad_sub_edge_ij = torch.matmul(gathered_edge_ij.transpose(0, 1), grad_output)

        grad_sub_edge_ik = torch.zeros((EK, K), device=grad_output.device, dtype=grad_output.dtype)
        grad_sub_edge_ij = torch.zeros_like(grad_sub_edge_ik)
        edge_sub_backward = fused_edge_ik_ij_sub_backward_v1(G=G, N=N_EDGE, E=M, K=EK, D=K, max_group=max_group)
        edge_sub_backward(flat_edge_ebd, grad_output, target_ids, eik_pos, \
                eij_pos, group_count, grad_sub_edge_ik, grad_sub_edge_ij)

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
