import torch
import tilelang
import tilelang.language as T
import weakref
from collections import OrderedDict


# Small, bounded cache shared by layers that use views of the same graph index.
# Weak references prevent stale data-pointer reuse; tensor versions invalidate
# metadata after normal in-place PyTorch updates. Graph indices are immutable
# during a forward/backward. Mutations through .data/external pointers are not
# supported (as with autograd's own saved-tensor version checks).
_sym_owner_cache = OrderedDict()
_SYM_OWNER_CACHE_SIZE = 8


def _sym_owner_metadata(owner, num_owner):
    """Return (uniform, offsets, edge_order) for an arbitrary owner vector.

    Uniform means exactly [0]*R + [1]*R + ... . Only metadata creation
    synchronizes with the CPU; cache hits do not. Offsets include empty owners.
    """
    if num_owner <= 0 or owner.ndim != 1:
        raise ValueError("owner must be one-dimensional and num_owner positive")
    if owner.dtype not in (torch.int32, torch.int64):
        raise TypeError("owner must use int32 or int64 indices")
    base = owner
    while base._base is not None:
        base = base._base
    key = (id(base), owner.data_ptr(), owner.numel(), owner.stride(),
           owner._version, num_owner, owner.device)
    cached = _sym_owner_cache.get(key)
    if cached is not None and cached[0]() is base:
        _sym_owner_cache.move_to_end(key)
        return cached[1]
    if owner.numel() and bool(((owner < 0) | (owner >= num_owner)).any()):
        raise ValueError("owner indices must be in [0, num_owner)")
    count = owner.numel()
    uniform = False
    if count and count % num_owner == 0:
        expected = torch.arange(num_owner, device=owner.device, dtype=owner.dtype)
        uniform = torch.equal(owner, expected.repeat_interleave(count // num_owner))
    if uniform:
        result = (True, None, None)
    else:
        counts = torch.bincount(owner.long(), minlength=num_owner)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        sorted_owner = count < 2 or bool((owner[1:] >= owner[:-1]).all())
        order = (torch.arange(count, device=owner.device, dtype=torch.int64)
                 if sorted_owner else torch.argsort(owner, stable=True))
        result = (False, offsets, order)
    _sym_owner_cache[key] = (weakref.ref(base), result)
    _sym_owner_cache.move_to_end(key)
    while len(_sym_owner_cache) > _SYM_OWNER_CACHE_SIZE:
        _sym_owner_cache.popitem(last=False)
    return result


@tilelang.jit
def fused_cal_hg_dynamic_forward_segmented(M, E, NO, BLOCK_N=64):
    @T.prim_func
    def kernel(
        edge: T.Tensor((M, E), "float32"),
        sw: T.Tensor((M,), "float32"),
        h: T.Tensor((M, 3), "float32"),
        offsets: T.Tensor((NO + 1,), "int64"),
        order: T.Tensor((M,), "int64"),
        scale: T.float32,
        out: T.Tensor((NO, 3 * E), "float32"),
    ):
        with T.Kernel(NO, T.ceildiv(3 * E, BLOCK_N), threads=64) as (o, tile):
            for j in T.Parallel(BLOCK_N):
                col = tile * BLOCK_N + j
                if col < 3 * E:
                    acc = T.alloc_var("float32", init=0)
                    for r in T.serial(offsets[o], offsets[o + 1]):
                        m = order[r]
                        acc += edge[m, col % E] * sw[m] * h[m, col // E]
                    out[o, col] = acc * scale
    return kernel

@tilelang.jit
def fused_cal_hg_dynamic_forward_v0(
    M,              # n_edge
    N,              # 3 * e_dim
    E,              # e_dim
    NO,             # num_owner
    dtype="float32",
    accum_dtype="float32",
    BLOCK_N: int = 64,
):
    assert M % NO == 0
    EDGES_PER_OWNER = M // NO

    @T.prim_func
    def hg_kernel(
        flat_edge_ebd: T.Tensor((M, E), dtype),
        flat_sw: T.Tensor((M,), dtype),
        flat_h2: T.Tensor((M, 3), dtype),
        scale_factor: T.float32,
        out: T.Tensor((NO, N), accum_dtype),
    ):
        with T.Kernel(
            NO,
            T.ceildiv(N, BLOCK_N),
            threads=64,
        ) as (bx, by):

            acc = T.alloc_fragment(
                (BLOCK_N,),
                accum_dtype,
            )
            T.clear(acc)

            # ---------------------------------------------------------
            # 纯融合版本：
            #
            # 1. 不使用 shared memory
            # 2. 不展开 EDGES_PER_OWNER
            # 3. 直接从 global memory 读取 sw / h2 / edge_ebd
            # 4. 使用普通 serial reduction
            # ---------------------------------------------------------
            for j in T.Parallel(BLOCK_N):
                col = by * BLOCK_N + j

                if col < N:
                    h2_idx = col // E
                    e_idx = col % E

                    for r in T.serial(EDGES_PER_OWNER):
                        edge_idx = bx * EDGES_PER_OWNER + r

                        acc[j] += (
                            flat_edge_ebd[edge_idx, e_idx]
                            * flat_sw[edge_idx]
                            * flat_h2[edge_idx, h2_idx]
                        )

                    acc[j] = acc[j] * scale_factor

            # 写回
            for j in T.Parallel(BLOCK_N):
                col = by * BLOCK_N + j

                if col < N:
                    out[bx, col] = acc[j]

    return hg_kernel

@tilelang.jit
def fused_cal_hg_dynamic_forward_v1(
    M,              # n_edge
    N,              # 3 * e_dim
    E,              # e_dim
    NO,             # num_owner
    dtype="float32",
    accum_dtype="float32",
    BLOCK_N: int = 64,
):
    assert M % NO == 0
    EDGES_PER_OWNER = M // NO

    @T.prim_func
    def hg_kernel(
        flat_edge_ebd: T.Tensor((M, E), dtype),
        flat_sw: T.Tensor((M,), dtype),
        flat_h2: T.Tensor((M, 3), dtype),
        scale_factor: T.float32,
        out: T.Tensor((NO, N), accum_dtype),
    ):
        with T.Kernel(
            NO,
            T.ceildiv(N, BLOCK_N),
            threads=64,
        ) as (bx, by):

            # ---------------------------------------------------------
            # Optimization 1:
            # sw + h2 合并加载到 shared memory
            # ---------------------------------------------------------
            meta_shared = T.alloc_shared(
                (EDGES_PER_OWNER, 4),
                dtype,
            )

            # sw
            for r in T.Parallel(EDGES_PER_OWNER):
                edge_idx = bx * EDGES_PER_OWNER + r
                meta_shared[r, 0] = flat_sw[edge_idx]

            # h2
            for r, k in T.Parallel(EDGES_PER_OWNER, 3):
                edge_idx = bx * EDGES_PER_OWNER + r
                meta_shared[r, k + 1] = flat_h2[edge_idx, k]

            # 必须保证 shared memory 写入完成
            T.sync_threads()

            acc = T.alloc_fragment(
                (BLOCK_N,),
                accum_dtype,
            )
            T.clear(acc)

            # ---------------------------------------------------------
            # Optimization 1 ONLY:
            # 使用 shared memory 中的 sw / h2
            #
            # 这里故意不做 v2 的手工展开
            # ---------------------------------------------------------
            for j in T.Parallel(BLOCK_N):
                col = by * BLOCK_N + j

                if col < N:
                    h2_idx = col // E
                    e_idx = col % E

                    for r in T.serial(EDGES_PER_OWNER):
                        edge_idx = bx * EDGES_PER_OWNER + r

                        acc[j] += (
                            flat_edge_ebd[edge_idx, e_idx]
                            * meta_shared[r, 0]
                            * meta_shared[r, h2_idx + 1]
                        )

                    acc[j] = acc[j] * scale_factor

            for j in T.Parallel(BLOCK_N):
                col = by * BLOCK_N + j

                if col < N:
                    out[bx, col] = acc[j]

    return hg_kernel

@tilelang.jit
def fused_cal_hg_dynamic_forward_v2(
    M,              # n_edge
    N,              # 3 * e_dim
    E,              # e_dim
    NO,             # num_owner
    dtype="float32",
    accum_dtype="float32",
    BLOCK_N: int = 64,
):
    assert M % NO == 0
    EDGES_PER_OWNER = M // NO

    @T.prim_func
    def hg_kernel(
        flat_edge_ebd: T.Tensor((M, E), dtype),
        flat_sw: T.Tensor((M,), dtype),
        flat_h2: T.Tensor((M, 3), dtype),
        scale_factor: T.float32,
        out: T.Tensor((NO, N), accum_dtype),
    ):
        with T.Kernel(
            NO,
            T.ceildiv(N, BLOCK_N),
            threads=64,
        ) as (bx, by):

            acc = T.alloc_fragment(
                (BLOCK_N,),
                accum_dtype,
            )
            T.clear(acc)

            # ---------------------------------------------------------
            # Optimization 2 ONLY:
            # 不使用 shared memory
            # 直接从 global memory 读取 sw / h2
            #
            # 但是将 reduction / index_add 逻辑手工展开
            # ---------------------------------------------------------
            for j in T.Parallel(BLOCK_N):
                col = by * BLOCK_N + j

                if col < N:
                    h2_idx = col // E
                    e_idx = col % E

                    edge_base = bx * EDGES_PER_OWNER

                    v0 = (
                        flat_edge_ebd[edge_base + 0, e_idx]
                        * flat_sw[edge_base + 0]
                        * flat_h2[edge_base + 0, h2_idx]
                    )

                    v1 = (
                        flat_edge_ebd[edge_base + 1, e_idx]
                        * flat_sw[edge_base + 1]
                        * flat_h2[edge_base + 1, h2_idx]
                    )

                    v2 = (
                        flat_edge_ebd[edge_base + 2, e_idx]
                        * flat_sw[edge_base + 2]
                        * flat_h2[edge_base + 2, h2_idx]
                    )

                    v3 = (
                        flat_edge_ebd[edge_base + 3, e_idx]
                        * flat_sw[edge_base + 3]
                        * flat_h2[edge_base + 3, h2_idx]
                    )

                    v4 = (
                        flat_edge_ebd[edge_base + 4, e_idx]
                        * flat_sw[edge_base + 4]
                        * flat_h2[edge_base + 4, h2_idx]
                    )

                    v5 = (
                        flat_edge_ebd[edge_base + 5, e_idx]
                        * flat_sw[edge_base + 5]
                        * flat_h2[edge_base + 5, h2_idx]
                    )

                    v6 = (
                        flat_edge_ebd[edge_base + 6, e_idx]
                        * flat_sw[edge_base + 6]
                        * flat_h2[edge_base + 6, h2_idx]
                    )

                    v7 = (
                        flat_edge_ebd[edge_base + 7, e_idx]
                        * flat_sw[edge_base + 7]
                        * flat_h2[edge_base + 7, h2_idx]
                    )

                    v8 = (
                        flat_edge_ebd[edge_base + 8, e_idx]
                        * flat_sw[edge_base + 8]
                        * flat_h2[edge_base + 8, h2_idx]
                    )

                    v9 = (
                        flat_edge_ebd[edge_base + 9, e_idx]
                        * flat_sw[edge_base + 9]
                        * flat_h2[edge_base + 9, h2_idx]
                    )

                    v10 = (
                        flat_edge_ebd[edge_base + 10, e_idx]
                        * flat_sw[edge_base + 10]
                        * flat_h2[edge_base + 10, h2_idx]
                    )

                    acc[j] = (
                        v0 + v1 + v2 + v3 + v4
                        + v5 + v6 + v7 + v8 + v9 + v10
                    ) * scale_factor

            for j in T.Parallel(BLOCK_N):
                col = by * BLOCK_N + j

                if col < N:
                    out[bx, col] = acc[j]

    return hg_kernel

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
        flat_edge_ebd: T.Tensor((M, E), dtype),
        flat_sw: T.Tensor((M,), dtype),
        flat_h2: T.Tensor((M, 3), dtype),
        scale_factor: T.float32,
        out: T.Tensor((NO, N), accum_dtype),
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
            
            T.sync_threads()

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
def fused_call_grrg_forward_v0(
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
        h2g2: T.Tensor((NB, NLOC, 3, E), dtype),
        out: T.Tensor((NB, NLOC, AXIS * E), accum_dtype),
    ):
        NUM_TILE_M = T.ceildiv(AXIS, BLOCK_M)
        NUM_TILE_N = T.ceildiv(E, BLOCK_N)

        with T.Kernel(
            NB,
            NLOC,
            NUM_TILE_M * NUM_TILE_N,
            threads=128,
        ) as (bx, by, bz):

            tile_m = bz // NUM_TILE_N
            tile_n = bz % NUM_TILE_N

            for a, e in T.Parallel(BLOCK_M, BLOCK_N):
                axis_idx = tile_m * BLOCK_M + a
                e_idx = tile_n * BLOCK_N + e

                if axis_idx < AXIS and e_idx < E:
                    acc = T.alloc_var(accum_dtype, 0)

                    for k in T.serial(3):
                        acc += (
                            h2g2[bx, by, k, axis_idx]
                            * h2g2[bx, by, k, e_idx]
                        )

                    acc = acc / (3.0**1)

                    out[
                        bx,
                        by,
                        axis_idx * E + e_idx,
                    ] = acc

    return grrg_kernel

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
        h2g2: T.Tensor((NB, NLOC, 3, E), dtype),
        out: T.Tensor((NB, NLOC, AXIS * E), accum_dtype),
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

        # Original uniform-only implementation:
        # hg_kernel = fused_cal_hg_dynamic_forward_v1(M=n_edge, N=3*e_dim, E=e_dim, NO=num_owner)
        # hg_kernel(flat_edge_ebd, flat_sw, flat_h2, scale_factor, h2g2)
        uniform, offsets, order = _sym_owner_metadata(owner, num_owner)
        if n_edge == 0:
            h2g2.zero_()
        elif uniform:
            hg_kernel = fused_cal_hg_dynamic_forward_v1(M=n_edge, N=3*e_dim, E=e_dim, NO=num_owner)
            hg_kernel(flat_edge_ebd, flat_sw, flat_h2, scale_factor, h2g2)
        else:
            hg_kernel = fused_cal_hg_dynamic_forward_segmented(M=n_edge, E=e_dim, NO=num_owner)
            hg_kernel(flat_edge_ebd, flat_sw, flat_h2, offsets, order, scale_factor, h2g2)
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

        grrg_kernel = fused_call_grrg_forward_v0(NB=nb, NLOC=nloc, E=e_dim, AXIS=axis_neuron)
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
        grad_h2g2: T.Tensor((NB, NLOC, 3, E), dtype),
        flat_edge_ebd: T.Tensor((M, E), dtype),
        flat_h2: T.Tensor((M, 3), dtype),
        flat_sw: T.Tensor((M,), dtype),
        owner: T.Tensor((M,), "int64"),
        grad_flat_h2g2: T.Tensor((M, 3, E), accum_dtype),
        grad_flat_edge_ebd: T.Tensor((M, E), accum_dtype),
        grad_h2: T.Tensor((M, 3), accum_dtype),
        grad_flat_sw: T.Tensor((M,), accum_dtype),
    ):
        with T.Kernel(M, threads=BLOCK_D) as (bx,):
            sh_acc_h0 = T.alloc_shared((BLOCK_D,), T.float32)
            sh_acc_h1 = T.alloc_shared((BLOCK_D,), T.float32)
            sh_acc_h2 = T.alloc_shared((BLOCK_D,), T.float32)
            sh_acc_sw = T.alloc_shared((BLOCK_D,), T.float32)

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
        grad_grrg: T.Tensor((NB, NLOC, A * E), dtype),
        h2g2: T.Tensor((NB, NLOC, 3, E), dtype),
        grad_h2g2: T.Tensor((NB, NLOC, 3, E), dtype),
        scale_factor: T.float32,
    ):
        with T.Kernel(NB * NLOC, threads=THREADS) as (bx,):
            nb_idx = bx // NLOC
            nloc_idx = bx % NLOC

            sh_G = T.alloc_shared((A, E), dtype)
            sh_H = T.alloc_shared((3, E), dtype)

            sh_right_tmp = T.alloc_shared((3, E, A), dtype)
            sh_right = T.alloc_shared((3, E), dtype)

            sh_left_tmp = T.alloc_shared((3, A, E), dtype)
            sh_left = T.alloc_shared((3, A), dtype)

            for a, e in T.Parallel(A, E):
                sh_G[a, e] = grad_grrg[nb_idx, nloc_idx, a * E + e]

            for b, e in T.Parallel(3, E):
                sh_H[b, e] = h2g2[nb_idx, nloc_idx, b, e]

            T.sync_threads()

            for b, e, a in T.Parallel(3, E, A):
                sh_right_tmp[b, e, a] = sh_H[b, a] * sh_G[a, e]

            for b, a, k in T.Parallel(3, A, E):
                sh_left_tmp[b, a, k] = sh_H[b, k] * sh_G[a, k]

            T.sync_threads()

            T.reduce_sum(sh_right_tmp, sh_right, dim=2)
            T.reduce_sum(sh_left_tmp, sh_left, dim=2)

            T.sync_threads()

            scale = scale_factor / 3.0

            for b, e in T.Parallel(3, E):
                if e < A:
                    grad_h2g2[nb_idx, nloc_idx, b, e] = (sh_right[b, e] + sh_left[b, e]) * scale
                else:
                    grad_h2g2[nb_idx, nloc_idx, b, e] = sh_right[b, e] * scale

    return grrg_backward


@tilelang.jit
def fused_call_grrg_backward_general(NB, NLOC, E, A, dtype="float32", THREADS=128):
    """Gram VJP without cross-thread reductions for arbitrary feature sizes.

    R[b,d] = scale/3 * (sum_a H[b,a] G[a,d]
                        + (d < A) sum_k H[b,k] G[d,k]).
    Each output has one writer, avoiding the XOR all-reduce width constraint.
    """
    @T.prim_func
    def kernel(
        grad_grrg: T.Tensor((NB, NLOC, A * E), dtype),
        h2g2: T.Tensor((NB, NLOC, 3, E), dtype),
        grad_h2g2: T.Tensor((NB, NLOC, 3, E), dtype),
        scale_factor: T.float32,
    ):
        with T.Kernel(NB * NLOC, threads=THREADS) as (owner_idx,):
            batch = owner_idx // NLOC
            loc = owner_idx % NLOC
            for b, d in T.Parallel(3, E):
                acc = T.alloc_var(dtype, init=0)
                for a in T.serial(A):
                    acc += h2g2[batch, loc, b, a] * grad_grrg[batch, loc, a * E + d]
                if d < A:
                    for k in T.serial(E):
                        acc += h2g2[batch, loc, b, k] * grad_grrg[batch, loc, d * E + k]
                grad_h2g2[batch, loc, b, d] = acc * (scale_factor / 3.0)
    return kernel


@tilelang.jit
def fused_symmetrization_double_backward_owner(
    M,
    E,
    NO,
    A,
    dtype="float32",
    accum_dtype="float32",
    THREADS=128,
):
    """Fuse the owner reduction and the two Gram-matrix VJPs."""
    assert M % NO == 0
    EDGES_PER_OWNER = M // NO

    @T.prim_func
    def double_backward_owner(
        grad_grrg: T.Tensor((1, NO, A * E), dtype),
        h2g2: T.Tensor((1, NO, 3, E), dtype),
        flat_edge_ebd: T.Tensor((M, E), dtype),
        flat_h2: T.Tensor((M, 3), dtype),
        flat_sw: T.Tensor((M,), dtype),
        grad_grad_edge: T.Tensor((M, E), dtype),
        grad_grad_h2: T.Tensor((M, 3), dtype),
        grad_grad_sw: T.Tensor((M,), dtype),
        scale_factor: T.float32,
        grad_grad_grrg: T.Tensor((1, NO, A * E), accum_dtype),
        grad_h2g2: T.Tensor((1, NO, 3, E), accum_dtype),
    ):
        with T.Kernel(NO, threads=THREADS) as (owner_idx,):
            grad_q = T.alloc_shared((3, E), accum_dtype)

            # d(phi) / dR, reduced over all edges owned by this atom.
            for b, d in T.Parallel(3, E):
                acc = T.alloc_var(accum_dtype, init=0)
                for r in T.serial(EDGES_PER_OWNER):
                    edge_idx = owner_idx * EDGES_PER_OWNER + r
                    edge = flat_edge_ebd[edge_idx, d]
                    h = flat_h2[edge_idx, b]
                    sw = flat_sw[edge_idx]
                    acc += (
                        grad_grad_h2[edge_idx, b] * edge * sw
                        + grad_grad_edge[edge_idx, d] * h * sw
                        + grad_grad_sw[edge_idx] * h * edge
                    )
                grad_q[b, d] = acc

            T.sync_threads()

            c = scale_factor / 3.0

            # Gradient with respect to the incoming first-order gradient G.
            for a, d in T.Parallel(A, E):
                acc = T.alloc_var(accum_dtype, init=0)
                for b in T.serial(3):
                    acc += (
                        h2g2[0, owner_idx, b, a] * grad_q[b, d]
                        + grad_q[b, a] * h2g2[0, owner_idx, b, d]
                    )
                grad_grad_grrg[0, owner_idx, a * E + d] = acc * c

            # Gradient with respect to H.  The next fused kernel propagates
            # this through H = scale * aggregate(h2 * edge * sw).
            for b, d in T.Parallel(3, E):
                acc = T.alloc_var(accum_dtype, init=0)
                for a in T.serial(A):
                    acc += grad_q[b, a] * grad_grrg[0, owner_idx, a * E + d]
                if d < A:
                    for k in T.serial(E):
                        acc += grad_q[b, k] * grad_grrg[0, owner_idx, d * E + k]
                grad_h2g2[0, owner_idx, b, d] = acc * c

    return double_backward_owner


@tilelang.jit
def fused_symmetrization_double_backward_edge(
    M,
    E,
    NO,
    dtype="float32",
    accum_dtype="float32",
    THREADS=128,
):
    """Fuse all direct and H-mediated second derivatives per edge."""

    @T.prim_func
    def double_backward_edge(
        grad_flat_h2g2: T.Tensor((M, 3, E), dtype),
        grad_h2g2: T.Tensor((1, NO, 3, E), dtype),
        flat_edge_ebd: T.Tensor((M, E), dtype),
        flat_h2: T.Tensor((M, 3), dtype),
        flat_sw: T.Tensor((M,), dtype),
        owner: T.Tensor((M,), "int64"),
        grad_grad_edge: T.Tensor((M, E), dtype),
        grad_grad_h2: T.Tensor((M, 3), dtype),
        grad_grad_sw: T.Tensor((M,), dtype),
        scale_factor: T.float32,
        grad_flat_edge_ebd: T.Tensor((M, E), accum_dtype),
        grad_flat_h2: T.Tensor((M, 3), accum_dtype),
        grad_flat_sw: T.Tensor((M,), accum_dtype),
    ):
        with T.Kernel(M, threads=THREADS) as (edge_idx,):
            owner_idx = owner[edge_idx]
            sw = flat_sw[edge_idx]

            partial_h = T.alloc_shared((3, E), accum_dtype)
            partial_sw = T.alloc_shared((3, E), accum_dtype)

            for b, d in T.Parallel(3, E):
                r = grad_flat_h2g2[edge_idx, b, d]
                gh = grad_h2g2[0, owner_idx, b, d] * scale_factor
                edge = flat_edge_ebd[edge_idx, d]
                h = flat_h2[edge_idx, b]

                partial_h[b, d] = (
                    grad_grad_edge[edge_idx, d] * sw * r
                    + grad_grad_sw[edge_idx] * edge * r
                    + gh * edge * sw
                )
                partial_sw[b, d] = (
                    grad_grad_h2[edge_idx, b] * edge * r
                    + grad_grad_edge[edge_idx, d] * h * r
                    + gh * h * edge
                )

            for d in T.Parallel(E):
                acc = T.alloc_var(accum_dtype, init=0)
                for b in T.serial(3):
                    r = grad_flat_h2g2[edge_idx, b, d]
                    gh = grad_h2g2[0, owner_idx, b, d] * scale_factor
                    acc += (
                        grad_grad_h2[edge_idx, b] * r * sw
                        + grad_grad_sw[edge_idx] * r * flat_h2[edge_idx, b]
                        + gh * flat_h2[edge_idx, b] * sw
                    )
                grad_flat_edge_ebd[edge_idx, d] = acc

            T.sync_threads()

            for b in T.Parallel(3):
                acc = T.alloc_var(accum_dtype, init=0)
                for d in T.serial(E):
                    acc += partial_h[b, d]
                grad_flat_h2[edge_idx, b] = acc

            if T.get_thread_binding() == 0:
                acc = T.alloc_var(accum_dtype, init=0)
                for b in T.serial(3):
                    for d in T.serial(E):
                        acc += partial_sw[b, d]
                grad_flat_sw[edge_idx] = acc

    return double_backward_edge

@tilelang.jit
def fused_symmetrization_double_backward_owner_segmented(
    M,
    E,
    NO,
    A,
    dtype="float32",
    accum_dtype="float32",
    THREADS=128,
):
    """Fuse the owner reduction and the two Gram-matrix VJPs."""
    # Original uniform-only bounds are retained in the original kernel above.

    @T.prim_func
    def double_backward_owner(
        grad_grrg: T.Tensor((1, NO, A * E), dtype),
        h2g2: T.Tensor((1, NO, 3, E), dtype),
        flat_edge_ebd: T.Tensor((M, E), dtype),
        flat_h2: T.Tensor((M, 3), dtype),
        flat_sw: T.Tensor((M,), dtype),
        grad_grad_edge: T.Tensor((M, E), dtype),
        grad_grad_h2: T.Tensor((M, 3), dtype),
        grad_grad_sw: T.Tensor((M,), dtype),
        offsets: T.Tensor((NO + 1,), "int64"),
        order: T.Tensor((M,), "int64"),
        scale_factor: T.float32,
        grad_grad_grrg: T.Tensor((1, NO, A * E), accum_dtype),
        grad_h2g2: T.Tensor((1, NO, 3, E), accum_dtype),
    ):
        with T.Kernel(NO, threads=THREADS) as (owner_idx,):
            grad_q = T.alloc_shared((3, E), accum_dtype)

            # d(phi) / dR, reduced over all edges owned by this atom.
            for b, d in T.Parallel(3, E):
                acc = T.alloc_var(accum_dtype, init=0)
                # Original: for r in T.serial(EDGES_PER_OWNER)
                # Original: edge_idx = owner_idx * EDGES_PER_OWNER + r
                for r in T.serial(offsets[owner_idx], offsets[owner_idx + 1]):
                    edge_idx = order[r]
                    edge = flat_edge_ebd[edge_idx, d]
                    h = flat_h2[edge_idx, b]
                    sw = flat_sw[edge_idx]
                    acc += (
                        grad_grad_h2[edge_idx, b] * edge * sw
                        + grad_grad_edge[edge_idx, d] * h * sw
                        + grad_grad_sw[edge_idx] * h * edge
                    )
                grad_q[b, d] = acc

            T.sync_threads()

            c = scale_factor / 3.0

            # Gradient with respect to the incoming first-order gradient G.
            for a, d in T.Parallel(A, E):
                acc = T.alloc_var(accum_dtype, init=0)
                for b in T.serial(3):
                    acc += (
                        h2g2[0, owner_idx, b, a] * grad_q[b, d]
                        + grad_q[b, a] * h2g2[0, owner_idx, b, d]
                    )
                grad_grad_grrg[0, owner_idx, a * E + d] = acc * c

            # Gradient with respect to H.  The next fused kernel propagates
            # this through H = scale * aggregate(h2 * edge * sw).
            for b, d in T.Parallel(3, E):
                acc = T.alloc_var(accum_dtype, init=0)
                for a in T.serial(A):
                    acc += grad_q[b, a] * grad_grrg[0, owner_idx, a * E + d]
                if d < A:
                    for k in T.serial(E):
                        acc += grad_q[b, k] * grad_grrg[0, owner_idx, d * E + k]
                grad_h2g2[0, owner_idx, b, d] = acc * c

    return double_backward_owner


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
        # Original unconditional dispatch:
        # grrg_backward_kernel = fused_call_grrg_backward(NB=nb, NLOC=nloc, E=E, A=axis_neuron)
        # Non-power-of-two reduction extents can produce an unsupported
        # XOR-butterfly width in TileLang (e.g. E=5, A=3).
        regular_extents = (E > 0 and (E & (E - 1)) == 0
                           and axis_neuron > 0 and (axis_neuron & (axis_neuron - 1)) == 0)
        grrg_factory = (fused_call_grrg_backward if regular_extents
                        else fused_call_grrg_backward_general)
        grrg_backward_kernel = grrg_factory(NB=nb, NLOC=nloc, E=E, A=axis_neuron)
        grrg_backward_kernel(grad_grrg, h2g2, grad_h2g2, float(scale_factor))

        # grad_g1 = grad_grrg.reshape(nb, nloc, axis_neuron, E)
        # h2g2m = h2g2[..., :axis_neuron]

        # grad_left = torch.matmul(grad_g1, h2g2.transpose(-1, -2))
        # grad_left = grad_left.transpose(-1, -2)

        # grad_right = torch.matmul(h2g2m, grad_g1)
        
        # grad_h2g2 = grad_right.clone()
        # grad_h2g2[..., :axis_neuron] += grad_left
        # grad_h2g2 /= 3.0
        # grad_h2g2 *= scale_factor

        # Resolve once, retaining the metadata even if the bounded cache evicts it.
        ctx.owner_metadata = _sym_owner_metadata(owner, num_owner)
        owner = owner.long()
        grad_flat_h2g2 = torch.empty(
            (M, 3, E),
            dtype=grad_h2g2.dtype,
            device=grad_h2g2.device,
        )
        grad_flat_edge_ebd = torch.empty_like(flat_edge_ebd)
        grad_h2 = torch.empty_like(flat_h2)
        grad_flat_sw = torch.empty_like(flat_sw)

        # hg_backward_kernel = fused_call_hg_dynamic_backward(M=M, E=E, NB=nb, NLOC=nloc)
        # hg_backward_kernel(grad_h2g2, flat_edge_ebd, flat_h2, flat_sw,
        #     owner, grad_flat_h2g2, grad_flat_edge_ebd, grad_h2, grad_flat_sw)
        if M:
            hg_backward_kernel = fused_call_hg_dynamic_backward(M=M, E=E, NB=nb, NLOC=nloc)
            hg_backward_kernel(grad_h2g2, flat_edge_ebd, flat_h2, flat_sw,
                owner, grad_flat_h2g2, grad_flat_edge_ebd, grad_h2, grad_flat_sw)

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

        # nb = ctx.nb  # Used by the original unfused implementation below.
        # nloc = ctx.nloc
        num_owner = ctx.num_owner
        scale_factor = ctx.scale_factor
        axis_neuron = ctx.axis_neuron

        """
        Original unfused second-order backward implementation.  Keep this
        reference next to the fused path so formula changes remain auditable.
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
        """

        M, E = flat_edge_ebd.shape
        dtype = str(grad_grrg.dtype).replace("torch.", "")

        # Autograd may omit any unused second-order cotangent.  Materializing
        # it as zero keeps a single compiled kernel signature for every case.
        if grad_grad_edge is None:
            grad_grad_edge = torch.zeros_like(flat_edge_ebd)
        else:
            grad_grad_edge = grad_grad_edge.contiguous()
        if grad_grad_h2 is None:
            grad_grad_h2 = torch.zeros_like(flat_h2)
        else:
            grad_grad_h2 = grad_grad_h2.contiguous()
        if grad_grad_sw is None:
            grad_grad_sw = torch.zeros_like(flat_sw)
        else:
            grad_grad_sw = grad_grad_sw.contiguous()

        grad_grad_grrg = torch.empty_like(grad_grrg)
        grad_h2g2 = torch.empty_like(h2g2)
        # Original uniform-only dispatch (kernel itself is preserved above):
        # owner_kernel = fused_symmetrization_double_backward_owner(
        #     M=M, E=E, NO=num_owner, A=axis_neuron, dtype=dtype)
        # owner_kernel(grad_grrg, h2g2, flat_edge_ebd, flat_h2, flat_sw,
        #     grad_grad_edge, grad_grad_h2, grad_grad_sw, float(scale_factor),
        #     grad_grad_grrg, grad_h2g2)
        if M == 0:
            # No edges: all first and second derivatives are zero. Avoid
            # launching zero-grid CUDA kernels or compiling zero-sized buffers.
            return (torch.zeros_like(grad_grrg), torch.zeros_like(flat_edge_ebd),
                    torch.zeros_like(flat_h2), torch.zeros_like(flat_sw),
                    None, None, None, None, None, None, None)
        uniform, offsets, order = ctx.owner_metadata
        factory = (fused_symmetrization_double_backward_owner if uniform
                   else fused_symmetrization_double_backward_owner_segmented)
        owner_kernel = factory(M=M, E=E, NO=num_owner, A=axis_neuron, dtype=dtype)
        metadata_args = () if uniform else (offsets, order)
        owner_kernel(
            grad_grrg, h2g2, flat_edge_ebd, flat_h2, flat_sw,
            grad_grad_edge, grad_grad_h2, grad_grad_sw,
            *metadata_args, float(scale_factor), grad_grad_grrg, grad_h2g2)

        grad_flat_edge_ebd = torch.empty_like(flat_edge_ebd)
        grad_flat_h2 = torch.empty_like(flat_h2)
        grad_flat_sw = torch.empty_like(flat_sw)
        edge_kernel = fused_symmetrization_double_backward_edge(
            M=M,
            E=E,
            NO=num_owner,
            dtype=dtype,
        )
        edge_kernel(
            grad_flat_h2g2,
            grad_h2g2,
            flat_edge_ebd,
            flat_h2,
            flat_sw,
            owner,
            grad_grad_edge,
            grad_grad_h2,
            grad_grad_sw,
            float(scale_factor),
            grad_flat_edge_ebd,
            grad_flat_h2,
            grad_flat_sw,
        )

        return (
            grad_grad_grrg,       # 0 grad_grrg
            grad_flat_edge_ebd,   # 1 flat_edge_ebd
            grad_flat_h2,         # 2 flat_h2
            grad_flat_sw,         # 3 flat_sw
            None,                 # 4 owner
            None,                 # 5 h2g2 (propagated explicitly above)
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
    BLK_M: int = 64,
    BLK_N: int = 64,
    BLK_K: int = 64,
):
    @T.prim_func
    def edge_forward(
        node_ebd: T.Tensor((N_NODES_LOC, NODE_DIM), "float32"),
        node_ebd_ext: T.Tensor((N_NODES_EXT, NODE_DIM), "float32"),
        flat_edge_ebd: T.Tensor((N_EDGES, EDGE_DIM), "float32"),
        n2e_index: T.Tensor((N_EDGES,), "int64"),
        n_ext2e_index: T.Tensor((N_EDGES,), "int64"),
        node: T.Tensor((NODE_DIM, OUT_DIM), "float32"),
        node_ext: T.Tensor((NODE_DIM, OUT_DIM), "float32"),
        edge: T.Tensor((EDGE_DIM, OUT_DIM), "float32"),
        bias: T.Tensor((OUT_DIM,), "float32"),
        out: T.Tensor((N_EDGES, OUT_DIM), "float32"),
        # sub_node_update: T.Tensor((N_EDGES, OUT_DIM), "float32"),
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
    return edge_forward

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
        grad_out: T.Tensor((E, K), dtype),
        node_weight: T.Tensor((D, K), dtype),
        node_ext_weight: T.Tensor((D, K), dtype),
        n2e_index: T.Tensor((E,), T.int64),
        n_ext2e_index: T.Tensor((E,), T.int64),
        grad_node: T.Tensor((NODE, D), accum_dtype),
        grad_node_ext: T.Tensor((NODE, D), accum_dtype),
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

        # nf = 1  # Kept from the original implementation; batch is flattened.

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

        edge_forward_kernel(node_ebd, node_ebd_ext, flat_edge_ebd, n2e_index,
            n_ext2e_index, node, node_ext, edge, bias, out)

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
        grad_out: T.Tensor((E, K), dtype),
        node_ebd: T.Tensor((N, D), dtype),
        n2e_index: T.Tensor((E,), "int64"),
        grad_node_weight: T.Tensor((D, K), accum_dtype),
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
        grad_out: T.Tensor((E, K), dtype),
        n2e_index: T.Tensor((E,), "int64"),
        node_weight: T.Tensor((D, K), dtype),
        grad_node: T.Tensor((N, D), accum_dtype),
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
        node_ebd_ext: T.Tensor((N, D), dtype),
        n_ext2e_index: T.Tensor((E,), "int64"),
        grad_out: T.Tensor((E, K), dtype),
        grad_node_ext_weight: T.Tensor((D, K), accum_dtype),
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
        grad_out: T.Tensor((E, K), dtype),
        node_ext_weight: T.Tensor((D, K), dtype),
        n_ext2e_index: T.Tensor((E,), "int64"),
        grad_node_ext: T.Tensor((N, D), accum_dtype),
        grad_bias: T.Tensor((K,), accum_dtype),
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

@tilelang.jit
def fused_edge_update_weight_backward_v1(
    E,
    K,
    D_edge,
    N_node,
    D_node,
    N_ext,
    D_ext,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_E=32,
    BLOCK_D=16,
    BLOCK_K=32,
):
    MAX_D = max(D_edge, D_node, D_ext)

    @T.prim_func
    def weight_backward(
        grad_out: T.Tensor((E, K), dtype),
        flat_edge_ebd: T.Tensor((E, D_edge), dtype),
        node_ebd: T.Tensor((N_node, D_node), dtype),
        node_ebd_ext: T.Tensor((N_ext, D_ext), dtype),
        n2e_index: T.Tensor((E,), "int64"),
        n_ext2e_index: T.Tensor((E,), "int64"),
        grad_edge_weight: T.Tensor((D_edge, K), accum_dtype),
        grad_node_weight: T.Tensor((D_node, K), accum_dtype),
        grad_node_ext_weight: T.Tensor((D_ext, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(MAX_D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=128) as (bx, by):
            d_base = bx * BLOCK_D
            k_base = by * BLOCK_K
            grad_shared = T.alloc_shared((BLOCK_E, BLOCK_K), dtype)
            feature_shared = T.alloc_shared((BLOCK_D, BLOCK_E), dtype)

            grad_edge_weight_acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            grad_node_weight_acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            grad_node_ext_weight_acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(grad_edge_weight_acc)
            T.clear(grad_node_weight_acc)
            T.clear(grad_node_ext_weight_acc)

            for eo in T.serial(T.ceildiv(E, BLOCK_E)):
                e_base = eo * BLOCK_E
                for ei, ki in T.Parallel(BLOCK_E, BLOCK_K):
                    e = e_base + ei
                    k = k_base + ki
                    if e < E and k < K:
                        grad_shared[ei, ki] = grad_out[e, k]
                    else:
                        grad_shared[ei, ki] = 0.0

                T.sync_threads()

                for di, ei in T.Parallel(BLOCK_D, BLOCK_E):
                    d = d_base + di
                    e = e_base + ei
                    if d < D_edge and e < E:
                        feature_shared[di, ei] = flat_edge_ebd[e, d]
                    else:
                        feature_shared[di, ei] = 0.0

                T.sync_threads()

                if d_base < D_edge:
                    T.gemm(feature_shared, grad_shared, grad_edge_weight_acc)

                T.sync_threads()

                for di, ei in T.Parallel(BLOCK_D, BLOCK_E):
                    d = d_base + di
                    e = e_base + ei
                    if d < D_node and e < E:
                        node_id = n2e_index[e]
                        feature_shared[di, ei] = node_ebd[node_id, d]
                    else:
                        feature_shared[di, ei] = 0.0

                T.sync_threads()

                if d_base < D_node:
                    T.gemm(feature_shared, grad_shared, grad_node_weight_acc)

                T.sync_threads()

                for di, ei in T.Parallel(BLOCK_D, BLOCK_E):
                    d = d_base + di
                    e = e_base + ei
                    if d < D_ext and e < E:
                        ext_id = n_ext2e_index[e]
                        feature_shared[di, ei] = node_ebd_ext[ext_id, d]
                    else:
                        feature_shared[di, ei] = 0.0

                T.sync_threads()

                if d_base < D_ext:
                    T.gemm(feature_shared, grad_shared, grad_node_ext_weight_acc)

                T.sync_threads()

            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = d_base + di
                k = k_base + ki
                if d < D_edge and k < K:
                    grad_edge_weight[d, k] = grad_edge_weight_acc[di, ki]

                if d < D_node and k < K:
                    grad_node_weight[d, k] = grad_node_weight_acc[di, ki]

                if d < D_ext and k < K:
                    grad_node_ext_weight[d, k] = grad_node_ext_weight_acc[di, ki]

    return weight_backward

@tilelang.jit
def fused_edge_update_input_backward_v1(
    E,
    K,
    D_edge,
    D_node,
    D_ext,
    N_node,
    N_ext,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_E=32,
    BLOCK_D=16,
    BLOCK_K=32,
):
    MAX_D = max(D_edge, D_node, D_ext)

    @T.prim_func
    def input_backward(
        grad_out: T.Tensor((E, K), dtype),
        edge_weight: T.Tensor((D_edge, K), dtype),
        node_weight: T.Tensor((D_node, K), dtype),
        node_ext_weight: T.Tensor((D_ext, K), dtype),
        n2e_index: T.Tensor((E,), "int64"),
        n_ext2e_index: T.Tensor((E,), "int64"),
        grad_edge_ebd: T.Tensor((E, D_edge), accum_dtype),
        grad_node: T.Tensor((N_node, D_node), accum_dtype),
        grad_node_ext: T.Tensor((N_ext, D_ext), accum_dtype),
        grad_bias: T.Tensor((K,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(E, BLOCK_E), T.ceildiv(MAX_D, BLOCK_D), threads=128) as (bx, by):
            e_base = bx * BLOCK_E
            d_base = by * BLOCK_D

            grad_shared = T.alloc_shared((BLOCK_E, BLOCK_K), dtype)
            weight_shared = T.alloc_shared((BLOCK_K, BLOCK_D), dtype)

            grad_edge_acc = T.alloc_fragment((BLOCK_E, BLOCK_D), accum_dtype)
            grad_node_acc = T.alloc_fragment((BLOCK_E, BLOCK_D), accum_dtype)
            grad_node_ext_acc = T.alloc_fragment((BLOCK_E, BLOCK_D), accum_dtype)
            bias_tile = T.alloc_fragment((BLOCK_K,), accum_dtype)
            T.clear(grad_edge_acc)
            T.clear(grad_node_acc)
            T.clear(grad_node_ext_acc)
            T.clear(bias_tile)

            for ko in T.serial(T.ceildiv(K, BLOCK_K)):
                k_base = ko * BLOCK_K
                for ei, ki in T.Parallel(BLOCK_E, BLOCK_K):
                    e = e_base + ei
                    k = k_base + ki
                    if e < E and k < K:
                        grad_shared[ei, ki] = grad_out[e, k]
                    else:
                        grad_shared[ei, ki] = 0.0

                T.sync_threads()

                for ki, di in T.Parallel(BLOCK_K, BLOCK_D):
                    k = k_base + ki
                    d = d_base + di
                    if k < K and d < D_edge:
                        weight_shared[ki, di] = edge_weight[d, k]
                    else:
                        weight_shared[ki, di] = 0.0

                T.sync_threads()

                if d_base < D_edge:
                    T.gemm(grad_shared, weight_shared, grad_edge_acc)

                T.sync_threads()

                for ki, di in T.Parallel(BLOCK_K, BLOCK_D):
                    k = k_base + ki
                    d = d_base + di
                    if k < K and d < D_node:
                        weight_shared[ki, di] = node_weight[d, k]
                    else:
                        weight_shared[ki, di] = 0.0

                T.sync_threads()

                if d_base < D_node:
                    T.gemm(grad_shared, weight_shared, grad_node_acc)

                T.sync_threads()

                for ki, di in T.Parallel(BLOCK_K, BLOCK_D):
                    k = k_base + ki
                    d = d_base + di
                    if k < K and d < D_ext:
                        weight_shared[ki, di] = node_ext_weight[d, k]
                    else:
                        weight_shared[ki, di] = 0.0

                T.sync_threads()

                if d_base < D_ext:
                    T.gemm(grad_shared, weight_shared, grad_node_ext_acc)

                T.sync_threads()

                if by == 0:
                    for ki in T.Parallel(BLOCK_K):
                        k = k_base + ki
                        if k < K:
                            for ei in T.serial(BLOCK_E):
                                e = e_base + ei
                                if e < E:
                                    bias_tile[ki] += T.cast(grad_shared[ei, ki], accum_dtype)

                    for ki in T.Parallel(BLOCK_K):
                        k = k_base + ki
                        if k < K:
                            T.atomic_add(grad_bias[k], bias_tile[ki])

            if d_base < D_edge:
                for ei, di in T.Parallel(BLOCK_E, BLOCK_D):
                    e = e_base + ei
                    d = d_base + di
                    if e < E and d < D_edge:
                        grad_edge_ebd[e, d] = grad_edge_acc[ei, di]

            if d_base < D_node:
                for ei, di in T.Parallel(BLOCK_E, BLOCK_D):
                    e = e_base + ei
                    d = d_base + di
                    if e < E and d < D_node:
                        node_id = n2e_index[e]
                        T.atomic_add(grad_node[node_id, d], grad_node_acc[ei, di])

            if d_base < D_ext:
                for ei, di in T.Parallel(BLOCK_E, BLOCK_D):
                    e = e_base + ei
                    d = d_base + di
                    if e < E and d < D_ext:
                        ext_id = n_ext2e_index[e]
                        T.atomic_add(grad_node_ext[ext_id, d], grad_node_ext_acc[ei, di])

    return input_backward


@tilelang.jit
def fused_edge_update_weight_backward_v2(
    E, K, D_edge, D_node, D_ext, N_node, N_ext,
    dtype="float32", accum_dtype="float32",
    BLOCK_D=32, BLOCK_K=16, THREADS=128, BLOCK_M=32, SPLIT_M=4,
):
    """Gather a logical feature concatenation and compute split-E partials.

    workspace[s] = X_s.T @ G_s, X = [node[n2e], node_ext[n_ext2e], edge].
    SPLIT_M partitions whole BLOCK_M tiles of the reduction dimension E.
    Empty partitions explicitly write zeros. A second kernel reduces workspace.
    No full gathered/concatenated input is materialized in global memory.
    """
    assert SPLIT_M > 0
    D = D_node + D_ext + D_edge
    TILES_PER_SPLIT = ((E + BLOCK_M - 1) // BLOCK_M + SPLIT_M - 1) // SPLIT_M

    @T.prim_func
    def edge_backward_weight_partials(
        grad_out: T.Tensor((E, K), dtype),
        n2e_index: T.Tensor((E,), "int64"),
        n_ext2e_index: T.Tensor((E,), "int64"),
        node_ebd: T.Tensor((N_node, D_node), dtype),
        node_ebd_ext: T.Tensor((N_ext, D_ext), dtype),
        flat_edge_ebd: T.Tensor((E, D_edge), dtype),
        workspace: T.Tensor((SPLIT_M, D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), SPLIT_M, threads=THREADS) as (bx, by, bs):
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc)
            for mo in T.Pipelined(TILES_PER_SPLIT, num_stages=2):
                # Row-major shared layout; GEMM handles the transpose.
                # Element guards also support feature tiles crossing segments.
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                    d = bx * BLOCK_D + di
                    if m < E and d < D:
                        if d < D_node:
                            feature_shared[mi, di] = node_ebd[n2e_index[m], d]
                        elif d < D_node + D_ext:
                            feature_shared[mi, di] = node_ebd_ext[n_ext2e_index[m], d - D_node]
                        else:
                            feature_shared[mi, di] = flat_edge_ebd[m, d - D_node - D_ext]
                    else:
                        feature_shared[mi, di] = 0
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < E and k < K:
                        grad_shared[mi, ki] = grad_out[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    workspace[bs, d, k] = acc[di, ki]
    return edge_backward_weight_partials


@tilelang.jit
def fused_edge_update_weight_backward_v3(
    E, K, D_edge, D_node, D_ext, N_node, N_ext,
    dtype="float32", accum_dtype="float32",
    BLOCK_D=32, BLOCK_K=32, THREADS=256, BLOCK_M=32, SPLIT_M=4,
):
    """V2 with V3.3-style float4 producers and XOR shared layouts."""
    assert SPLIT_M > 0
    assert dtype == "float32"
    assert BLOCK_D == 32
    assert BLOCK_K == 32
    assert BLOCK_M == 32
    assert THREADS == 256
    # A float4 task must not cross a logical feature-segment boundary.
    assert D_node % 4 == 0
    assert D_ext % 4 == 0
    assert D_edge % 4 == 0
    assert K % 4 == 0
    D = D_node + D_ext + D_edge
    TILES_PER_SPLIT = ((E + BLOCK_M - 1) // BLOCK_M + SPLIT_M - 1) // SPLIT_M

    @T.prim_func
    def edge_backward_weight_partials(
        grad_out: T.Tensor((E, K), dtype),
        n2e_index: T.Tensor((E,), "int64"),
        n_ext2e_index: T.Tensor((E,), "int64"),
        node_ebd: T.Tensor((N_node, D_node), dtype),
        node_ebd_ext: T.Tensor((N_ext, D_ext), dtype),
        flat_edge_ebd: T.Tensor((E, D_edge), dtype),
        workspace: T.Tensor((SPLIT_M, D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), SPLIT_M, threads=THREADS) as (bx, by, bs):
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.annotate_layout({
                feature_shared: T.Layout(
                    (BLOCK_M, BLOCK_D),
                    lambda mi, di: [
                        mi,
                        (di // 4) ^ ((mi % 4) * 2),
                        di % 4,
                    ],
                ),
                grad_shared: T.Layout(
                    (BLOCK_M, BLOCK_K),
                    lambda mi, ki: [
                        mi,
                        (ki // 4) ^ ((mi % 4) * 2),
                        ki % 4,
                    ],
                ),
            })
            T.clear(acc)
            for mo in T.Pipelined(TILES_PER_SPLIT, num_stages=2):
                for mi, di4 in T.Parallel(BLOCK_M, BLOCK_D // 4):
                    m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                    d_base = bx * BLOCK_D + di4 * 4
                    if m < E and d_base < D:
                        if d_base < D_node:
                            for vi in T.vectorized(4):
                                feature_shared[mi, di4 * 4 + vi] = node_ebd[n2e_index[m], d_base + vi]
                        elif d_base < D_node + D_ext:
                            for vi in T.vectorized(4):
                                feature_shared[mi, di4 * 4 + vi] = node_ebd_ext[n_ext2e_index[m], d_base - D_node + vi]
                        else:
                            for vi in T.vectorized(4):
                                feature_shared[mi, di4 * 4 + vi] = flat_edge_ebd[m, d_base - D_node - D_ext + vi]
                    else:
                        for vi in T.vectorized(4):
                            feature_shared[mi, di4 * 4 + vi] = 0

                for mi, ki4 in T.Parallel(BLOCK_M, BLOCK_K // 4):
                    m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                    k_base = by * BLOCK_K + ki4 * 4
                    if m < E and k_base < K:
                        for vi in T.vectorized(4):
                            grad_shared[mi, ki4 * 4 + vi] = grad_out[m, k_base + vi]
                    else:
                        for vi in T.vectorized(4):
                            grad_shared[mi, ki4 * 4 + vi] = 0
                T.sync_threads()
                T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    workspace[bs, d, k] = acc[di, ki]
    return edge_backward_weight_partials

@tilelang.jit
def fused_edge_update_weight_backward_v2_reduce(
    K, D_edge, D_node, D_ext, SPLIT_M=4, accum_dtype="float32",
    BLOCK_D=32, BLOCK_K=16, THREADS=128,
):
    """Sum partials in increasing split order and write three contiguous outputs.

    No atomic additions: each output element has a unique writer and an explicit
    serial accumulation order. This does not change the GEMM operand precision.
    """
    D = D_node + D_ext + D_edge

    @T.prim_func
    def edge_backward_reduce_partials(
        workspace: T.Tensor((SPLIT_M, D, K), accum_dtype),
        grad_node_weight: T.Tensor((D_node, K), accum_dtype),
        grad_node_ext_weight: T.Tensor((D_ext, K), accum_dtype),
        grad_edge_weight: T.Tensor((D_edge, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=THREADS) as (bx, by):
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc)
            for split in T.serial(SPLIT_M):
                for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                    d = bx * BLOCK_D + di
                    k = by * BLOCK_K + ki
                    if d < D and k < K:
                        acc[di, ki] += workspace[split, d, k]
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    if d < D_node:
                        grad_node_weight[d, k] = acc[di, ki]
                    elif d < D_node + D_ext:
                        grad_node_ext_weight[d - D_node, k] = acc[di, ki]
                    else:
                        grad_edge_weight[d - D_node - D_ext, k] = acc[di, ki]
    return edge_backward_reduce_partials


@tilelang.jit
def fused_edge_update_backward_inputs_v2(
    E, K, D_edge, D_node, D_ext, N_node, N_ext,
    dtype="float32", accum_dtype="float32",
    BLOCK_E=32, BLOCK_D=32, BLOCK_K=32, THREADS=128,
):
    """Compute G @ [W_edge; W_node; W_ext].T without materializing concat.

    Node/ext outputs and bias must be zero-initialized before launch.
    Arbitrary repeated/unordered indices are supported. Each direct edge
    output is uniquely written; node/ext contributions use scatter-add.
    """
    D = D_edge + D_node + D_ext

    @T.prim_func
    def edge_backward_inputs(
        grad_out: T.Tensor((E, K), dtype),
        edge_weight: T.Tensor((D_edge, K), dtype),
        node_weight: T.Tensor((D_node, K), dtype),
        node_ext_weight: T.Tensor((D_ext, K), dtype),
        n2e_index: T.Tensor((E,), "int64"),
        n_ext2e_index: T.Tensor((E,), "int64"),
        grad_edge_ebd: T.Tensor((E, D_edge), accum_dtype),
        grad_node: T.Tensor((N_node, D_node), accum_dtype),
        grad_node_ext: T.Tensor((N_ext, D_ext), accum_dtype),
        grad_bias: T.Tensor((K,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(E, BLOCK_E), T.ceildiv(D, BLOCK_D), threads=THREADS) as (bx, by):
            grad_shared = T.alloc_shared((BLOCK_E, BLOCK_K), dtype)
            weight_shared = T.alloc_shared((BLOCK_D, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_E, BLOCK_D), accum_dtype)
            bias_tile = T.alloc_fragment((BLOCK_K,), accum_dtype)
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                for ei, ki in T.Parallel(BLOCK_E, BLOCK_K):
                    e = bx * BLOCK_E + ei
                    k = ko * BLOCK_K + ki
                    if e < E and k < K:
                        grad_shared[ei, ki] = grad_out[e, k]
                    else:
                        grad_shared[ei, ki] = 0
                for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                    k = ko * BLOCK_K + ki
                    d = by * BLOCK_D + di
                    if k < K and d < D:
                        if d < D_edge:
                            weight_shared[di, ki] = edge_weight[d, k]
                        elif d < D_edge + D_node:
                            weight_shared[di, ki] = node_weight[d - D_edge, k]
                        else:
                            weight_shared[di, ki] = node_ext_weight[d - D_edge - D_node, k]
                    else:
                        weight_shared[di, ki] = 0
                T.sync_threads()
                T.gemm(grad_shared, weight_shared, acc, transpose_B=True)
                T.sync_threads()
            # Separate writeback; gather's adjoint is scatter-add, not a split.
            for ei, di in T.Parallel(BLOCK_E, BLOCK_D):
                e = bx * BLOCK_E + ei
                d = by * BLOCK_D + di
                if e < E and d < D:
                    if d < D_edge:
                        grad_edge_ebd[e, d] = acc[ei, di]
                    elif d < D_edge + D_node:
                        T.atomic_add(grad_node[n2e_index[e], d - D_edge], acc[ei, di])
                    else:
                        T.atomic_add(grad_node_ext[n_ext2e_index[e], d - D_edge - D_node], acc[ei, di])
            # Bias is independent of feature segments and GEMM multibuffering.
            # Exactly one feature block per edge tile contributes. Reset for
            # EVERY K tile: carrying the previous tile corrupts K > BLOCK_K.
            if by == 0:
                for ko in T.serial(T.ceildiv(K, BLOCK_K)):
                    T.clear(bias_tile)
                    for ki in T.Parallel(BLOCK_K):
                        k = ko * BLOCK_K + ki
                        if k < K:
                            for ei in T.serial(BLOCK_E):
                                e = bx * BLOCK_E + ei
                                if e < E:
                                    bias_tile[ki] += grad_out[e, k]
                    for ki in T.Parallel(BLOCK_K):
                        k = ko * BLOCK_K + ki
                        if k < K:
                            T.atomic_add(grad_bias[k], bias_tile[ki])
    return edge_backward_inputs


@tilelang.jit
def fused_edge_update_double_backward_inputs(
    E,
    K,
    D_edge,
    D_node,
    D_ext,
    N_node,
    N_ext,
    dtype="float32",
    accum_dtype="float32",
    THREADS=128,
    BLOCK_M=32,
    BLOCK_N=32,
    BLOCK_K=32,
    HAS_NODE=True, HAS_EXT=True, HAS_EDGE=True, HAS_NODE_WEIGHT=True, HAS_EXT_WEIGHT=True, HAS_EDGE_WEIGHT=True, HAS_BIAS=True,
):
    """Fuse grad-grad-output and all three input-side VJPs.

    HAS_* are JIT specialization flags, not device tensor arguments. An absent
    cotangent's pointer is never read. Scatter outputs must still start at zero.
    """

    @T.prim_func
    def edge_double_backward_inputs(
        grad_out: T.Tensor((E, K), dtype),
        node_ebd: T.Tensor((N_node, D_node), dtype),
        node_ebd_ext: T.Tensor((N_ext, D_ext), dtype),
        flat_edge_ebd: T.Tensor((E, D_edge), dtype),
        n2e_index: T.Tensor((E,), "int64"),
        n_ext2e_index: T.Tensor((E,), "int64"),
        node_weight: T.Tensor((D_node, K), dtype),
        node_ext_weight: T.Tensor((D_ext, K), dtype),
        edge_weight: T.Tensor((D_edge, K), dtype),
        grad_grad_node: T.Tensor((N_node, D_node), dtype),
        grad_grad_node_ext: T.Tensor((N_ext, D_ext), dtype),
        grad_grad_edge_ebd: T.Tensor((E, D_edge), dtype),
        grad_grad_node_weight: T.Tensor((D_node, K), dtype),
        grad_grad_node_ext_weight: T.Tensor((D_ext, K), dtype),
        grad_grad_edge_weight: T.Tensor((D_edge, K), dtype),
        grad_grad_bias: T.Tensor((K,), dtype),
        grad_grad_out: T.Tensor((E, K), accum_dtype),
        grad_node_ebd: T.Tensor((N_node, D_node), accum_dtype),
        grad_node_ebd_ext: T.Tensor((N_ext, D_ext), accum_dtype),
        grad_flat_edge_ebd: T.Tensor((E, D_edge), accum_dtype),
    ):
        # Original scalar implementation (retained for reference):
        # with T.Kernel(E, threads=THREADS) as (edge_idx,):
        # node_idx = n2e_index[edge_idx]
        # ext_idx = n_ext2e_index[edge_idx]
        #
        # for k in T.Parallel(K):
        # acc = T.alloc_var(accum_dtype)
        # acc = grad_grad_bias[k]
        # for d in T.serial(D_edge):
        # acc += (
        # grad_grad_edge_ebd[edge_idx, d] * edge_weight[d, k]
        # + flat_edge_ebd[edge_idx, d] * grad_grad_edge_weight[d, k]
        # )
        # for d in T.serial(D_node):
        # acc += (
        # node_ebd[node_idx, d] * grad_grad_node_weight[d, k]
        # + grad_grad_node[node_idx, d] * node_weight[d, k]
        # )
        # for d in T.serial(D_ext):
        # acc += (
        # node_ebd_ext[ext_idx, d] * grad_grad_node_ext_weight[d, k]
        # + grad_grad_node_ext[ext_idx, d] * node_ext_weight[d, k]
        # )
        # grad_grad_out[edge_idx, k] = acc
        #
        # for d in T.Parallel(D_edge):
        # acc = T.alloc_var(accum_dtype, init=0)
        # for k in T.serial(K):
        # acc += grad_out[edge_idx, k] * grad_grad_edge_weight[d, k]
        # grad_flat_edge_ebd[edge_idx, d] = acc
        #
        # for d in T.Parallel(D_node):
        # acc = T.alloc_var(accum_dtype, init=0)
        # for k in T.serial(K):
        # acc += grad_out[edge_idx, k] * grad_grad_node_weight[d, k]
        # T.atomic_add(grad_node_ebd[node_idx, d], acc)
        #
        # for d in T.Parallel(D_ext):
        # acc = T.alloc_var(accum_dtype, init=0)
        # for k in T.serial(K):
        # acc += grad_out[edge_idx, k] * grad_grad_node_ext_weight[d, k]
        # T.atomic_add(grad_node_ebd_ext[ext_idx, d], acc)

        # A common output-column grid covers grad-grad-output and input VJPs.
        with T.Kernel(T.ceildiv(E, BLOCK_M), T.ceildiv(max(K, D_edge, D_node, D_ext), BLOCK_N), threads=THREADS) as (bx, by):
            lhs_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            rhs_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
            acc_output = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            acc_input = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            T.clear(acc_output)
            if by * BLOCK_N < K:
                # edge: U_x W and X U_w; each has its own pipeline.
                if HAS_EDGE:
                    for ro in T.Pipelined(T.ceildiv(D_edge, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < E and r < D_edge:
                                lhs_shared[mi, ri] = grad_grad_edge_ebd[m, r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < D_edge and n < K:
                                rhs_shared[ri, ni] = edge_weight[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_output)
                        T.sync_threads()
                T.sync_threads()
                if HAS_EDGE_WEIGHT:
                    for ro in T.Pipelined(T.ceildiv(D_edge, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < E and r < D_edge:
                                lhs_shared[mi, ri] = flat_edge_ebd[m, r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < D_edge and n < K:
                                rhs_shared[ri, ni] = grad_grad_edge_weight[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_output)
                        T.sync_threads()
                T.sync_threads()
                # node: U_x W and X U_w; each has its own pipeline.
                if HAS_NODE:
                    for ro in T.Pipelined(T.ceildiv(D_node, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < E and r < D_node:
                                lhs_shared[mi, ri] = grad_grad_node[n2e_index[m], r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < D_node and n < K:
                                rhs_shared[ri, ni] = node_weight[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_output)
                        T.sync_threads()
                T.sync_threads()
                if HAS_NODE_WEIGHT:
                    for ro in T.Pipelined(T.ceildiv(D_node, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < E and r < D_node:
                                lhs_shared[mi, ri] = node_ebd[n2e_index[m], r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < D_node and n < K:
                                rhs_shared[ri, ni] = grad_grad_node_weight[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_output)
                        T.sync_threads()
                T.sync_threads()
                # ext: U_x W and X U_w; each has its own pipeline.
                if HAS_EXT:
                    for ro in T.Pipelined(T.ceildiv(D_ext, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < E and r < D_ext:
                                lhs_shared[mi, ri] = grad_grad_node_ext[n_ext2e_index[m], r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < D_ext and n < K:
                                rhs_shared[ri, ni] = node_ext_weight[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_output)
                        T.sync_threads()
                T.sync_threads()
                if HAS_EXT_WEIGHT:
                    for ro in T.Pipelined(T.ceildiv(D_ext, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < E and r < D_ext:
                                lhs_shared[mi, ri] = node_ebd_ext[n_ext2e_index[m], r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < D_ext and n < K:
                                rhs_shared[ri, ni] = grad_grad_node_ext_weight[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_output)
                        T.sync_threads()
                T.sync_threads()
            for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                m = bx * BLOCK_M + mi
                n = by * BLOCK_N + ni
                if m < E and n < K:
                    if HAS_BIAS:
                        grad_grad_out[m, n] = acc_output[mi, ni] + grad_grad_bias[n]
                    else:
                        grad_grad_out[m, n] = acc_output[mi, ni]

            # edge input VJP: G U_w.T; reuse the fragment after writeback.
            if by * BLOCK_N < D_edge:
                T.clear(acc_input)
                if HAS_EDGE_WEIGHT:
                    for ro in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < E and r < K:
                                lhs_shared[mi, ri] = grad_out[m, r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < K and n < D_edge:
                                rhs_shared[ri, ni] = grad_grad_edge_weight[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_input, transpose_B=True)
                        T.sync_threads()
                T.sync_threads()
                for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                    m = bx * BLOCK_M + mi
                    n = by * BLOCK_N + ni
                    if m < E and n < D_edge:
                        grad_flat_edge_ebd[m, n] = acc_input[mi, ni]
                T.sync_threads()

            # node input VJP: missing cotangent leaves the zeroed output unchanged.
            if HAS_NODE_WEIGHT and by * BLOCK_N < D_node:
                T.clear(acc_input)
                if HAS_NODE_WEIGHT:
                    for ro in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < E and r < K:
                                lhs_shared[mi, ri] = grad_out[m, r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < K and n < D_node:
                                rhs_shared[ri, ni] = grad_grad_node_weight[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_input, transpose_B=True)
                        T.sync_threads()
                T.sync_threads()
                for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                    m = bx * BLOCK_M + mi
                    n = by * BLOCK_N + ni
                    if m < E and n < D_node:
                        T.atomic_add(grad_node_ebd[n2e_index[m], n], acc_input[mi, ni])
                T.sync_threads()

            # ext input VJP: missing cotangent leaves the zeroed output unchanged.
            if HAS_EXT_WEIGHT and by * BLOCK_N < D_ext:
                T.clear(acc_input)
                if HAS_EXT_WEIGHT:
                    for ro in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < E and r < K:
                                lhs_shared[mi, ri] = grad_out[m, r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < K and n < D_ext:
                                rhs_shared[ri, ni] = grad_grad_node_ext_weight[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_input, transpose_B=True)
                        T.sync_threads()
                T.sync_threads()
                for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                    m = bx * BLOCK_M + mi
                    n = by * BLOCK_N + ni
                    if m < E and n < D_ext:
                        T.atomic_add(grad_node_ebd_ext[n_ext2e_index[m], n], acc_input[mi, ni])
                T.sync_threads()

    return edge_double_backward_inputs


@tilelang.jit
def fused_edge_update_double_backward_weights(
    E,
    K,
    D_edge,
    D_node,
    D_ext,
    N_node,
    N_ext,
    dtype="float32",
    accum_dtype="float32",
    # Original scalar tile: BLOCK_D=16.
    BLOCK_D=32,
    BLOCK_K=16,
    THREADS=128,
    BLOCK_M=32,
):
    """Fuse the three weight-side VJPs into one balanced reduction kernel."""
    MAX_D = max(D_edge, D_node, D_ext)

    @T.prim_func
    def double_backward_weights(
        grad_out: T.Tensor((E, K), dtype),
        n2e_index: T.Tensor((E,), "int64"),
        n_ext2e_index: T.Tensor((E,), "int64"),
        grad_grad_node: T.Tensor((N_node, D_node), dtype),
        grad_grad_node_ext: T.Tensor((N_ext, D_ext), dtype),
        grad_grad_edge_ebd: T.Tensor((E, D_edge), dtype),
        grad_node_weight: T.Tensor((D_node, K), accum_dtype),
        grad_node_ext_weight: T.Tensor((D_ext, K), accum_dtype),
        grad_edge_weight: T.Tensor((D_edge, K), accum_dtype),
    ):
        # Original scalar implementation (retained for reference):
        # with T.Kernel(
        # T.ceildiv(MAX_D, BLOCK_D),
        # T.ceildiv(K, BLOCK_K),
        # threads=THREADS,
        # ) as (bx, by):
        # for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
        # d = bx * BLOCK_D + di
        # k = by * BLOCK_K + ki
        # if k < K:
        # if d < D_edge:
        # acc_edge = T.alloc_var(accum_dtype, init=0)
        # for e in T.serial(E):
        # acc_edge += grad_grad_edge_ebd[e, d] * grad_out[e, k]
        # grad_edge_weight[d, k] = acc_edge
        # if d < D_node:
        # acc_node = T.alloc_var(accum_dtype, init=0)
        # for e in T.serial(E):
        # acc_node += grad_grad_node[n2e_index[e], d] * grad_out[e, k]
        # grad_node_weight[d, k] = acc_node
        # if d < D_ext:
        # acc_ext = T.alloc_var(accum_dtype, init=0)
        # for e in T.serial(E):
        # acc_ext += grad_grad_node_ext[n_ext2e_index[e], d] * grad_out[e, k]
        # grad_node_ext_weight[d, k] = acc_ext

        with T.Kernel(T.ceildiv(MAX_D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=THREADS) as (bx, by):
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            # edge: gathered U_x.T @ G, with row-major shared loads.
            T.clear(acc)
            for mo in T.Pipelined(T.ceildiv(E, BLOCK_M), num_stages=2):
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    m = mo * BLOCK_M + mi
                    d = bx * BLOCK_D + di
                    if m < E and d < D_edge:
                        feature_shared[mi, di] = grad_grad_edge_ebd[m, d]
                    else:
                        feature_shared[mi, di] = 0
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = mo * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < E and k < K:
                        grad_shared[mi, ki] = grad_out[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                T.sync_threads()
            T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D_edge and k < K:
                    grad_edge_weight[d, k] = acc[di, ki]
            T.sync_threads()

            # node: gathered U_x.T @ G, with row-major shared loads.
            T.clear(acc)
            for mo in T.Pipelined(T.ceildiv(E, BLOCK_M), num_stages=2):
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    m = mo * BLOCK_M + mi
                    d = bx * BLOCK_D + di
                    if m < E and d < D_node:
                        feature_shared[mi, di] = grad_grad_node[n2e_index[m], d]
                    else:
                        feature_shared[mi, di] = 0
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = mo * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < E and k < K:
                        grad_shared[mi, ki] = grad_out[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                T.sync_threads()
            T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D_node and k < K:
                    grad_node_weight[d, k] = acc[di, ki]
            T.sync_threads()

            # ext: gathered U_x.T @ G, with row-major shared loads.
            T.clear(acc)
            for mo in T.Pipelined(T.ceildiv(E, BLOCK_M), num_stages=2):
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    m = mo * BLOCK_M + mi
                    d = bx * BLOCK_D + di
                    if m < E and d < D_ext:
                        feature_shared[mi, di] = grad_grad_node_ext[n_ext2e_index[m], d]
                    else:
                        feature_shared[mi, di] = 0
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = mo * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < E and k < K:
                        grad_shared[mi, ki] = grad_out[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                T.sync_threads()
            T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D_ext and k < K:
                    grad_node_ext_weight[d, k] = acc[di, ki]
            T.sync_threads()

    return double_backward_weights

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
        
        """
        grad_edge_ebd = grad_out @ edge_weight.T
        grad_edge_weight = flat_edge_ebd.T @ grad_out

        # gathered_node = node_ebd[n2e_index]
        # grad_node_weight = gathered_node.T @ grad_out
        grad_node = torch.zeros(
            (N_node, D_node),
            device=grad_out.device,
            dtype=grad_out.dtype,
        )
        # grad_gathered_node = grad_out @ node_weight.T
        # grad_node.index_add_(0, n2e_index, grad_gathered_node)
        
        # gathered_node_ext = node_ebd_ext[n_ext2e_index]
        # grad_node_ext_weight = gathered_node_ext.T @ grad_out

        grad_node_ext = torch.zeros(
            (N_ext, D_ext),
            device=grad_out.device,
            dtype=grad_out.dtype,
        )
        # grad_gathered_node_ext = grad_out @ node_ext_weight.T
        # grad_node_ext.index_add_(0, n_ext2e_index, grad_gathered_node_ext)
        # grad_bias = grad_out.sum(dim=0)
        """
        grad_edge_weight = torch.empty(
            (D_edge, K),
            device=grad_out.device,
            dtype=grad_out.dtype,
        )

        grad_node_weight = torch.empty(
            (D_node, K),
            device=grad_out.device,
            dtype=grad_out.dtype,
        )

        grad_node_ext_weight = torch.empty(
            (D_ext, K),
            device=grad_out.device,
            dtype=grad_out.dtype,
        )

        # weight_kernel = fused_edge_update_weight_backward_v1(
        #     E=E,
        #     K=K,
        #     D_edge=D_edge,
        #     N_node=N_node,
        #     D_node=D_node,
        #     N_ext=N_ext,
        #     D_ext=D_ext,
        # )
        #
        # weight_kernel(
        #     grad_out,
        #     flat_edge_ebd,
        #     node_ebd,
        #     node_ebd_ext,
        #     n2e_index,
        #     n_ext2e_index,
        #     grad_edge_weight,
        #     grad_node_weight,
        #     grad_node_ext_weight,
        # )
        dtype = str(grad_out.dtype).replace("torch.", "")
        split_m = min(30, max(1, (E + 31) // 32))
        workspace = torch.empty(
            (split_m, D_node + D_ext + D_edge, K),
            device=grad_out.device, dtype=grad_out.dtype,
        )
        weight_kernel = fused_edge_update_weight_backward_v3(
            E=E, K=K, D_edge=D_edge, D_node=D_node, D_ext=D_ext,
            N_node=N_node, N_ext=N_ext, dtype=dtype, accum_dtype=dtype,
            SPLIT_M=split_m,
        )
        weight_kernel(
            grad_out, n2e_index, n_ext2e_index,
            node_ebd, node_ebd_ext, flat_edge_ebd, workspace,
        )
        weight_reduce = fused_edge_update_weight_backward_v2_reduce(
            K=K, D_edge=D_edge, D_node=D_node, D_ext=D_ext,
            SPLIT_M=split_m, accum_dtype=dtype,
        )
        weight_reduce(
            workspace, grad_node_weight, grad_node_ext_weight, grad_edge_weight,
        )

        grad_edge_ebd = torch.empty(
            (E, D_edge),
            device=grad_out.device,
            dtype=grad_out.dtype,
        )

        grad_node = torch.zeros(
            (N_node, D_node),
            device=grad_out.device,
            dtype=grad_out.dtype,
        )

        grad_node_ext = torch.zeros(
            (N_ext, D_ext),
            device=grad_out.device,
            dtype=grad_out.dtype,
        )

        grad_bias = torch.zeros(
            (K,),
            device=grad_out.device,
            dtype=grad_out.dtype,
        )

        # input_kernel = fused_edge_update_input_backward_v1(
        #     E=E,
        #     K=K,
        #     D_edge=D_edge,
        #     D_node=D_node,
        #     D_ext=D_ext,
        #     N_node=N_node,
        #     N_ext=N_ext,
        # )
        #
        # input_kernel(
        #     grad_out,
        #     edge_weight,
        #     node_weight,
        #     node_ext_weight,
        #     n2e_index,
        #     n_ext2e_index,
        #     grad_edge_ebd,
        #     grad_node,
        #     grad_node_ext,
        #     grad_bias,
        # )
        input_kernel = fused_edge_update_backward_inputs_v2(
            E=E,
            K=K,
            D_edge=D_edge,
            D_node=D_node,
            D_ext=D_ext,
            N_node=N_node,
            N_ext=N_ext,
            dtype=dtype, accum_dtype=dtype,
        )

        input_kernel(
            grad_out,
            edge_weight,
            node_weight,
            node_ext_weight,
            n2e_index,
            n_ext2e_index,
            grad_edge_ebd,
            grad_node,
            grad_node_ext,
            grad_bias,
        )

        # Preserve undefined second-order cotangents instead of autograd zeros.
        ctx.set_materialize_grads(False)
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

        """
        Original unfused second-order backward implementation.
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
        """

        E, K = grad_out.shape
        N_node, D_node = node_ebd.shape
        N_ext, D_ext = node_ebd_ext.shape
        _, D_edge = flat_edge_ebd.shape
        dtype = str(grad_out.dtype).replace("torch.", "")

        # Original zero-materialization code retained below for reference.
        # # Use zero cotangents for outputs that were not used by the caller.
        # optional_grads = (
        #     ("grad_grad_node", grad_grad_node, node_ebd),
        #     ("grad_grad_node_ext", grad_grad_node_ext, node_ebd_ext),
        #     ("grad_grad_edge_ebd", grad_grad_edge_ebd, flat_edge_ebd),
        #     ("grad_grad_node_weight", grad_grad_node_weight, node_weight),
        #     ("grad_grad_node_ext_weight", grad_grad_node_ext_weight, node_ext_weight),
        #     ("grad_grad_edge_weight", grad_grad_edge_weight, edge_weight),
        # )
        # materialized = {
        #     name: torch.zeros_like(reference) if value is None else value.contiguous()
        #     for name, value, reference in optional_grads
        # }
        # if grad_grad_bias is None:
        #     grad_grad_bias = torch.zeros((K,), device=grad_out.device, dtype=grad_out.dtype)
        # else:
        #     grad_grad_bias = grad_grad_bias.contiguous()
        #
        # grad_grad_node = materialized["grad_grad_node"]
        # grad_grad_node_ext = materialized["grad_grad_node_ext"]
        # grad_grad_edge_ebd = materialized["grad_grad_edge_ebd"]
        # grad_grad_node_weight = materialized["grad_grad_node_weight"]
        # grad_grad_node_ext_weight = materialized["grad_grad_node_ext_weight"]
        # grad_grad_edge_weight = materialized["grad_grad_edge_weight"]
        HAS_NODE = grad_grad_node is not None
        grad_grad_node = None if grad_grad_node is None else grad_grad_node.contiguous()
        HAS_EXT = grad_grad_node_ext is not None
        grad_grad_node_ext = None if grad_grad_node_ext is None else grad_grad_node_ext.contiguous()
        HAS_EDGE = grad_grad_edge_ebd is not None
        grad_grad_edge_ebd = None if grad_grad_edge_ebd is None else grad_grad_edge_ebd.contiguous()
        HAS_NODE_WEIGHT = grad_grad_node_weight is not None
        grad_grad_node_weight = None if grad_grad_node_weight is None else grad_grad_node_weight.contiguous()
        HAS_EXT_WEIGHT = grad_grad_node_ext_weight is not None
        grad_grad_node_ext_weight = None if grad_grad_node_ext_weight is None else grad_grad_node_ext_weight.contiguous()
        HAS_EDGE_WEIGHT = grad_grad_edge_weight is not None
        grad_grad_edge_weight = None if grad_grad_edge_weight is None else grad_grad_edge_weight.contiguous()
        HAS_BIAS = grad_grad_bias is not None
        grad_grad_bias = None if grad_grad_bias is None else grad_grad_bias.contiguous()

        # Saved tensors serve as unread, shape-compatible launch placeholders.
        # The missing cotangents themselves remain None in Python.
        grad_grad_out = torch.empty_like(grad_out)
        grad_grad_node_ebd = torch.zeros_like(node_ebd)
        grad_grad_node_ebd_ext = torch.zeros_like(node_ebd_ext)
        grad_grad_flat_edge_ebd = torch.empty_like(flat_edge_ebd)
        input_kernel = fused_edge_update_double_backward_inputs(
            E=E,
            K=K,
            D_edge=D_edge,
            D_node=D_node,
            D_ext=D_ext,
            N_node=N_node,
            N_ext=N_ext,
            dtype=dtype,
            HAS_NODE=HAS_NODE,
            HAS_EXT=HAS_EXT,
            HAS_EDGE=HAS_EDGE,
            HAS_NODE_WEIGHT=HAS_NODE_WEIGHT,
            HAS_EXT_WEIGHT=HAS_EXT_WEIGHT,
            HAS_EDGE_WEIGHT=HAS_EDGE_WEIGHT,
            HAS_BIAS=HAS_BIAS,
        )
        input_kernel(
            grad_out,
            node_ebd,
            node_ebd_ext,
            flat_edge_ebd,
            n2e_index,
            n_ext2e_index,
            node_weight,
            node_ext_weight,
            edge_weight,
            grad_grad_node if HAS_NODE else node_ebd,
            grad_grad_node_ext if HAS_EXT else node_ebd_ext,
            grad_grad_edge_ebd if HAS_EDGE else flat_edge_ebd,
            grad_grad_node_weight if HAS_NODE_WEIGHT else node_weight,
            grad_grad_node_ext_weight if HAS_EXT_WEIGHT else node_ext_weight,
            grad_grad_edge_weight if HAS_EDGE_WEIGHT else edge_weight,
            grad_grad_bias if HAS_BIAS else node_weight[0],
            grad_grad_out,
            grad_grad_node_ebd,
            grad_grad_node_ebd_ext,
            grad_grad_flat_edge_ebd,
        )

        grad_grad_node_weight_input = torch.empty_like(node_weight)
        grad_grad_node_ext_weight_input = torch.empty_like(node_ext_weight)
        grad_grad_edge_weight_input = torch.empty_like(edge_weight)
        # weight_kernel = fused_edge_update_double_backward_weights(
        #     E=E,
        #     K=K,
        #     D_edge=D_edge,
        #     D_node=D_node,
        #     D_ext=D_ext,
        #     N_node=N_node,
        #     N_ext=N_ext,
        #     dtype=dtype,
        # )
        # weight_kernel(
        #     grad_out,
        #     n2e_index,
        #     n_ext2e_index,
        #     grad_grad_node,
        #     grad_grad_node_ext,
        #     grad_grad_edge_ebd,
        #     grad_grad_node_weight_input,
        #     grad_grad_node_ext_weight_input,
        #     grad_grad_edge_weight_input,
        # )
        # Split whole reduction tiles; keep the short edge reduction modest.
        split_m = min(33, max(1, (E + 31) // 32))
        workspace = torch.empty(
            (split_m, D_node + D_ext + D_edge, K),
            dtype=grad_out.dtype, device=grad_out.device,
        )
        weight_kernel = fused_edge_update_double_backward_weights_v3(
            E=E, K=K, D_edge=D_edge, D_node=D_node, D_ext=D_ext,
            N_node=N_node, N_ext=N_ext, dtype=dtype,
            accum_dtype=dtype, SPLIT_M=split_m,
            HAS_NODE=HAS_NODE, HAS_EXT=HAS_EXT, HAS_EDGE=HAS_EDGE,
        )
        weight_kernel(
            grad_out, n2e_index, n_ext2e_index,
            grad_grad_node if HAS_NODE else node_ebd,
            grad_grad_node_ext if HAS_EXT else node_ebd_ext,
            grad_grad_edge_ebd if HAS_EDGE else flat_edge_ebd, workspace,
        )
        reduce_kernel_v2 = fused_edge_update_double_backward_weights_v2_reduce(
            K=K, D_edge=D_edge, D_node=D_node, D_ext=D_ext,
            SPLIT_M=split_m, accum_dtype=dtype,
        )
        reduce_kernel_v2(
            workspace, grad_grad_node_weight_input,
            grad_grad_node_ext_weight_input, grad_grad_edge_weight_input,
        )

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
    BLK_M: int = 32,
    BLK_N: int = 32,
    BLK_K: int = 32,
):
    @T.prim_func
    def angle_forward(
        flat_angle_ebd: T.Tensor((N_ANGLE, ANGLE_DIM), "float32"),
        node_ebd: T.Tensor((N_NODE, NODE_DIM), "float32"),
        flat_edge_ebd: T.Tensor((N_EDGE, EDGE_DIM), "float32"),
        n2a_index: T.Tensor((N_ANGLE,), "int64"),
        eij2a_index: T.Tensor((N_ANGLE,), "int64"),
        eik2a_index: T.Tensor((N_ANGLE,), "int64"),
        angle_weight: T.Tensor((ANGLE_DIM, OUT_DIM), "float32"),
        node_weight: T.Tensor((NODE_DIM, OUT_DIM), "float32"),
        edge_ik_weight: T.Tensor((EDGE_DIM, OUT_DIM), "float32"),
        edge_ij_weight: T.Tensor((EDGE_DIM, OUT_DIM), "float32"),
        bias: T.Tensor((OUT_DIM,), "float32"),
        out: T.Tensor((N_ANGLE, OUT_DIM), "float32"),
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
    return angle_forward

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
def fused_angle_update_backward_inputs(
    M,
    K,
    A,
    N,
    EK,
    N_NODE,
    N_EDGE,
    dtype="float32",
    accum_dtype="float32",
    THREADS=128,
    BLOCK_M=32,
    BLOCK_D=16,
    BLOCK_K=32,
):
    """Tiled activation VJPs; two shared tiles reused by four GEMMs."""
    MAX_D = max(A, N, EK)

    @T.prim_func
    def angle_backward_inputs(
        grad_output: T.Tensor((M, K), dtype),
        n2a_index: T.Tensor((M,), "int64"),
        eij2a_index: T.Tensor((M,), "int64"),
        eik2a_index: T.Tensor((M,), "int64"),
        sub_angle: T.Tensor((A, K), dtype),
        sub_node: T.Tensor((N, K), dtype),
        sub_edge_ik: T.Tensor((EK, K), dtype),
        sub_edge_ij: T.Tensor((EK, K), dtype),
        grad_flat_angle: T.Tensor((M, A), accum_dtype),
        grad_flat_node: T.Tensor((N_NODE, N), accum_dtype),
        grad_flat_edge: T.Tensor((N_EDGE, EK), accum_dtype),
        grad_bias: T.Tensor((K,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(MAX_D, BLOCK_D), threads=THREADS) as (bx, by):
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            weight_shared = T.alloc_shared((BLOCK_K, BLOCK_D), dtype)
            acc_angle = T.alloc_fragment((BLOCK_M, BLOCK_D), accum_dtype)
            T.clear(acc_angle)
            acc_node = T.alloc_fragment((BLOCK_M, BLOCK_D), accum_dtype)
            T.clear(acc_node)
            acc_ik = T.alloc_fragment((BLOCK_M, BLOCK_D), accum_dtype)
            T.clear(acc_ik)
            acc_ij = T.alloc_fragment((BLOCK_M, BLOCK_D), accum_dtype)
            T.clear(acc_ij)
            bias_tile = T.alloc_fragment((BLOCK_K,), accum_dtype)

            # Independent angle pipeline: reload grad_output for this GEMM.
            for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = bx * BLOCK_M + mi
                    k = ko * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                for ki, di in T.Parallel(BLOCK_K, BLOCK_D):
                    k = ko * BLOCK_K + ki
                    d = by * BLOCK_D + di
                    if k < K and d < A:
                        weight_shared[ki, di] = sub_angle[d, k]
                    else:
                        weight_shared[ki, di] = 0
                T.sync_threads()
                if by * BLOCK_D < A:
                    T.gemm(grad_shared, weight_shared, acc_angle)
                T.sync_threads()
            T.sync_threads()

            # Independent node pipeline: reload grad_output for this GEMM.
            for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = bx * BLOCK_M + mi
                    k = ko * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                for ki, di in T.Parallel(BLOCK_K, BLOCK_D):
                    k = ko * BLOCK_K + ki
                    d = by * BLOCK_D + di
                    if k < K and d < N:
                        weight_shared[ki, di] = sub_node[d, k]
                    else:
                        weight_shared[ki, di] = 0
                T.sync_threads()
                if by * BLOCK_D < N:
                    T.gemm(grad_shared, weight_shared, acc_node)
                T.sync_threads()
            T.sync_threads()

            # Independent ik pipeline: reload grad_output for this GEMM.
            for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = bx * BLOCK_M + mi
                    k = ko * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                for ki, di in T.Parallel(BLOCK_K, BLOCK_D):
                    k = ko * BLOCK_K + ki
                    d = by * BLOCK_D + di
                    if k < K and d < EK:
                        weight_shared[ki, di] = sub_edge_ik[d, k]
                    else:
                        weight_shared[ki, di] = 0
                T.sync_threads()
                if by * BLOCK_D < EK:
                    T.gemm(grad_shared, weight_shared, acc_ik)
                T.sync_threads()
            T.sync_threads()

            # Independent ij pipeline: reload grad_output for this GEMM.
            for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = bx * BLOCK_M + mi
                    k = ko * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                for ki, di in T.Parallel(BLOCK_K, BLOCK_D):
                    k = ko * BLOCK_K + ki
                    d = by * BLOCK_D + di
                    if k < K and d < EK:
                        weight_shared[ki, di] = sub_edge_ij[d, k]
                    else:
                        weight_shared[ki, di] = 0
                T.sync_threads()
                if by * BLOCK_D < EK:
                    T.gemm(grad_shared, weight_shared, acc_ij)
                T.sync_threads()
            T.sync_threads()

            # Bias is reduced once, outside the four GEMM pipelines.
            # Read global memory here: a pipelined shared tile must not be
            # consumed after the loop that versions it.
            if by == 0:
                for ko in T.serial(T.ceildiv(K, BLOCK_K)):
                    T.clear(bias_tile)
                    for ki in T.Parallel(BLOCK_K):
                        k = ko * BLOCK_K + ki
                        if k < K:
                            for mi in T.serial(BLOCK_M):
                                m = bx * BLOCK_M + mi
                                if m < M:
                                    bias_tile[ki] += T.cast(grad_output[m, k], accum_dtype)
                    for ki in T.Parallel(BLOCK_K):
                        k = ko * BLOCK_K + ki
                        if k < K:
                            T.atomic_add(grad_bias[k], bias_tile[ki])

            # Separate epilogues: reductions are complete before global writes.
            for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                m = bx * BLOCK_M + mi
                d = by * BLOCK_D + di
                if m < M and d < A:
                    grad_flat_angle[m, d] = acc_angle[mi, di]
            for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                m = bx * BLOCK_M + mi
                d = by * BLOCK_D + di
                if m < M and d < N:
                    T.atomic_add(grad_flat_node[n2a_index[m], d], acc_node[mi, di])
            for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                m = bx * BLOCK_M + mi
                d = by * BLOCK_D + di
                if m < M and d < EK:
                    T.atomic_add(grad_flat_edge[eik2a_index[m], d], acc_ik[mi, di])
                    T.atomic_add(grad_flat_edge[eij2a_index[m], d], acc_ij[mi, di])

    return angle_backward_inputs


@tilelang.jit
def fused_angle_update_backward_inputs_v2(
    M,
    K,
    A,
    N,
    EK,
    N_NODE,
    N_EDGE,
    dtype="float32",
    accum_dtype="float32",
    THREADS=128,
    BLOCK_M=32,
    BLOCK_D=32,
    BLOCK_K=32,
):
    """Compute G @ [W_angle; W_node; W_ik; W_ij].T as logical D tiles.

    The concatenated output is never materialized. Each block owns one
    ``BLOCK_M x BLOCK_D`` tile and routes it directly to the four destination
    tensors. Node and edge destinations must be zero-initialized because their
    gather adjoints are scatter-adds.
    """
    D = A + N + 2 * EK

    @T.prim_func
    def angle_backward_inputs(
        grad_output: T.Tensor((M, K), dtype),
        n2a_index: T.Tensor((M,), "int64"),
        eij2a_index: T.Tensor((M,), "int64"),
        eik2a_index: T.Tensor((M,), "int64"),
        sub_angle: T.Tensor((A, K), dtype),
        sub_node: T.Tensor((N, K), dtype),
        sub_edge_ik: T.Tensor((EK, K), dtype),
        sub_edge_ij: T.Tensor((EK, K), dtype),
        grad_flat_angle: T.Tensor((M, A), accum_dtype),
        grad_flat_node: T.Tensor((N_NODE, N), accum_dtype),
        grad_flat_edge: T.Tensor((N_EDGE, EK), accum_dtype),
        grad_bias: T.Tensor((K,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(D, BLOCK_D), threads=THREADS) as (bx, by):
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            weight_shared = T.alloc_shared((BLOCK_D, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_M, BLOCK_D), accum_dtype)
            bias_tile = T.alloc_fragment((BLOCK_K,), accum_dtype)
            T.clear(acc)

            for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = bx * BLOCK_M + mi
                    k = ko * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0

                # Keep the v1 shared-memory orientation in v2 so this version
                # isolates logical concatenation and the single accumulator.
                for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                    k = ko * BLOCK_K + ki
                    d = by * BLOCK_D + di
                    if k < K and d < D:
                        if d < A:
                            weight_shared[di, ki] = sub_angle[d, k]
                        elif d < A + N:
                            weight_shared[di, ki] = sub_node[d - A, k]
                        elif d < A + N + EK:
                            weight_shared[di, ki] = sub_edge_ik[d - A - N, k]
                        else:
                            weight_shared[di, ki] = sub_edge_ij[d - A - N - EK, k]
                    else:
                        weight_shared[di, ki] = 0
                T.sync_threads()
                T.gemm(grad_shared, weight_shared, acc, transpose_B=True)
                T.sync_threads()

            # Bias is independent of D. Exactly one D tile per M tile reduces
            # it, preserving the v1 launch-wide scatter-reduction semantics.
            if by == 0:
                for ko in T.serial(T.ceildiv(K, BLOCK_K)):
                    T.clear(bias_tile)
                    for ki in T.Parallel(BLOCK_K):
                        k = ko * BLOCK_K + ki
                        if k < K:
                            for mi in T.serial(BLOCK_M):
                                m = bx * BLOCK_M + mi
                                if m < M:
                                    bias_tile[ki] += T.cast(grad_output[m, k], accum_dtype)
                    for ki in T.Parallel(BLOCK_K):
                        k = ko * BLOCK_K + ki
                        if k < K:
                            T.atomic_add(grad_bias[k], bias_tile[ki])

            for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                m = bx * BLOCK_M + mi
                d = by * BLOCK_D + di
                if m < M and d < D:
                    if d < A:
                        grad_flat_angle[m, d] = acc[mi, di]
                    elif d < A + N:
                        T.atomic_add(grad_flat_node[n2a_index[m], d - A], acc[mi, di])
                    elif d < A + N + EK:
                        T.atomic_add(grad_flat_edge[eik2a_index[m], d - A - N], acc[mi, di])
                    else:
                        T.atomic_add(grad_flat_edge[eij2a_index[m], d - A - N - EK], acc[mi, di])

    return angle_backward_inputs

@tilelang.jit
def fused_angle_update_backward_weights(
    M,
    K,
    A,
    N,
    EK,
    N_NODE,
    N_EDGE,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_D=32,
    BLOCK_K=16,
    THREADS=128,
    BLOCK_M=32,
):
    """Four independent GEMM pipelines sharing two scratch allocations.

    Load features in row-major order; GEMM performs X.T @ grad_output.
    Keeping the inner load dimension contiguous avoids transposed gathers
    across the angle-index dimension during vectorization planning.
    """
    MAX_D = max(A, N, EK)

    @T.prim_func
    def backward_weights(
        grad_output: T.Tensor((M, K), dtype),
        flat_angle_ebd: T.Tensor((M, A), dtype),
        flat_node_ebd: T.Tensor((N_NODE, N), dtype),
        flat_edge_ebd: T.Tensor((N_EDGE, EK), dtype),
        n2a_index: T.Tensor((M,), "int64"),
        eij2a_index: T.Tensor((M,), "int64"),
        eik2a_index: T.Tensor((M,), "int64"),
        grad_sub_angle: T.Tensor((A, K), accum_dtype),
        grad_sub_node: T.Tensor((N, K), accum_dtype),
        grad_sub_edge_ik: T.Tensor((EK, K), accum_dtype),
        grad_sub_edge_ij: T.Tensor((EK, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(MAX_D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=THREADS) as (bx, by):
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            # Original: feature_shared = T.alloc_shared((BLOCK_D, BLOCK_M), dtype)
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            acc_angle = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc_angle)
            acc_node = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc_node)
            acc_ik = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc_ik)
            acc_ij = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc_ij)

            # Independent angle pipeline: reload grad_output for this GEMM.
            for mo in T.Pipelined(T.ceildiv(M, BLOCK_M), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = mo * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                # Original: for di, mi in T.Parallel(BLOCK_D, BLOCK_M):
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    d = bx * BLOCK_D + di
                    m = mo * BLOCK_M + mi
                    if d < A and m < M:
                        # Original: feature_shared[di, mi] = flat_angle_ebd[m, d]
                        feature_shared[mi, di] = flat_angle_ebd[m, d]
                    else:
                        # Original: feature_shared[di, mi] = 0
                        feature_shared[mi, di] = 0
                T.sync_threads()
                if bx * BLOCK_D < A:
                    # Original: T.gemm(feature_shared, grad_shared, acc_angle)
                    T.gemm(feature_shared, grad_shared, acc_angle, transpose_A=True)
                T.sync_threads()
            T.sync_threads()

            # Independent node pipeline: reload grad_output for this GEMM.
            for mo in T.Pipelined(T.ceildiv(M, BLOCK_M), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = mo * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                # Original: for di, mi in T.Parallel(BLOCK_D, BLOCK_M):
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    d = bx * BLOCK_D + di
                    m = mo * BLOCK_M + mi
                    if d < N and m < M:
                        # Original: feature_shared[di, mi] = flat_node_ebd[n2a_index[m], d]
                        feature_shared[mi, di] = flat_node_ebd[n2a_index[m], d]
                    else:
                        # Original: feature_shared[di, mi] = 0
                        feature_shared[mi, di] = 0
                T.sync_threads()
                if bx * BLOCK_D < N:
                    # Original: T.gemm(feature_shared, grad_shared, acc_node)
                    T.gemm(feature_shared, grad_shared, acc_node, transpose_A=True)
                T.sync_threads()
            T.sync_threads()

            # Independent ik pipeline: reload grad_output for this GEMM.
            for mo in T.Pipelined(T.ceildiv(M, BLOCK_M), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = mo * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                # Original: for di, mi in T.Parallel(BLOCK_D, BLOCK_M):
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    d = bx * BLOCK_D + di
                    m = mo * BLOCK_M + mi
                    if d < EK and m < M:
                        # Original: feature_shared[di, mi] = flat_edge_ebd[eik2a_index[m], d]
                        feature_shared[mi, di] = flat_edge_ebd[eik2a_index[m], d]
                    else:
                        # Original: feature_shared[di, mi] = 0
                        feature_shared[mi, di] = 0
                T.sync_threads()
                if bx * BLOCK_D < EK:
                    # Original: T.gemm(feature_shared, grad_shared, acc_ik)
                    T.gemm(feature_shared, grad_shared, acc_ik, transpose_A=True)
                T.sync_threads()
            T.sync_threads()

            # Independent ij pipeline: reload grad_output for this GEMM.
            for mo in T.Pipelined(T.ceildiv(M, BLOCK_M), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = mo * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                # Original: for di, mi in T.Parallel(BLOCK_D, BLOCK_M):
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    d = bx * BLOCK_D + di
                    m = mo * BLOCK_M + mi
                    if d < EK and m < M:
                        # Original: feature_shared[di, mi] = flat_edge_ebd[eij2a_index[m], d]
                        feature_shared[mi, di] = flat_edge_ebd[eij2a_index[m], d]
                    else:
                        # Original: feature_shared[di, mi] = 0
                        feature_shared[mi, di] = 0
                T.sync_threads()
                if bx * BLOCK_D < EK:
                    # Original: T.gemm(feature_shared, grad_shared, acc_ij)
                    T.gemm(feature_shared, grad_shared, acc_ij, transpose_A=True)
                T.sync_threads()
            T.sync_threads()

            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < A and k < K:
                    grad_sub_angle[d, k] = acc_angle[di, ki]
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < N and k < K:
                    grad_sub_node[d, k] = acc_node[di, ki]
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < EK and k < K:
                    grad_sub_edge_ik[d, k] = acc_ik[di, ki]
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < EK and k < K:
                    grad_sub_edge_ij[d, k] = acc_ij[di, ki]

    return backward_weights


@tilelang.jit
def fused_angle_update_backward_weights_v2(
    M, K, A, N, EK, N_NODE, N_EDGE,
    dtype="float32", accum_dtype="float32",
    BLOCK_D=32, BLOCK_K=16, THREADS=128, BLOCK_M=32, SPLIT_M=8,
):
    """Gather a logical feature concatenation and compute split-M partials.

    workspace[s] = X_s.T @ G_s, X = [angle, node[index], ik[index], ij[index]].
    SPLIT_M partitions whole BLOCK_M tiles of the reduction dimension M.
    Empty partitions explicitly write zeros. A second kernel reduces workspace.
    No full gathered/concatenated input is materialized in global memory.
    """
    assert SPLIT_M > 0
    D = A + N + 2 * EK
    TILES_PER_SPLIT = ((M + BLOCK_M - 1) // BLOCK_M + SPLIT_M - 1) // SPLIT_M

    @T.prim_func
    def angle_backward_weight_partials(
        grad_output: T.Tensor((M, K), dtype),
        flat_angle_ebd: T.Tensor((M, A), dtype),
        flat_node_ebd: T.Tensor((N_NODE, N), dtype),
        flat_edge_ebd: T.Tensor((N_EDGE, EK), dtype),
        n2a_index: T.Tensor((M,), "int64"),
        eij2a_index: T.Tensor((M,), "int64"),
        eik2a_index: T.Tensor((M,), "int64"),
        workspace: T.Tensor((SPLIT_M, D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), SPLIT_M, threads=THREADS) as (bx, by, bs):
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc)
            for mo in T.Pipelined(TILES_PER_SPLIT, num_stages=2):
                # Row-major shared layout; GEMM handles the transpose.
                # Element guards also support feature tiles crossing segments.
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                    d = bx * BLOCK_D + di
                    if m < M and d < D:
                        if d < A:
                            feature_shared[mi, di] = flat_angle_ebd[m, d]
                        elif d < A + N:
                            feature_shared[mi, di] = flat_node_ebd[n2a_index[m], d - A]
                        elif d < A + N + EK:
                            feature_shared[mi, di] = flat_edge_ebd[eik2a_index[m], d - A - N]
                        else:
                            feature_shared[mi, di] = flat_edge_ebd[eij2a_index[m], d - A - N - EK]
                    else:
                        feature_shared[mi, di] = 0
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    workspace[bs, d, k] = acc[di, ki]
    return angle_backward_weight_partials


@tilelang.jit
def fused_angle_update_backward_weights_v3_1(
    M, K, A, N, EK, N_NODE, N_EDGE,
    dtype="float32", accum_dtype="float32",
    BLOCK_D=32, BLOCK_K=16, THREADS=128, BLOCK_M=32, SPLIT_M=8,
):
    """V2 with a custom 16-byte-granular XOR layout for the feature tile.

    For FP32, four adjacent elements form one 16-byte group.  The physical
    layout exposes that group explicitly as ``[row, float4_group, lane]`` and
    applies XOR only to ``float4_group``.  This is equivalent to XORing the
    scalar column by an aligned eight-float offset, but keeps ``lane`` as a
    provably contiguous innermost dimension for vectorized loads/stores.
    """
    assert SPLIT_M > 0
    assert dtype == "float32"
    assert BLOCK_D == 32
    assert BLOCK_M == 32
    assert THREADS == 128
    assert A % 4 == 0
    assert N % 4 == 0
    assert EK % 4 == 0
    D = A + N + 2 * EK
    TILES_PER_SPLIT = ((M + BLOCK_M - 1) // BLOCK_M + SPLIT_M - 1) // SPLIT_M

    @T.prim_func
    def angle_backward_weight_partials(
        grad_output: T.Tensor((M, K), dtype),
        flat_angle_ebd: T.Tensor((M, A), dtype),
        flat_node_ebd: T.Tensor((N_NODE, N), dtype),
        flat_edge_ebd: T.Tensor((N_EDGE, EK), dtype),
        n2a_index: T.Tensor((M,), "int64"),
        eij2a_index: T.Tensor((M,), "int64"),
        eik2a_index: T.Tensor((M,), "int64"),
        workspace: T.Tensor((SPLIT_M, D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), SPLIT_M, threads=THREADS) as (bx, by, bs):
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            # Decompose the physical column into a float4 group and its lane.
            #
            #   di ^ ((mi % 4) * 8)
            # == ((di // 4) ^ ((mi % 4) * 2)) * 4 + di % 4
            #
            # The two forms have the same linear shared-memory address.  The
            # decomposed form makes the vectorized `vi` appear as the final
            # physical dimension instead of keeping it inside an XOR node.
            T.annotate_layout({
                feature_shared: T.Layout(
                    (BLOCK_M, BLOCK_D),
                    lambda mi, di: [
                        mi,
                        (di // 4) ^ ((mi % 4) * 2),
                        di % 4,
                    ],
                ),
            })
            T.clear(acc)
            for mo in T.Pipelined(TILES_PER_SPLIT, num_stages=2):
                # Schedule one complete 16-row half tile at a time.  Each
                # T.Parallel(16, 8) has exactly 128 float4 tasks, so within a
                # warp di4 runs through all eight groups for four complete
                # rows.  This keeps both the global LDG.128 sectors and the
                # shared STS.128 bank wavefronts contiguous/minimal.
                for m_half in T.unroll(2):
                    for mi_inner, di4 in T.Parallel(BLOCK_M // 2, BLOCK_D // 4):
                        mi = m_half * (BLOCK_M // 2) + mi_inner
                        m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                        d_base = bx * BLOCK_D + di4 * 4
                        if m < M and d_base < D:
                            if d_base < A:
                                for vi in T.vectorized(4):
                                    feature_shared[mi, di4 * 4 + vi] = flat_angle_ebd[m, d_base + vi]
                            elif d_base < A + N:
                                for vi in T.vectorized(4):
                                    feature_shared[mi, di4 * 4 + vi] = flat_node_ebd[n2a_index[m], d_base - A + vi]
                            elif d_base < A + N + EK:
                                for vi in T.vectorized(4):
                                    feature_shared[mi, di4 * 4 + vi] = flat_edge_ebd[eik2a_index[m], d_base - A - N + vi]
                            else:
                                for vi in T.vectorized(4):
                                    feature_shared[mi, di4 * 4 + vi] = flat_edge_ebd[eij2a_index[m], d_base - A - N - EK + vi]
                        else:
                            for vi in T.vectorized(4):
                                feature_shared[mi, di4 * 4 + vi] = 0
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    workspace[bs, d, k] = acc[di, ki]
    return angle_backward_weight_partials


@tilelang.jit
def fused_angle_update_backward_weights_v3_2(
    M, K, A, N, EK, N_NODE, N_EDGE,
    dtype="float32", accum_dtype="float32",
    BLOCK_D=32, BLOCK_K=16, THREADS=128, BLOCK_M=32, SPLIT_M=8,
):
    """V3.1 plus a conflict-free float4-group layout for the B operand.

    ``feature_shared`` keeps the V3.1 producer schedule and XOR layout.
    ``grad_shared`` XORs its four float4 column groups with the low two
    row bits.  The latter maps each MMA B load across all 32 banks while
    preserving a contiguous, 16-byte-aligned lane dimension for LDG/STS.
    """
    assert SPLIT_M > 0
    assert dtype == "float32"
    assert BLOCK_D == 32
    assert BLOCK_K == 16
    assert BLOCK_M == 32
    assert THREADS == 128
    assert A % 4 == 0
    assert N % 4 == 0
    assert EK % 4 == 0
    assert K % 4 == 0
    D = A + N + 2 * EK
    TILES_PER_SPLIT = ((M + BLOCK_M - 1) // BLOCK_M + SPLIT_M - 1) // SPLIT_M

    @T.prim_func
    def angle_backward_weight_partials(
        grad_output: T.Tensor((M, K), dtype),
        flat_angle_ebd: T.Tensor((M, A), dtype),
        flat_node_ebd: T.Tensor((N_NODE, N), dtype),
        flat_edge_ebd: T.Tensor((N_EDGE, EK), dtype),
        n2a_index: T.Tensor((M,), "int64"),
        eij2a_index: T.Tensor((M,), "int64"),
        eik2a_index: T.Tensor((M,), "int64"),
        workspace: T.Tensor((SPLIT_M, D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), SPLIT_M, threads=THREADS) as (bx, by, bs):
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.annotate_layout({
                feature_shared: T.Layout(
                    (BLOCK_M, BLOCK_D),
                    lambda mi, di: [
                        mi,
                        (di // 4) ^ ((mi % 4) * 2),
                        di % 4,
                    ],
                ),
                grad_shared: T.Layout(
                    (BLOCK_M, BLOCK_K),
                    lambda mi, ki: [
                        mi,
                        (ki // 4) ^ (mi % 4),
                        ki % 4,
                    ],
                ),
            })
            T.clear(acc)
            for mo in T.Pipelined(TILES_PER_SPLIT, num_stages=2):
                # Each pass assigns one complete 16-row half tile to the
                # block: row = threadIdx.x // 8, float4 group = threadIdx.x % 8.
                for m_half in T.unroll(2):
                    for mi_inner, di4 in T.Parallel(BLOCK_M // 2, BLOCK_D // 4):
                        mi = m_half * (BLOCK_M // 2) + mi_inner
                        m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                        d_base = bx * BLOCK_D + di4 * 4
                        if m < M and d_base < D:
                            if d_base < A:
                                for vi in T.vectorized(4):
                                    feature_shared[mi, di4 * 4 + vi] = flat_angle_ebd[m, d_base + vi]
                            elif d_base < A + N:
                                for vi in T.vectorized(4):
                                    feature_shared[mi, di4 * 4 + vi] = flat_node_ebd[n2a_index[m], d_base - A + vi]
                            elif d_base < A + N + EK:
                                for vi in T.vectorized(4):
                                    feature_shared[mi, di4 * 4 + vi] = flat_edge_ebd[eik2a_index[m], d_base - A - N + vi]
                            else:
                                for vi in T.vectorized(4):
                                    feature_shared[mi, di4 * 4 + vi] = flat_edge_ebd[eij2a_index[m], d_base - A - N - EK + vi]
                        else:
                            for vi in T.vectorized(4):
                                feature_shared[mi, di4 * 4 + vi] = 0
                # One thread owns one full float4.  Keeping the group and lane
                # explicit prevents the custom B layout from scalarizing this
                # global-to-shared producer.
                for mi, ki4 in T.Parallel(BLOCK_M, BLOCK_K // 4):
                    m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                    k_base = by * BLOCK_K + ki4 * 4
                    if m < M and k_base < K:
                        for vi in T.vectorized(4):
                            grad_shared[mi, ki4 * 4 + vi] = grad_output[m, k_base + vi]
                    else:
                        for vi in T.vectorized(4):
                            grad_shared[mi, ki4 * 4 + vi] = 0
                T.sync_threads()
                T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    workspace[bs, d, k] = acc[di, ki]
    return angle_backward_weight_partials


@tilelang.jit
def fused_angle_update_backward_weights_v3_3(
    M, K, A, N, EK, N_NODE, N_EDGE,
    dtype="float32", accum_dtype="float32",
    BLOCK_D=32, BLOCK_K=32, THREADS=256, BLOCK_M=32, SPLIT_M=8,
):
    """V3.2 with a 32-column B tile and one float4 task per thread.

    Both shared-memory producers contain exactly 256 float4 groups:

      feature: 32 rows * (32 columns / 4) = 256 groups
      gradient: 32 rows * (32 columns / 4) = 256 groups

    With 256 threads, ``T.Parallel(32, 8)`` therefore maps one complete
    float4 to each thread.  A warp covers all eight float4 groups of four
    complete rows, so the V3.1 16-row half-tile schedule is unnecessary.

    The B layout extends the V3.2 four-group swizzle to eight groups.  XORing
    by ``(row % 4) * 2`` selects offsets 0, 2, 4, and 6 in float4 units
    (0, 8, 16, and 24 FP32 words), while the innermost four lanes remain
    contiguous and 16-byte aligned.
    """
    assert SPLIT_M > 0
    assert dtype == "float32"
    assert BLOCK_D == 32
    assert BLOCK_K == 32
    assert BLOCK_M == 32
    assert THREADS == 256
    assert A % 4 == 0
    assert N % 4 == 0
    assert EK % 4 == 0
    assert K % 4 == 0
    D = A + N + 2 * EK
    TILES_PER_SPLIT = ((M + BLOCK_M - 1) // BLOCK_M + SPLIT_M - 1) // SPLIT_M

    @T.prim_func
    def angle_backward_weight_partials(
        grad_output: T.Tensor((M, K), dtype),
        flat_angle_ebd: T.Tensor((M, A), dtype),
        flat_node_ebd: T.Tensor((N_NODE, N), dtype),
        flat_edge_ebd: T.Tensor((N_EDGE, EK), dtype),
        n2a_index: T.Tensor((M,), "int64"),
        eij2a_index: T.Tensor((M,), "int64"),
        eik2a_index: T.Tensor((M,), "int64"),
        workspace: T.Tensor((SPLIT_M, D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), SPLIT_M, threads=THREADS) as (bx, by, bs):
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.annotate_layout({
                feature_shared: T.Layout(
                    (BLOCK_M, BLOCK_D),
                    lambda mi, di: [
                        mi,
                        (di // 4) ^ ((mi % 4) * 2),
                        di % 4,
                    ],
                ),
                grad_shared: T.Layout(
                    (BLOCK_M, BLOCK_K),
                    lambda mi, ki: [
                        mi,
                        (ki // 4) ^ ((mi % 4) * 2),
                        ki % 4,
                    ],
                ),
            })
            T.clear(acc)
            for mo in T.Pipelined(TILES_PER_SPLIT, num_stages=2):
                # There are 32 * 8 = 256 float4 tasks.  With 256 threads,
                # threadIdx.x // 8 selects the row and threadIdx.x % 8 the
                # group, so every warp covers four complete 128-byte rows.
                for mi, di4 in T.Parallel(BLOCK_M, BLOCK_D // 4):
                    m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                    d_base = bx * BLOCK_D + di4 * 4
                    if m < M and d_base < D:
                        if d_base < A:
                            for vi in T.vectorized(4):
                                feature_shared[mi, di4 * 4 + vi] = flat_angle_ebd[m, d_base + vi]
                        elif d_base < A + N:
                            for vi in T.vectorized(4):
                                feature_shared[mi, di4 * 4 + vi] = flat_node_ebd[n2a_index[m], d_base - A + vi]
                        elif d_base < A + N + EK:
                            for vi in T.vectorized(4):
                                feature_shared[mi, di4 * 4 + vi] = flat_edge_ebd[eik2a_index[m], d_base - A - N + vi]
                        else:
                            for vi in T.vectorized(4):
                                feature_shared[mi, di4 * 4 + vi] = flat_edge_ebd[eij2a_index[m], d_base - A - N - EK + vi]
                    else:
                        for vi in T.vectorized(4):
                            feature_shared[mi, di4 * 4 + vi] = 0

                # BLOCK_K=32 also gives exactly 32 * 8 = 256 float4 tasks.
                for mi, ki4 in T.Parallel(BLOCK_M, BLOCK_K // 4):
                    m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                    k_base = by * BLOCK_K + ki4 * 4
                    if m < M and k_base < K:
                        for vi in T.vectorized(4):
                            grad_shared[mi, ki4 * 4 + vi] = grad_output[m, k_base + vi]
                    else:
                        for vi in T.vectorized(4):
                            grad_shared[mi, ki4 * 4 + vi] = 0
                T.sync_threads()
                T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    workspace[bs, d, k] = acc[di, ki]
    return angle_backward_weight_partials


@tilelang.jit
def fused_angle_update_backward_weights_v2_reduce(
    K, A, N, EK, SPLIT_M=8, accum_dtype="float32",
    BLOCK_D=32, BLOCK_K=16, THREADS=128,
):
    """Sum partials in increasing split order and write four contiguous outputs.

    No atomic additions: each output element has a unique writer and an explicit
    serial accumulation order. This does not change the GEMM operand precision.
    """
    D = A + N + 2 * EK

    @T.prim_func
    def angle_backward_reduce_partials(
        workspace: T.Tensor((SPLIT_M, D, K), accum_dtype),
        grad_sub_angle: T.Tensor((A, K), accum_dtype),
        grad_sub_node: T.Tensor((N, K), accum_dtype),
        grad_sub_edge_ik: T.Tensor((EK, K), accum_dtype),
        grad_sub_edge_ij: T.Tensor((EK, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=THREADS) as (bx, by):
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc)
            for split in T.serial(SPLIT_M):
                for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                    d = bx * BLOCK_D + di
                    k = by * BLOCK_K + ki
                    if d < D and k < K:
                        acc[di, ki] += workspace[split, d, k]
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    if d < A:
                        grad_sub_angle[d, k] = acc[di, ki]
                    elif d < A + N:
                        grad_sub_node[d - A, k] = acc[di, ki]
                    elif d < A + N + EK:
                        grad_sub_edge_ik[d - A - N, k] = acc[di, ki]
                    else:
                        grad_sub_edge_ij[d - A - N - EK, k] = acc[di, ki]
    return angle_backward_reduce_partials


@tilelang.jit
def fused_edge_update_double_backward_weights_v2(
    E, K, D_edge, D_node, D_ext, N_node, N_ext,
    dtype="float32", accum_dtype="float32",
    BLOCK_D=32, BLOCK_K=16, THREADS=128, BLOCK_M=32, SPLIT_M=4,
    HAS_NODE=True, HAS_EXT=True, HAS_EDGE=True,
):
    """Gather a logical feature concatenation and compute split-E partials.

    workspace[s] = X_s.T @ G_s, X = [U_node[n2e], U_ext[n_ext2e], U_edge].
    SPLIT_M partitions whole BLOCK_M tiles of the reduction dimension E.
    Empty partitions explicitly write zeros. A second kernel reduces workspace.
    No full gathered/concatenated input is materialized in global memory.
    """
    assert SPLIT_M > 0
    D = D_node + D_ext + D_edge
    TILES_PER_SPLIT = ((E + BLOCK_M - 1) // BLOCK_M + SPLIT_M - 1) // SPLIT_M

    @T.prim_func
    def edge_double_backward_weight_partials(
        grad_out: T.Tensor((E, K), dtype),
        n2e_index: T.Tensor((E,), "int64"),
        n_ext2e_index: T.Tensor((E,), "int64"),
        grad_grad_node: T.Tensor((N_node, D_node), dtype),
        grad_grad_node_ext: T.Tensor((N_ext, D_ext), dtype),
        grad_grad_edge_ebd: T.Tensor((E, D_edge), dtype),
        workspace: T.Tensor((SPLIT_M, D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), SPLIT_M, threads=THREADS) as (bx, by, bs):
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc)
            if (HAS_NODE and bx * BLOCK_D < D_node) or (HAS_EXT and bx * BLOCK_D < D_node + D_ext and (bx + 1) * BLOCK_D > D_node) or (HAS_EDGE and (bx + 1) * BLOCK_D > D_node + D_ext):
                for mo in T.Pipelined(TILES_PER_SPLIT, num_stages=2):
                    # Row-major shared layout; GEMM handles the transpose.
                    # Element guards also support feature tiles crossing segments.
                    for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                        m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                        d = bx * BLOCK_D + di
                        if m < E and d < D:
                            if d < D_node:
                                if HAS_NODE:
                                    feature_shared[mi, di] = grad_grad_node[n2e_index[m], d]
                                else:
                                    feature_shared[mi, di] = 0
                            elif d < D_node + D_ext:
                                if HAS_EXT:
                                    feature_shared[mi, di] = grad_grad_node_ext[n_ext2e_index[m], d - D_node]
                                else:
                                    feature_shared[mi, di] = 0
                            else:
                                if HAS_EDGE:
                                    feature_shared[mi, di] = grad_grad_edge_ebd[m, d - D_node - D_ext]
                                else:
                                    feature_shared[mi, di] = 0
                        else:
                            feature_shared[mi, di] = 0
                    for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                        m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                        k = by * BLOCK_K + ki
                        if m < E and k < K:
                            grad_shared[mi, ki] = grad_out[m, k]
                        else:
                            grad_shared[mi, ki] = 0
                    T.sync_threads()
                    T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                    T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    workspace[bs, d, k] = acc[di, ki]
    return edge_double_backward_weight_partials


@tilelang.jit
def fused_edge_update_double_backward_weights_v3(
    E, K, D_edge, D_node, D_ext, N_node, N_ext,
    dtype="float32", accum_dtype="float32",
    BLOCK_D=32, BLOCK_K=32, THREADS=256, BLOCK_M=32, SPLIT_M=4,
    HAS_NODE=True, HAS_EXT=True, HAS_EDGE=True,
):
    """V2 with V3.3-style float4 producers and XOR shared layouts."""
    assert SPLIT_M > 0
    assert dtype == "float32"
    assert BLOCK_D == 32
    assert BLOCK_K == 32
    assert BLOCK_M == 32
    assert THREADS == 256
    assert D_node % 4 == 0
    assert D_ext % 4 == 0
    assert D_edge % 4 == 0
    assert K % 4 == 0
    D = D_node + D_ext + D_edge
    TILES_PER_SPLIT = ((E + BLOCK_M - 1) // BLOCK_M + SPLIT_M - 1) // SPLIT_M

    @T.prim_func
    def edge_double_backward_weight_partials(
        grad_out: T.Tensor((E, K), dtype),
        n2e_index: T.Tensor((E,), "int64"),
        n_ext2e_index: T.Tensor((E,), "int64"),
        grad_grad_node: T.Tensor((N_node, D_node), dtype),
        grad_grad_node_ext: T.Tensor((N_ext, D_ext), dtype),
        grad_grad_edge_ebd: T.Tensor((E, D_edge), dtype),
        workspace: T.Tensor((SPLIT_M, D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), SPLIT_M, threads=THREADS) as (bx, by, bs):
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.annotate_layout({
                feature_shared: T.Layout(
                    (BLOCK_M, BLOCK_D),
                    lambda mi, di: [
                        mi,
                        (di // 4) ^ ((mi % 4) * 2),
                        di % 4,
                    ],
                ),
                grad_shared: T.Layout(
                    (BLOCK_M, BLOCK_K),
                    lambda mi, ki: [
                        mi,
                        (ki // 4) ^ ((mi % 4) * 2),
                        ki % 4,
                    ],
                ),
            })
            T.clear(acc)
            if (HAS_NODE and bx * BLOCK_D < D_node) or (HAS_EXT and bx * BLOCK_D < D_node + D_ext and (bx + 1) * BLOCK_D > D_node) or (HAS_EDGE and (bx + 1) * BLOCK_D > D_node + D_ext):
                for mo in T.Pipelined(TILES_PER_SPLIT, num_stages=2):
                    for mi, di4 in T.Parallel(BLOCK_M, BLOCK_D // 4):
                        m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                        d_base = bx * BLOCK_D + di4 * 4
                        if m < E and d_base < D:
                            if d_base < D_node:
                                if HAS_NODE:
                                    for vi in T.vectorized(4):
                                        feature_shared[mi, di4 * 4 + vi] = grad_grad_node[n2e_index[m], d_base + vi]
                                else:
                                    for vi in T.vectorized(4):
                                        feature_shared[mi, di4 * 4 + vi] = 0
                            elif d_base < D_node + D_ext:
                                if HAS_EXT:
                                    for vi in T.vectorized(4):
                                        feature_shared[mi, di4 * 4 + vi] = grad_grad_node_ext[n_ext2e_index[m], d_base - D_node + vi]
                                else:
                                    for vi in T.vectorized(4):
                                        feature_shared[mi, di4 * 4 + vi] = 0
                            else:
                                if HAS_EDGE:
                                    for vi in T.vectorized(4):
                                        feature_shared[mi, di4 * 4 + vi] = grad_grad_edge_ebd[m, d_base - D_node - D_ext + vi]
                                else:
                                    for vi in T.vectorized(4):
                                        feature_shared[mi, di4 * 4 + vi] = 0
                        else:
                            for vi in T.vectorized(4):
                                feature_shared[mi, di4 * 4 + vi] = 0

                    for mi, ki4 in T.Parallel(BLOCK_M, BLOCK_K // 4):
                        m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                        k_base = by * BLOCK_K + ki4 * 4
                        if m < E and k_base < K:
                            for vi in T.vectorized(4):
                                grad_shared[mi, ki4 * 4 + vi] = grad_out[m, k_base + vi]
                        else:
                            for vi in T.vectorized(4):
                                grad_shared[mi, ki4 * 4 + vi] = 0
                    T.sync_threads()
                    T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                    T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    workspace[bs, d, k] = acc[di, ki]
    return edge_double_backward_weight_partials


@tilelang.jit
def fused_edge_update_double_backward_weights_v2_reduce(
    K, D_edge, D_node, D_ext, SPLIT_M=4, accum_dtype="float32",
    BLOCK_D=32, BLOCK_K=16, THREADS=128,
):
    """Sum partials in increasing split order and write three contiguous outputs.

    No atomic additions: each output element has a unique writer and an explicit
    serial accumulation order. This does not change the GEMM operand precision.
    """
    D = D_node + D_ext + D_edge

    @T.prim_func
    def edge_double_backward_reduce_partials(
        workspace: T.Tensor((SPLIT_M, D, K), accum_dtype),
        grad_node_weight: T.Tensor((D_node, K), accum_dtype),
        grad_node_ext_weight: T.Tensor((D_ext, K), accum_dtype),
        grad_edge_weight: T.Tensor((D_edge, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=THREADS) as (bx, by):
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc)
            for split in T.serial(SPLIT_M):
                for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                    d = bx * BLOCK_D + di
                    k = by * BLOCK_K + ki
                    if d < D and k < K:
                        acc[di, ki] += workspace[split, d, k]
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    if d < D_node:
                        grad_node_weight[d, k] = acc[di, ki]
                    elif d < D_node + D_ext:
                        grad_node_ext_weight[d - D_node, k] = acc[di, ki]
                    else:
                        grad_edge_weight[d - D_node - D_ext, k] = acc[di, ki]
    return edge_double_backward_reduce_partials


@tilelang.jit
def fused_angle_update_double_backward_weights_v2(
    M, K, A, N, EK, N_NODE, N_EDGE,
    dtype="float32", accum_dtype="float32",
    BLOCK_D=32, BLOCK_K=16, THREADS=128, BLOCK_M=32, SPLIT_M=8,
    HAS_ANGLE=True, HAS_NODE=True, HAS_EDGE=True,
):
    """Gather a logical feature concatenation and compute split-M partials.

    workspace[s] = X_s.T @ G_s, X = [U_angle, U_node[n2a], U_edge[eik], U_edge[eij]].
    SPLIT_M partitions whole BLOCK_M tiles of the reduction dimension M.
    Empty partitions explicitly write zeros. A second kernel reduces workspace.
    No full gathered/concatenated input is materialized in global memory.
    """
    assert SPLIT_M > 0
    D = A + N + 2 * EK
    TILES_PER_SPLIT = ((M + BLOCK_M - 1) // BLOCK_M + SPLIT_M - 1) // SPLIT_M

    @T.prim_func
    def angle_double_backward_weight_partials(
        grad_output: T.Tensor((M, K), dtype),
        gg_flat_angle: T.Tensor((M, A), dtype),
        gg_flat_node: T.Tensor((N_NODE, N), dtype),
        gg_flat_edge: T.Tensor((N_EDGE, EK), dtype),
        n2a_index: T.Tensor((M,), "int64"),
        eij2a_index: T.Tensor((M,), "int64"),
        eik2a_index: T.Tensor((M,), "int64"),
        workspace: T.Tensor((SPLIT_M, D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), SPLIT_M, threads=THREADS) as (bx, by, bs):
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc)
            if (HAS_ANGLE and bx * BLOCK_D < A) or (HAS_NODE and bx * BLOCK_D < A + N and (bx + 1) * BLOCK_D > A) or (HAS_EDGE and (bx + 1) * BLOCK_D > A + N):
                for mo in T.Pipelined(TILES_PER_SPLIT, num_stages=2):
                    # Row-major shared layout; GEMM handles the transpose.
                    # Element guards also support feature tiles crossing segments.
                    for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                        m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                        d = bx * BLOCK_D + di
                        if m < M and d < D:
                            if d < A:
                                if HAS_ANGLE:
                                    feature_shared[mi, di] = gg_flat_angle[m, d]
                                else:
                                    feature_shared[mi, di] = 0
                            elif d < A + N:
                                if HAS_NODE:
                                    feature_shared[mi, di] = gg_flat_node[n2a_index[m], d - A]
                                else:
                                    feature_shared[mi, di] = 0
                            elif d < A + N + EK:
                                if HAS_EDGE:
                                    feature_shared[mi, di] = gg_flat_edge[eik2a_index[m], d - A - N]
                                else:
                                    feature_shared[mi, di] = 0
                            else:
                                if HAS_EDGE:
                                    feature_shared[mi, di] = gg_flat_edge[eij2a_index[m], d - A - N - EK]
                                else:
                                    feature_shared[mi, di] = 0
                        else:
                            feature_shared[mi, di] = 0
                    for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                        m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                        k = by * BLOCK_K + ki
                        if m < M and k < K:
                            grad_shared[mi, ki] = grad_output[m, k]
                        else:
                            grad_shared[mi, ki] = 0
                    T.sync_threads()
                    T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                    T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    workspace[bs, d, k] = acc[di, ki]
    return angle_double_backward_weight_partials


@tilelang.jit
def fused_angle_update_double_backward_weights_v3(
    M, K, A, N, EK, N_NODE, N_EDGE,
    dtype="float32", accum_dtype="float32",
    BLOCK_D=32, BLOCK_K=32, THREADS=256, BLOCK_M=32, SPLIT_M=8,
    HAS_ANGLE=True, HAS_NODE=True, HAS_EDGE=True,
):
    """V2 with V3.3-style float4 producers and XOR shared layouts."""
    assert SPLIT_M > 0
    assert dtype == "float32"
    assert BLOCK_D == 32
    assert BLOCK_K == 32
    assert BLOCK_M == 32
    assert THREADS == 256
    assert A % 4 == 0
    assert N % 4 == 0
    assert EK % 4 == 0
    assert K % 4 == 0
    D = A + N + 2 * EK
    TILES_PER_SPLIT = ((M + BLOCK_M - 1) // BLOCK_M + SPLIT_M - 1) // SPLIT_M

    @T.prim_func
    def angle_double_backward_weight_partials(
        grad_output: T.Tensor((M, K), dtype),
        gg_flat_angle: T.Tensor((M, A), dtype),
        gg_flat_node: T.Tensor((N_NODE, N), dtype),
        gg_flat_edge: T.Tensor((N_EDGE, EK), dtype),
        n2a_index: T.Tensor((M,), "int64"),
        eij2a_index: T.Tensor((M,), "int64"),
        eik2a_index: T.Tensor((M,), "int64"),
        workspace: T.Tensor((SPLIT_M, D, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), SPLIT_M, threads=THREADS) as (bx, by, bs):
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.annotate_layout({
                feature_shared: T.Layout(
                    (BLOCK_M, BLOCK_D),
                    lambda mi, di: [
                        mi,
                        (di // 4) ^ ((mi % 4) * 2),
                        di % 4,
                    ],
                ),
                grad_shared: T.Layout(
                    (BLOCK_M, BLOCK_K),
                    lambda mi, ki: [
                        mi,
                        (ki // 4) ^ ((mi % 4) * 2),
                        ki % 4,
                    ],
                ),
            })
            T.clear(acc)
            if (HAS_ANGLE and bx * BLOCK_D < A) or (HAS_NODE and bx * BLOCK_D < A + N and (bx + 1) * BLOCK_D > A) or (HAS_EDGE and (bx + 1) * BLOCK_D > A + N):
                for mo in T.Pipelined(TILES_PER_SPLIT, num_stages=2):
                    for mi, di4 in T.Parallel(BLOCK_M, BLOCK_D // 4):
                        m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                        d_base = bx * BLOCK_D + di4 * 4
                        if m < M and d_base < D:
                            if d_base < A:
                                if HAS_ANGLE:
                                    for vi in T.vectorized(4):
                                        feature_shared[mi, di4 * 4 + vi] = gg_flat_angle[m, d_base + vi]
                                else:
                                    for vi in T.vectorized(4):
                                        feature_shared[mi, di4 * 4 + vi] = 0
                            elif d_base < A + N:
                                if HAS_NODE:
                                    for vi in T.vectorized(4):
                                        feature_shared[mi, di4 * 4 + vi] = gg_flat_node[n2a_index[m], d_base - A + vi]
                                else:
                                    for vi in T.vectorized(4):
                                        feature_shared[mi, di4 * 4 + vi] = 0
                            elif d_base < A + N + EK:
                                if HAS_EDGE:
                                    for vi in T.vectorized(4):
                                        feature_shared[mi, di4 * 4 + vi] = gg_flat_edge[eik2a_index[m], d_base - A - N + vi]
                                else:
                                    for vi in T.vectorized(4):
                                        feature_shared[mi, di4 * 4 + vi] = 0
                            else:
                                if HAS_EDGE:
                                    for vi in T.vectorized(4):
                                        feature_shared[mi, di4 * 4 + vi] = gg_flat_edge[eij2a_index[m], d_base - A - N - EK + vi]
                                else:
                                    for vi in T.vectorized(4):
                                        feature_shared[mi, di4 * 4 + vi] = 0
                        else:
                            for vi in T.vectorized(4):
                                feature_shared[mi, di4 * 4 + vi] = 0

                    for mi, ki4 in T.Parallel(BLOCK_M, BLOCK_K // 4):
                        m = (bs * TILES_PER_SPLIT + mo) * BLOCK_M + mi
                        k_base = by * BLOCK_K + ki4 * 4
                        if m < M and k_base < K:
                            for vi in T.vectorized(4):
                                grad_shared[mi, ki4 * 4 + vi] = grad_output[m, k_base + vi]
                        else:
                            for vi in T.vectorized(4):
                                grad_shared[mi, ki4 * 4 + vi] = 0
                    T.sync_threads()
                    T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                    T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    workspace[bs, d, k] = acc[di, ki]
    return angle_double_backward_weight_partials


@tilelang.jit
def fused_angle_update_double_backward_weights_v2_reduce(
    K, A, N, EK, SPLIT_M=8, accum_dtype="float32",
    BLOCK_D=32, BLOCK_K=16, THREADS=128,
):
    """Sum partials in increasing split order and write four contiguous outputs.

    No atomic additions: each output element has a unique writer and an explicit
    serial accumulation order. This does not change the GEMM operand precision.
    """
    D = A + N + 2 * EK

    @T.prim_func
    def angle_double_backward_reduce_partials(
        workspace: T.Tensor((SPLIT_M, D, K), accum_dtype),
        grad_sub_angle: T.Tensor((A, K), accum_dtype),
        grad_sub_node: T.Tensor((N, K), accum_dtype),
        grad_sub_edge_ik: T.Tensor((EK, K), accum_dtype),
        grad_sub_edge_ij: T.Tensor((EK, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=THREADS) as (bx, by):
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc)
            for split in T.serial(SPLIT_M):
                for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                    d = bx * BLOCK_D + di
                    k = by * BLOCK_K + ki
                    if d < D and k < K:
                        acc[di, ki] += workspace[split, d, k]
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < D and k < K:
                    if d < A:
                        grad_sub_angle[d, k] = acc[di, ki]
                    elif d < A + N:
                        grad_sub_node[d - A, k] = acc[di, ki]
                    elif d < A + N + EK:
                        grad_sub_edge_ik[d - A - N, k] = acc[di, ki]
                    else:
                        grad_sub_edge_ij[d - A - N - EK, k] = acc[di, ki]
    return angle_double_backward_reduce_partials

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
        grad_output: T.Tensor((M, K), dtype),
        sub_node: T.Tensor((N, K), dtype),
        node_start: T.Tensor((A,), "int32"),
        node_count: T.Tensor((A,), "int32"),
        grad_flat_node_ebd: T.Tensor((A, N), accum_dtype),
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
        flat_node_ebd: T.Tensor((A, N), dtype),
        grad_output: T.Tensor((M, K), dtype),
        node_start: T.Tensor((A,), "int32"),
        node_count: T.Tensor((A,), "int32"),
        grad_sub_node: T.Tensor((N, K), accum_dtype),
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

@tilelang.jit
def fused_angle_update_double_backward_inputs(
    M,
    K,
    A,
    N,
    EK,
    N_NODE,
    N_EDGE,
    dtype="float32",
    accum_dtype="float32",
    THREADS=128,
    BLOCK_M=32,
    BLOCK_N=32,
    BLOCK_K=32,
    HAS_ANGLE=True, HAS_NODE=True, HAS_EDGE=True, HAS_ANGLE_WEIGHT=True, HAS_NODE_WEIGHT=True, HAS_IK_WEIGHT=True, HAS_IJ_WEIGHT=True, HAS_BIAS=True,
):
    """Fuse grad-grad-output and angle/node/edge input-side VJPs.

    HAS_* remove missing-cotangent pipelines at compile time. Direct outputs
    remain fully written; scatter outputs retain their zero-initialization.
    """

    @T.prim_func
    def angle_double_backward_inputs(
        grad_output: T.Tensor((M, K), dtype),
        flat_angle_ebd: T.Tensor((M, A), dtype),
        flat_node_ebd: T.Tensor((N_NODE, N), dtype),
        flat_edge_ebd: T.Tensor((N_EDGE, EK), dtype),
        n2a_index: T.Tensor((M,), "int64"),
        eij2a_index: T.Tensor((M,), "int64"),
        eik2a_index: T.Tensor((M,), "int64"),
        sub_angle: T.Tensor((A, K), dtype),
        sub_node: T.Tensor((N, K), dtype),
        sub_edge_ik: T.Tensor((EK, K), dtype),
        sub_edge_ij: T.Tensor((EK, K), dtype),
        gg_flat_angle: T.Tensor((M, A), dtype),
        gg_flat_node: T.Tensor((N_NODE, N), dtype),
        gg_flat_edge: T.Tensor((N_EDGE, EK), dtype),
        gg_sub_angle: T.Tensor((A, K), dtype),
        gg_sub_node: T.Tensor((N, K), dtype),
        gg_sub_edge_ik: T.Tensor((EK, K), dtype),
        gg_sub_edge_ij: T.Tensor((EK, K), dtype),
        gg_bias: T.Tensor((K,), dtype),
        grad_grad_output: T.Tensor((M, K), accum_dtype),
        grad_flat_angle: T.Tensor((M, A), accum_dtype),
        grad_flat_node: T.Tensor((N_NODE, N), accum_dtype),
        grad_flat_edge: T.Tensor((N_EDGE, EK), accum_dtype),
    ):
        # Original scalar implementation (retained for reference):
        # with T.Kernel(M, threads=THREADS) as (angle_idx,):
        # node_idx = n2a_index[angle_idx]
        # edge_ik_idx = eik2a_index[angle_idx]
        # edge_ij_idx = eij2a_index[angle_idx]
        #
        # for k in T.Parallel(K):
        # acc = T.alloc_var(accum_dtype)
        # acc = gg_bias[k]
        # for d in T.serial(A):
        # acc += (
        # gg_flat_angle[angle_idx, d] * sub_angle[d, k]
        # + flat_angle_ebd[angle_idx, d] * gg_sub_angle[d, k]
        # )
        # for d in T.serial(N):
        # acc += (
        # gg_flat_node[node_idx, d] * sub_node[d, k]
        # + flat_node_ebd[node_idx, d] * gg_sub_node[d, k]
        # )
        # for d in T.serial(EK):
        # acc += (
        # gg_flat_edge[edge_ik_idx, d] * sub_edge_ik[d, k]
        # + flat_edge_ebd[edge_ik_idx, d] * gg_sub_edge_ik[d, k]
        # + gg_flat_edge[edge_ij_idx, d] * sub_edge_ij[d, k]
        # + flat_edge_ebd[edge_ij_idx, d] * gg_sub_edge_ij[d, k]
        # )
        # grad_grad_output[angle_idx, k] = acc
        #
        # for d in T.Parallel(A):
        # acc = T.alloc_var(accum_dtype, init=0)
        # for k in T.serial(K):
        # acc += grad_output[angle_idx, k] * gg_sub_angle[d, k]
        # grad_flat_angle[angle_idx, d] = acc
        #
        # for d in T.Parallel(N):
        # acc = T.alloc_var(accum_dtype, init=0)
        # for k in T.serial(K):
        # acc += grad_output[angle_idx, k] * gg_sub_node[d, k]
        # T.atomic_add(grad_flat_node[node_idx, d], acc)
        #
        # for d in T.Parallel(EK):
        # acc_ik = T.alloc_var(accum_dtype, init=0)
        # acc_ij = T.alloc_var(accum_dtype, init=0)
        # for k in T.serial(K):
        # acc_ik += grad_output[angle_idx, k] * gg_sub_edge_ik[d, k]
        # acc_ij += grad_output[angle_idx, k] * gg_sub_edge_ij[d, k]
        # T.atomic_add(grad_flat_edge[edge_ik_idx, d], acc_ik)
        # T.atomic_add(grad_flat_edge[edge_ij_idx, d], acc_ij)

        # A common output-column grid covers grad-grad-output and input VJPs.
        with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(max(K, A, N, EK, EK), BLOCK_N), threads=THREADS) as (bx, by):
            lhs_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            rhs_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
            acc_output = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            acc_input = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            T.clear(acc_output)
            if by * BLOCK_N < K:
                # angle: U_x W and X U_w; each has its own pipeline.
                if HAS_ANGLE:
                    for ro in T.Pipelined(T.ceildiv(A, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < M and r < A:
                                lhs_shared[mi, ri] = gg_flat_angle[m, r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < A and n < K:
                                rhs_shared[ri, ni] = sub_angle[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_output)
                        T.sync_threads()
                T.sync_threads()
                if HAS_ANGLE_WEIGHT:
                    for ro in T.Pipelined(T.ceildiv(A, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < M and r < A:
                                lhs_shared[mi, ri] = flat_angle_ebd[m, r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < A and n < K:
                                rhs_shared[ri, ni] = gg_sub_angle[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_output)
                        T.sync_threads()
                T.sync_threads()
                # node: U_x W and X U_w; each has its own pipeline.
                if HAS_NODE:
                    for ro in T.Pipelined(T.ceildiv(N, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < M and r < N:
                                lhs_shared[mi, ri] = gg_flat_node[n2a_index[m], r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < N and n < K:
                                rhs_shared[ri, ni] = sub_node[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_output)
                        T.sync_threads()
                T.sync_threads()
                if HAS_NODE_WEIGHT:
                    for ro in T.Pipelined(T.ceildiv(N, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < M and r < N:
                                lhs_shared[mi, ri] = flat_node_ebd[n2a_index[m], r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < N and n < K:
                                rhs_shared[ri, ni] = gg_sub_node[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_output)
                        T.sync_threads()
                T.sync_threads()
                # ik: U_x W and X U_w; each has its own pipeline.
                if HAS_EDGE:
                    for ro in T.Pipelined(T.ceildiv(EK, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < M and r < EK:
                                lhs_shared[mi, ri] = gg_flat_edge[eik2a_index[m], r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < EK and n < K:
                                rhs_shared[ri, ni] = sub_edge_ik[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_output)
                        T.sync_threads()
                T.sync_threads()
                if HAS_IK_WEIGHT:
                    for ro in T.Pipelined(T.ceildiv(EK, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < M and r < EK:
                                lhs_shared[mi, ri] = flat_edge_ebd[eik2a_index[m], r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < EK and n < K:
                                rhs_shared[ri, ni] = gg_sub_edge_ik[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_output)
                        T.sync_threads()
                T.sync_threads()
                # ij: U_x W and X U_w; each has its own pipeline.
                if HAS_EDGE:
                    for ro in T.Pipelined(T.ceildiv(EK, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < M and r < EK:
                                lhs_shared[mi, ri] = gg_flat_edge[eij2a_index[m], r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < EK and n < K:
                                rhs_shared[ri, ni] = sub_edge_ij[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_output)
                        T.sync_threads()
                T.sync_threads()
                if HAS_IJ_WEIGHT:
                    for ro in T.Pipelined(T.ceildiv(EK, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < M and r < EK:
                                lhs_shared[mi, ri] = flat_edge_ebd[eij2a_index[m], r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < EK and n < K:
                                rhs_shared[ri, ni] = gg_sub_edge_ij[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_output)
                        T.sync_threads()
                T.sync_threads()
            for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                m = bx * BLOCK_M + mi
                n = by * BLOCK_N + ni
                if m < M and n < K:
                    if HAS_BIAS:
                        grad_grad_output[m, n] = acc_output[mi, ni] + gg_bias[n]
                    else:
                        grad_grad_output[m, n] = acc_output[mi, ni]

            # angle input VJP: G U_w.T; reuse the fragment after writeback.
            if by * BLOCK_N < A:
                T.clear(acc_input)
                if HAS_ANGLE_WEIGHT:
                    for ro in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < M and r < K:
                                lhs_shared[mi, ri] = grad_output[m, r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < K and n < A:
                                rhs_shared[ri, ni] = gg_sub_angle[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_input, transpose_B=True)
                        T.sync_threads()
                T.sync_threads()
                for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                    m = bx * BLOCK_M + mi
                    n = by * BLOCK_N + ni
                    if m < M and n < A:
                        grad_flat_angle[m, n] = acc_input[mi, ni]
                T.sync_threads()

            # node input VJP: missing cotangent leaves the zeroed output unchanged.
            if HAS_NODE_WEIGHT and by * BLOCK_N < N:
                T.clear(acc_input)
                if HAS_NODE_WEIGHT:
                    for ro in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < M and r < K:
                                lhs_shared[mi, ri] = grad_output[m, r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < K and n < N:
                                rhs_shared[ri, ni] = gg_sub_node[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_input, transpose_B=True)
                        T.sync_threads()
                T.sync_threads()
                for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                    m = bx * BLOCK_M + mi
                    n = by * BLOCK_N + ni
                    if m < M and n < N:
                        T.atomic_add(grad_flat_node[n2a_index[m], n], acc_input[mi, ni])
                T.sync_threads()

            # ik input VJP: missing cotangent leaves the zeroed output unchanged.
            if HAS_IK_WEIGHT and by * BLOCK_N < EK:
                T.clear(acc_input)
                if HAS_IK_WEIGHT:
                    for ro in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < M and r < K:
                                lhs_shared[mi, ri] = grad_output[m, r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < K and n < EK:
                                rhs_shared[ri, ni] = gg_sub_edge_ik[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_input, transpose_B=True)
                        T.sync_threads()
                T.sync_threads()
                for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                    m = bx * BLOCK_M + mi
                    n = by * BLOCK_N + ni
                    if m < M and n < EK:
                        T.atomic_add(grad_flat_edge[eik2a_index[m], n], acc_input[mi, ni])
                T.sync_threads()

            # ij input VJP: missing cotangent leaves the zeroed output unchanged.
            if HAS_IJ_WEIGHT and by * BLOCK_N < EK:
                T.clear(acc_input)
                if HAS_IJ_WEIGHT:
                    for ro in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                        for mi, ri in T.Parallel(BLOCK_M, BLOCK_K):
                            m = bx * BLOCK_M + mi
                            r = ro * BLOCK_K + ri
                            if m < M and r < K:
                                lhs_shared[mi, ri] = grad_output[m, r]
                            else:
                                lhs_shared[mi, ri] = 0
                        for ri, ni in T.Parallel(BLOCK_K, BLOCK_N):
                            r = ro * BLOCK_K + ri
                            n = by * BLOCK_N + ni
                            if r < K and n < EK:
                                rhs_shared[ri, ni] = gg_sub_edge_ij[r, n]
                            else:
                                rhs_shared[ri, ni] = 0
                        T.sync_threads()
                        T.gemm(lhs_shared, rhs_shared, acc_input, transpose_B=True)
                        T.sync_threads()
                T.sync_threads()
                for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                    m = bx * BLOCK_M + mi
                    n = by * BLOCK_N + ni
                    if m < M and n < EK:
                        T.atomic_add(grad_flat_edge[eij2a_index[m], n], acc_input[mi, ni])
                T.sync_threads()

    return angle_double_backward_inputs


@tilelang.jit
def fused_angle_update_double_backward_weights(
    M,
    K,
    A,
    N,
    EK,
    N_NODE,
    N_EDGE,
    dtype="float32",
    accum_dtype="float32",
    # Original scalar tile: BLOCK_D=16.
    BLOCK_D=32,
    BLOCK_K=16,
    THREADS=128,
    BLOCK_M=32,
):
    """Fuse all four weight-side VJPs into one reduction kernel."""
    MAX_D = max(A, N, EK)

    @T.prim_func
    def double_backward_weights(
        grad_output: T.Tensor((M, K), dtype),
        n2a_index: T.Tensor((M,), "int64"),
        eij2a_index: T.Tensor((M,), "int64"),
        eik2a_index: T.Tensor((M,), "int64"),
        gg_flat_angle: T.Tensor((M, A), dtype),
        gg_flat_node: T.Tensor((N_NODE, N), dtype),
        gg_flat_edge: T.Tensor((N_EDGE, EK), dtype),
        grad_sub_angle: T.Tensor((A, K), accum_dtype),
        grad_sub_node: T.Tensor((N, K), accum_dtype),
        grad_sub_edge_ik: T.Tensor((EK, K), accum_dtype),
        grad_sub_edge_ij: T.Tensor((EK, K), accum_dtype),
    ):
        # Original scalar implementation (retained for reference):
        # with T.Kernel(
        # T.ceildiv(MAX_D, BLOCK_D),
        # T.ceildiv(K, BLOCK_K),
        # threads=THREADS,
        # ) as (bx, by):
        # for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
        # d = bx * BLOCK_D + di
        # k = by * BLOCK_K + ki
        # if k < K:
        # if d < A:
        # acc_angle = T.alloc_var(accum_dtype, init=0)
        # for m in T.serial(M):
        # acc_angle += gg_flat_angle[m, d] * grad_output[m, k]
        # grad_sub_angle[d, k] = acc_angle
        # if d < N:
        # acc_node = T.alloc_var(accum_dtype, init=0)
        # for m in T.serial(M):
        # acc_node += gg_flat_node[n2a_index[m], d] * grad_output[m, k]
        # grad_sub_node[d, k] = acc_node
        # if d < EK:
        # acc_ik = T.alloc_var(accum_dtype, init=0)
        # acc_ij = T.alloc_var(accum_dtype, init=0)
        # for m in T.serial(M):
        # acc_ik += gg_flat_edge[eik2a_index[m], d] * grad_output[m, k]
        # acc_ij += gg_flat_edge[eij2a_index[m], d] * grad_output[m, k]
        # grad_sub_edge_ik[d, k] = acc_ik
        # grad_sub_edge_ij[d, k] = acc_ij

        with T.Kernel(T.ceildiv(MAX_D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=THREADS) as (bx, by):
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            # angle: gathered U_x.T @ G, with row-major shared loads.
            T.clear(acc)
            for mo in T.Pipelined(T.ceildiv(M, BLOCK_M), num_stages=2):
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    m = mo * BLOCK_M + mi
                    d = bx * BLOCK_D + di
                    if m < M and d < A:
                        feature_shared[mi, di] = gg_flat_angle[m, d]
                    else:
                        feature_shared[mi, di] = 0
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = mo * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                T.sync_threads()
            T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < A and k < K:
                    grad_sub_angle[d, k] = acc[di, ki]
            T.sync_threads()

            # node: gathered U_x.T @ G, with row-major shared loads.
            T.clear(acc)
            for mo in T.Pipelined(T.ceildiv(M, BLOCK_M), num_stages=2):
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    m = mo * BLOCK_M + mi
                    d = bx * BLOCK_D + di
                    if m < M and d < N:
                        feature_shared[mi, di] = gg_flat_node[n2a_index[m], d]
                    else:
                        feature_shared[mi, di] = 0
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = mo * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                T.sync_threads()
            T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < N and k < K:
                    grad_sub_node[d, k] = acc[di, ki]
            T.sync_threads()

            # ik: gathered U_x.T @ G, with row-major shared loads.
            T.clear(acc)
            for mo in T.Pipelined(T.ceildiv(M, BLOCK_M), num_stages=2):
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    m = mo * BLOCK_M + mi
                    d = bx * BLOCK_D + di
                    if m < M and d < EK:
                        feature_shared[mi, di] = gg_flat_edge[eik2a_index[m], d]
                    else:
                        feature_shared[mi, di] = 0
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = mo * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                T.sync_threads()
            T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < EK and k < K:
                    grad_sub_edge_ik[d, k] = acc[di, ki]
            T.sync_threads()

            # ij: gathered U_x.T @ G, with row-major shared loads.
            T.clear(acc)
            for mo in T.Pipelined(T.ceildiv(M, BLOCK_M), num_stages=2):
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    m = mo * BLOCK_M + mi
                    d = bx * BLOCK_D + di
                    if m < M and d < EK:
                        feature_shared[mi, di] = gg_flat_edge[eij2a_index[m], d]
                    else:
                        feature_shared[mi, di] = 0
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = mo * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                T.gemm(feature_shared, grad_shared, acc, transpose_A=True)
                T.sync_threads()
            T.sync_threads()
            for di, ki in T.Parallel(BLOCK_D, BLOCK_K):
                d = bx * BLOCK_D + di
                k = by * BLOCK_K + ki
                if d < EK and k < K:
                    grad_sub_edge_ij[d, k] = acc[di, ki]
            T.sync_threads()

    return double_backward_weights


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
        
        """
        grad_flat_angle_ebd = torch.matmul(grad_output, sub_angle.transpose(0, 1))
        
        grad_flat_node_ebd = torch.zeros(flat_node_ebd.shape, device=grad_output.device, dtype=grad_output.dtype)
        # grad_gathered_node = torch.matmul(grad_output, sub_node.transpose(0, 1))
        # grad_flat_node_ebd.index_add_(0, n2a_index, grad_gathered_node)
        grad_node_ebd = grad_flat_node_ebd.reshape(node_ebd_shape)
        
        grad_flat_edge_ebd = torch.zeros(flat_edge_ebd.shape, device=grad_output.device, dtype=grad_output.dtype)
        # grad_gathered_edge_ik = torch.matmul(grad_output, sub_edge_ik.transpose(0, 1))
        # grad_gathered_edge_ij = torch.matmul(grad_output, sub_edge_ij.transpose(0, 1))
        # grad_flat_edge_ebd.index_add_(0, eik2a_index, grad_gathered_edge_ik)
        # grad_flat_edge_ebd.index_add_(0, eij2a_index, grad_gathered_edge_ij)
    
        grad_sub_angle = torch.matmul(flat_angle_ebd.transpose(0, 1), grad_output)

        # gathered_node = torch.index_select(flat_node_ebd, 0, n2a_index)
        # grad_sub_node = torch.matmul(gathered_node.transpose(0, 1), grad_output)

        # gathered_edge_ik = torch.index_select(flat_edge_ebd, 0, eik2a_index)
        # grad_sub_edge_ik = torch.matmul(gathered_edge_ik.transpose(0, 1), grad_output)
        
        # gathered_edge_ij = torch.index_select(flat_edge_ebd, 0, eij2a_index)
        # grad_sub_edge_ij = torch.matmul(gathered_edge_ij.transpose(0, 1), grad_output)

        grad_bias = grad_output.sum(dim=0)
        """

        M, K = grad_output.shape
        _, A = flat_angle_ebd.shape
        N_NODE, N = flat_node_ebd.shape
        N_EDGE, EK = flat_edge_ebd.shape
        dtype = str(grad_output.dtype).replace("torch.", "")

        grad_flat_angle_ebd = torch.empty_like(flat_angle_ebd)
        grad_flat_node_ebd = torch.zeros_like(flat_node_ebd)
        grad_flat_edge_ebd = torch.zeros_like(flat_edge_ebd)
        grad_bias = torch.zeros((K,), device=grad_output.device, dtype=grad_output.dtype)
        input_kernel = fused_angle_update_backward_inputs_v2(
            M=M,
            K=K,
            A=A,
            N=N,
            EK=EK,
            N_NODE=N_NODE,
            N_EDGE=N_EDGE,
            dtype=dtype,
        )
        input_kernel(
            grad_output,
            n2a_index,
            eij2a_index,
            eik2a_index,
            sub_angle,
            sub_node,
            sub_edge_ik,
            sub_edge_ij,
            grad_flat_angle_ebd,
            grad_flat_node_ebd,
            grad_flat_edge_ebd,
            grad_bias,
        )
        grad_node_ebd = grad_flat_node_ebd.reshape(node_ebd_shape)

        grad_sub_angle = torch.empty_like(sub_angle)
        grad_sub_node = torch.empty_like(sub_node)
        grad_sub_edge_ik = torch.empty_like(sub_edge_ik)
        grad_sub_edge_ij = torch.empty_like(sub_edge_ij)
        # weight_kernel = fused_angle_update_backward_weights(
        # M=M,
        # K=K,
        # A=A,
        # N=N,
        # EK=EK,
        # N_NODE=N_NODE,
        # N_EDGE=N_EDGE,
        # dtype=dtype,
        # )
        # weight_kernel(
        # grad_output,
        # flat_angle_ebd,
        # flat_node_ebd,
        # flat_edge_ebd,
        # n2a_index,
        # eij2a_index,
        # eik2a_index,
        # grad_sub_angle,
        # grad_sub_node,
        # grad_sub_edge_ik,
        # grad_sub_edge_ij,
        # )
        # Bound the split count by available reduction tiles. Workspace is
        # temporary and is not saved for double backward.
        split_m = min(80, max(1, (M + 31) // 32))
        workspace = torch.empty(
            (split_m, A + N + 2 * EK, K),
            device=grad_output.device, dtype=grad_output.dtype,
        )
        weight_kernel = fused_angle_update_backward_weights_v3_3(
            M=M, K=K, A=A, N=N, EK=EK, N_NODE=N_NODE, N_EDGE=N_EDGE,
            dtype=dtype, accum_dtype=dtype, SPLIT_M=split_m,
        )
        # weight_kernel.export_sources(
        #     kernel_path="/workspace/DeepModelingCommunity/angle_backward_weight_partials_v3_1.cu"
        # )
        weight_kernel(
            grad_output, flat_angle_ebd, flat_node_ebd, flat_edge_ebd,
            n2a_index, eij2a_index, eik2a_index, workspace,
        )
        weight_reduce = fused_angle_update_backward_weights_v2_reduce(
            K=K, A=A, N=N, EK=EK, SPLIT_M=split_m, accum_dtype=dtype,
        )
        weight_reduce(
            workspace, grad_sub_angle, grad_sub_node,
            grad_sub_edge_ik, grad_sub_edge_ij,
        )

        # Preserve undefined second-order cotangents instead of autograd zeros.
        ctx.set_materialize_grads(False)
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

        """
        Original unfused second-order backward implementation.
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
        """

        M, K = grad_output.shape
        _, A = flat_angle_ebd.shape
        N_NODE, N = flat_node_ebd.shape
        N_EDGE, EK = flat_edge_ebd.shape
        dtype = str(grad_output.dtype).replace("torch.", "")

        # Original zero-materialization code retained for reference.
        # grad_grad_flat_angle_ebd = (
        #     torch.zeros_like(flat_angle_ebd)
        #     if grad_grad_flat_angle_ebd is None
        #     else grad_grad_flat_angle_ebd.contiguous()
        # )
        # grad_grad_flat_node_ebd = (
        #     torch.zeros_like(flat_node_ebd)
        #     if grad_grad_node_ebd is None
        #     else grad_grad_node_ebd.reshape_as(flat_node_ebd).contiguous()
        # )
        # grad_grad_flat_edge_ebd = (
        #     torch.zeros_like(flat_edge_ebd)
        #     if grad_grad_flat_edge_ebd is None
        #     else grad_grad_flat_edge_ebd.contiguous()
        # )
        # grad_grad_sub_angle = (
        #     torch.zeros_like(sub_angle)
        #     if grad_grad_sub_angle is None
        #     else grad_grad_sub_angle.contiguous()
        # )
        # grad_grad_sub_node = (
        #     torch.zeros_like(sub_node)
        #     if grad_grad_sub_node is None
        #     else grad_grad_sub_node.contiguous()
        # )
        # grad_grad_sub_edge_ik = (
        #     torch.zeros_like(sub_edge_ik)
        #     if grad_grad_sub_edge_ik is None
        #     else grad_grad_sub_edge_ik.contiguous()
        # )
        # grad_grad_sub_edge_ij = (
        #     torch.zeros_like(sub_edge_ij)
        #     if grad_grad_sub_edge_ij is None
        #     else grad_grad_sub_edge_ij.contiguous()
        # )
        # grad_grad_bias = (
        #     torch.zeros((K,), device=grad_output.device, dtype=grad_output.dtype)
        #     if grad_grad_bias is None
        #     else grad_grad_bias.contiguous()
        # )
        HAS_ANGLE = grad_grad_flat_angle_ebd is not None
        grad_grad_flat_angle_ebd = None if grad_grad_flat_angle_ebd is None else grad_grad_flat_angle_ebd.contiguous()
        HAS_NODE = grad_grad_node_ebd is not None
        grad_grad_flat_node_ebd = None if grad_grad_node_ebd is None else grad_grad_node_ebd.reshape_as(flat_node_ebd).contiguous()
        HAS_EDGE = grad_grad_flat_edge_ebd is not None
        grad_grad_flat_edge_ebd = None if grad_grad_flat_edge_ebd is None else grad_grad_flat_edge_ebd.contiguous()
        HAS_ANGLE_WEIGHT = grad_grad_sub_angle is not None
        grad_grad_sub_angle = None if grad_grad_sub_angle is None else grad_grad_sub_angle.contiguous()
        HAS_NODE_WEIGHT = grad_grad_sub_node is not None
        grad_grad_sub_node = None if grad_grad_sub_node is None else grad_grad_sub_node.contiguous()
        HAS_IK_WEIGHT = grad_grad_sub_edge_ik is not None
        grad_grad_sub_edge_ik = None if grad_grad_sub_edge_ik is None else grad_grad_sub_edge_ik.contiguous()
        HAS_IJ_WEIGHT = grad_grad_sub_edge_ij is not None
        grad_grad_sub_edge_ij = None if grad_grad_sub_edge_ij is None else grad_grad_sub_edge_ij.contiguous()
        HAS_BIAS = grad_grad_bias is not None
        grad_grad_bias = None if grad_grad_bias is None else grad_grad_bias.contiguous()

        # Saved tensors serve as unread, shape-compatible launch placeholders.
        # The missing cotangents themselves remain None in Python.
        grad_grad_output = torch.empty_like(grad_output)
        grad_flat_angle_ebd = torch.empty_like(flat_angle_ebd)
        grad_flat_node_ebd = torch.zeros_like(flat_node_ebd)
        grad_flat_edge_ebd = torch.zeros_like(flat_edge_ebd)
        input_kernel = fused_angle_update_double_backward_inputs(
            M=M,
            K=K,
            A=A,
            N=N,
            EK=EK,
            N_NODE=N_NODE,
            N_EDGE=N_EDGE,
            dtype=dtype,
            accum_dtype=dtype,
            HAS_ANGLE=HAS_ANGLE,
            HAS_NODE=HAS_NODE,
            HAS_EDGE=HAS_EDGE,
            HAS_ANGLE_WEIGHT=HAS_ANGLE_WEIGHT,
            HAS_NODE_WEIGHT=HAS_NODE_WEIGHT,
            HAS_IK_WEIGHT=HAS_IK_WEIGHT,
            HAS_IJ_WEIGHT=HAS_IJ_WEIGHT,
            HAS_BIAS=HAS_BIAS,
        )
        input_kernel(
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
            grad_grad_flat_angle_ebd if HAS_ANGLE else flat_angle_ebd,
            grad_grad_flat_node_ebd if HAS_NODE else flat_node_ebd,
            grad_grad_flat_edge_ebd if HAS_EDGE else flat_edge_ebd,
            grad_grad_sub_angle if HAS_ANGLE_WEIGHT else sub_angle,
            grad_grad_sub_node if HAS_NODE_WEIGHT else sub_node,
            grad_grad_sub_edge_ik if HAS_IK_WEIGHT else sub_edge_ik,
            grad_grad_sub_edge_ij if HAS_IJ_WEIGHT else sub_edge_ij,
            grad_grad_bias if HAS_BIAS else sub_angle[0],
            grad_grad_output,
            grad_flat_angle_ebd,
            grad_flat_node_ebd,
            grad_flat_edge_ebd,
        )

        grad_sub_angle = torch.empty_like(sub_angle)
        grad_sub_node = torch.empty_like(sub_node)
        grad_sub_edge_ik = torch.empty_like(sub_edge_ik)
        grad_sub_edge_ij = torch.empty_like(sub_edge_ij)
        # weight_kernel = fused_angle_update_double_backward_weights(
        #     M=M,
        #     K=K,
        #     A=A,
        #     N=N,
        #     EK=EK,
        #     N_NODE=N_NODE,
        #     N_EDGE=N_EDGE,
        #     dtype=dtype,
        # )
        # weight_kernel(
        #     grad_output,
        #     n2a_index,
        #     eij2a_index,
        #     eik2a_index,
        #     grad_grad_flat_angle_ebd,
        #     grad_grad_flat_node_ebd,
        #     grad_grad_flat_edge_ebd,
        #     grad_sub_angle,
        #     grad_sub_node,
        #     grad_sub_edge_ik,
        #     grad_sub_edge_ij,
        # )
        split_m = min(132, max(1, (M + 31) // 32))
        workspace = torch.empty(
            (split_m, A + N + 2 * EK, K),
            dtype=grad_output.dtype, device=grad_output.device,
        )
        weight_kernel = fused_angle_update_double_backward_weights_v3(
            M=M, K=K, A=A, N=N, EK=EK, N_NODE=N_NODE, N_EDGE=N_EDGE,
            dtype=dtype, accum_dtype=dtype, SPLIT_M=split_m,
            HAS_ANGLE=HAS_ANGLE, HAS_NODE=HAS_NODE, HAS_EDGE=HAS_EDGE,
        )
        weight_kernel(
            grad_output,
            grad_grad_flat_angle_ebd if HAS_ANGLE else flat_angle_ebd,
            grad_grad_flat_node_ebd if HAS_NODE else flat_node_ebd,
            grad_grad_flat_edge_ebd if HAS_EDGE else flat_edge_ebd,
            n2a_index, eij2a_index, eik2a_index, workspace,
        )
        reduce_kernel_v2 = fused_angle_update_double_backward_weights_v2_reduce(
            K=K, A=A, N=N, EK=EK, SPLIT_M=split_m, accum_dtype=dtype,
        )
        reduce_kernel_v2(
            workspace, grad_sub_angle, grad_sub_node,
            grad_sub_edge_ik, grad_sub_edge_ij,
        )

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


# ============================================================================
# Wider RepFlow symmetrization block
#
# Keep this implementation append-only.  The original fused symmetrization,
# edge-update, and angle-update kernels above are retained as independent
# reference/fallback implementations.
# ============================================================================


@tilelang.jit
def fused_sym_block_dual_hg_forward_uniform(
    M, E_EDGE, E_NODE, N_NODE_EXT, NO, BLOCK_N=64,
):
    """Compute both HG tensors while gathering the node branch on demand."""
    assert M % NO == 0
    EDGES_PER_OWNER = M // NO
    D = 3 * (E_EDGE + E_NODE)

    @T.prim_func
    def hg_forward_uniform(
        edge_ebd: T.Tensor((M, E_EDGE), "float32"),
        node_ebd_ext: T.Tensor((N_NODE_EXT, E_NODE), "float32"),
        h2: T.Tensor((M, 3), "float32"),
        sw: T.Tensor((M,), "float32"),
        n_ext2e_index: T.Tensor((M,), "int64"),
        scale: T.float32,
        h_edge: T.Tensor((NO, 3 * E_EDGE), "float32"),
        h_node: T.Tensor((NO, 3 * E_NODE), "float32"),
    ):
        with T.Kernel(NO, T.ceildiv(D, BLOCK_N), threads=64) as (owner, tile):
            meta = T.alloc_shared((EDGES_PER_OWNER, 4), "float32")
            for r in T.Parallel(EDGES_PER_OWNER):
                edge = owner * EDGES_PER_OWNER + r
                meta[r, 0] = sw[edge]
            for r, b in T.Parallel(EDGES_PER_OWNER, 3):
                edge = owner * EDGES_PER_OWNER + r
                meta[r, b + 1] = h2[edge, b]
            T.sync_threads()

            for j in T.Parallel(BLOCK_N):
                col = tile * BLOCK_N + j
                if col < D:
                    acc = T.alloc_var("float32", init=0)
                    if col < 3 * E_EDGE:
                        edge_b = col // E_EDGE
                        edge_d = col % E_EDGE
                        for r in T.serial(EDGES_PER_OWNER):
                            edge = owner * EDGES_PER_OWNER + r
                            acc += edge_ebd[edge, edge_d] * meta[r, 0] * meta[r, edge_b + 1]
                        h_edge[owner, col] = acc * scale
                    else:
                        local = col - 3 * E_EDGE
                        node_b = local // E_NODE
                        node_d = local % E_NODE
                        for r in T.serial(EDGES_PER_OWNER):
                            edge = owner * EDGES_PER_OWNER + r
                            node = n_ext2e_index[edge]
                            acc += node_ebd_ext[node, node_d] * meta[r, 0] * meta[r, node_b + 1]
                        h_node[owner, local] = acc * scale

    return hg_forward_uniform


@tilelang.jit
def fused_sym_block_dual_hg_forward_segmented(
    M, E_EDGE, E_NODE, N_NODE_EXT, NO, BLOCK_N=64,
):
    """Segmented-owner variant of the dual HG kernel."""
    D = 3 * (E_EDGE + E_NODE)

    @T.prim_func
    def hg_forward_segmented(
        edge_ebd: T.Tensor((M, E_EDGE), "float32"),
        node_ebd_ext: T.Tensor((N_NODE_EXT, E_NODE), "float32"),
        h2: T.Tensor((M, 3), "float32"),
        sw: T.Tensor((M,), "float32"),
        n_ext2e_index: T.Tensor((M,), "int64"),
        offsets: T.Tensor((NO + 1,), "int64"),
        order: T.Tensor((M,), "int64"),
        scale: T.float32,
        h_edge: T.Tensor((NO, 3 * E_EDGE), "float32"),
        h_node: T.Tensor((NO, 3 * E_NODE), "float32"),
    ):
        with T.Kernel(NO, T.ceildiv(D, BLOCK_N), threads=64) as (owner, tile):
            for j in T.Parallel(BLOCK_N):
                col = tile * BLOCK_N + j
                if col < D:
                    acc = T.alloc_var("float32", init=0)
                    if col < 3 * E_EDGE:
                        edge_b = col // E_EDGE
                        edge_d = col % E_EDGE
                        for r in T.serial(offsets[owner], offsets[owner + 1]):
                            edge = order[r]
                            acc += edge_ebd[edge, edge_d] * sw[edge] * h2[edge, edge_b]
                        h_edge[owner, col] = acc * scale
                    else:
                        local = col - 3 * E_EDGE
                        node_b = local // E_NODE
                        node_d = local % E_NODE
                        for r in T.serial(offsets[owner], offsets[owner + 1]):
                            edge = order[r]
                            node = n_ext2e_index[edge]
                            acc += node_ebd_ext[node, node_d] * sw[edge] * h2[edge, node_b]
                        h_node[owner, local] = acc * scale

    return hg_forward_segmented


@tilelang.jit
def fused_sym_block_dual_grrg_forward(NO, E_EDGE, E_NODE, A, BLOCK_N=128):
    """Grouped Gram products; the two branches keep branch-local axes."""
    DQ_EDGE = A * E_EDGE
    DQ = A * (E_EDGE + E_NODE)

    @T.prim_func
    def grrg_forward(
        h_edge: T.Tensor((NO, 3 * E_EDGE), "float32"),
        h_node: T.Tensor((NO, 3 * E_NODE), "float32"),
        q_edge: T.Tensor((NO, DQ_EDGE), "float32"),
        q_node: T.Tensor((NO, A * E_NODE), "float32"),
    ):
        with T.Kernel(NO, T.ceildiv(DQ, BLOCK_N), threads=128) as (owner, tile):
            for j in T.Parallel(BLOCK_N):
                col = tile * BLOCK_N + j
                if col < DQ:
                    acc = T.alloc_var("float32", init=0)
                    if col < DQ_EDGE:
                        edge_a = col // E_EDGE
                        edge_d = col % E_EDGE
                        for b in T.serial(3):
                            acc += h_edge[owner, b * E_EDGE + edge_a] * h_edge[owner, b * E_EDGE + edge_d]
                        q_edge[owner, col] = acc / 3.0
                    else:
                        local = col - DQ_EDGE
                        node_a = local // E_NODE
                        node_d = local % E_NODE
                        for b in T.serial(3):
                            acc += h_node[owner, b * E_NODE + node_a] * h_node[owner, b * E_NODE + node_d]
                        q_node[owner, local] = acc / 3.0

    return grrg_forward


@tilelang.jit
def fused_sym_block_projection_act_residual_forward(
    N, D_EDGE, D_NODE, C, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
):
    """Logical-cat projection followed by custom-SiLU and residual update."""
    D = D_EDGE + D_NODE

    @T.prim_func
    def projection_forward(
        node_base: T.Tensor((N, C), "float32"),
        q_edge: T.Tensor((N, D_EDGE), "float32"),
        q_node: T.Tensor((N, D_NODE), "float32"),
        weight: T.Tensor((D, C), "float32"),
        bias: T.Tensor((C,), "float32"),
        residual: T.Tensor((C,), "float32"),
        threshold: T.float32,
        slope: T.float32,
        const_value: T.float32,
        preact: T.Tensor((N, C), "float32"),
        out: T.Tensor((N, C), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_M), T.ceildiv(C, BLOCK_N), threads=128) as (bx, by):
            a_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float32")
            b_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float32")
            acc = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(D, BLOCK_K), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    n = bx * BLOCK_M + mi
                    d = ko * BLOCK_K + ki
                    if n < N and d < D:
                        if d < D_EDGE:
                            a_shared[mi, ki] = q_edge[n, d]
                        else:
                            a_shared[mi, ki] = q_node[n, d - D_EDGE]
                    else:
                        a_shared[mi, ki] = 0
                for ki, ni in T.Parallel(BLOCK_K, BLOCK_N):
                    d = ko * BLOCK_K + ki
                    c = by * BLOCK_N + ni
                    if d < D and c < C:
                        b_shared[ki, ni] = weight[d, c]
                    else:
                        b_shared[ki, ni] = 0
                T.sync_threads()
                T.gemm(a_shared, b_shared, acc)
                T.sync_threads()

            for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                n = bx * BLOCK_M + mi
                c = by * BLOCK_N + ni
                if n < N and c < C:
                    z = acc[mi, ni] + bias[c]
                    value = T.alloc_var("float32")
                    if z >= threshold:
                        value = T.tanh(slope * (z - threshold)) + const_value
                    else:
                        sig = 1.0 / (1.0 + T.exp(-z))
                        value = z * sig
                    preact[n, c] = z
                    out[n, c] = node_base[n, c] + residual[c] * value

    return projection_forward


@tilelang.jit
def fused_sym_block_projection_backward_q(
    N, D_EDGE, D_NODE, C, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
):
    """Compute p @ W.T as one logical output-concatenated GEMM."""
    D = D_EDGE + D_NODE

    @T.prim_func
    def backward_q(
        grad_output: T.Tensor((N, C), "float32"),
        preact: T.Tensor((N, C), "float32"),
        weight: T.Tensor((D, C), "float32"),
        residual: T.Tensor((C,), "float32"),
        threshold: T.float32,
        slope: T.float32,
        grad_q_edge: T.Tensor((N, D_EDGE), "float32"),
        grad_q_node: T.Tensor((N, D_NODE), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_M), T.ceildiv(D, BLOCK_N), threads=128) as (bx, by):
            p_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float32")
            w_shared = T.alloc_shared((BLOCK_N, BLOCK_K), "float32")
            acc = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(C, BLOCK_K), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    n = bx * BLOCK_M + mi
                    c = ko * BLOCK_K + ki
                    if n < N and c < C:
                        z = preact[n, c]
                        deriv = T.alloc_var("float32")
                        if z >= threshold:
                            th = T.tanh(slope * (z - threshold))
                            deriv = slope * (1.0 - th * th)
                        else:
                            sig = 1.0 / (1.0 + T.exp(-z))
                            deriv = sig * (1.0 + z * (1.0 - sig))
                        p_shared[mi, ki] = grad_output[n, c] * residual[c] * deriv
                    else:
                        p_shared[mi, ki] = 0
                for ni, ki in T.Parallel(BLOCK_N, BLOCK_K):
                    d = by * BLOCK_N + ni
                    c = ko * BLOCK_K + ki
                    if d < D and c < C:
                        w_shared[ni, ki] = weight[d, c]
                    else:
                        w_shared[ni, ki] = 0
                T.sync_threads()
                T.gemm(p_shared, w_shared, acc, transpose_B=True)
                T.sync_threads()
            for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                n = bx * BLOCK_M + mi
                d = by * BLOCK_N + ni
                if n < N and d < D:
                    if d < D_EDGE:
                        grad_q_edge[n, d] = acc[mi, ni]
                    else:
                        grad_q_node[n, d - D_EDGE] = acc[mi, ni]

    return backward_q


@tilelang.jit
def fused_sym_block_projection_backward_weight(
    N, D_EDGE, D_NODE, C, BLOCK_D=32, BLOCK_C=32, BLOCK_N=32,
):
    """Compute logical [Q_edge|Q_node].T @ p without a cat buffer."""
    D = D_EDGE + D_NODE

    @T.prim_func
    def backward_weight(
        grad_output: T.Tensor((N, C), "float32"),
        preact: T.Tensor((N, C), "float32"),
        q_edge: T.Tensor((N, D_EDGE), "float32"),
        q_node: T.Tensor((N, D_NODE), "float32"),
        residual: T.Tensor((C,), "float32"),
        threshold: T.float32,
        slope: T.float32,
        grad_weight: T.Tensor((D, C), "float32"),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(C, BLOCK_C), threads=128) as (bx, by):
            q_shared = T.alloc_shared((BLOCK_N, BLOCK_D), "float32")
            p_shared = T.alloc_shared((BLOCK_N, BLOCK_C), "float32")
            acc = T.alloc_fragment((BLOCK_D, BLOCK_C), "float32")
            T.clear(acc)
            for no in T.Pipelined(T.ceildiv(N, BLOCK_N), num_stages=2):
                for ni, di in T.Parallel(BLOCK_N, BLOCK_D):
                    n = no * BLOCK_N + ni
                    d = bx * BLOCK_D + di
                    if n < N and d < D:
                        if d < D_EDGE:
                            q_shared[ni, di] = q_edge[n, d]
                        else:
                            q_shared[ni, di] = q_node[n, d - D_EDGE]
                    else:
                        q_shared[ni, di] = 0
                for ni, ci in T.Parallel(BLOCK_N, BLOCK_C):
                    n = no * BLOCK_N + ni
                    c = by * BLOCK_C + ci
                    if n < N and c < C:
                        z = preact[n, c]
                        deriv = T.alloc_var("float32")
                        if z >= threshold:
                            th = T.tanh(slope * (z - threshold))
                            deriv = slope * (1.0 - th * th)
                        else:
                            sig = 1.0 / (1.0 + T.exp(-z))
                            deriv = sig * (1.0 + z * (1.0 - sig))
                        p_shared[ni, ci] = grad_output[n, c] * residual[c] * deriv
                    else:
                        p_shared[ni, ci] = 0
                T.sync_threads()
                T.gemm(q_shared, p_shared, acc, transpose_A=True)
                T.sync_threads()
            for di, ci in T.Parallel(BLOCK_D, BLOCK_C):
                d = bx * BLOCK_D + di
                c = by * BLOCK_C + ci
                if d < D and c < C:
                    grad_weight[d, c] = acc[di, ci]

    return backward_weight


@tilelang.jit
def fused_sym_block_projection_backward_vector(N, C, THREADS=128):
    """Reduce projection bias and residual gradients per output channel."""

    @T.prim_func
    def backward_vector(
        grad_output: T.Tensor((N, C), "float32"),
        preact: T.Tensor((N, C), "float32"),
        residual: T.Tensor((C,), "float32"),
        threshold: T.float32,
        slope: T.float32,
        const_value: T.float32,
        grad_bias: T.Tensor((C,), "float32"),
        grad_residual: T.Tensor((C,), "float32"),
    ):
        with T.Kernel(C, threads=THREADS) as (c,):
            tx = T.get_thread_binding()
            partial_b = T.alloc_shared((THREADS,), "float32")
            partial_r = T.alloc_shared((THREADS,), "float32")
            acc_b = T.alloc_var("float32", init=0)
            acc_r = T.alloc_var("float32", init=0)
            for n in T.serial(tx, N, THREADS):
                z = preact[n, c]
                value = T.alloc_var("float32")
                deriv = T.alloc_var("float32")
                if z >= threshold:
                    th = T.tanh(slope * (z - threshold))
                    value = th + const_value
                    deriv = slope * (1.0 - th * th)
                else:
                    sig = 1.0 / (1.0 + T.exp(-z))
                    value = z * sig
                    deriv = sig * (1.0 + z * (1.0 - sig))
                g = grad_output[n, c]
                acc_b += g * residual[c] * deriv
                acc_r += g * value
            partial_b[tx] = acc_b
            partial_r[tx] = acc_r
            T.sync_threads()
            reduced_b = T.alloc_shared((1,), "float32")
            reduced_r = T.alloc_shared((1,), "float32")
            T.reduce_sum(partial_b, reduced_b, dim=0)
            T.reduce_sum(partial_r, reduced_r, dim=0)
            if tx == 0:
                grad_bias[c] = reduced_b[0]
                grad_residual[c] = reduced_r[0]

    return backward_vector


@tilelang.jit
def fused_sym_block_dual_grrg_backward(NO, E_EDGE, E_NODE, A, BLOCK_D=128):
    """Grouped Gram VJP; branch-local axis semantics are preserved."""
    MAX_E = max(E_EDGE, E_NODE)

    @T.prim_func
    def grrg_backward(
        grad_q_edge: T.Tensor((NO, A * E_EDGE), "float32"),
        grad_q_node: T.Tensor((NO, A * E_NODE), "float32"),
        h_edge: T.Tensor((NO, 3 * E_EDGE), "float32"),
        h_node: T.Tensor((NO, 3 * E_NODE), "float32"),
        scale: T.float32,
        grad_h_edge: T.Tensor((NO, 3 * E_EDGE), "float32"),
        grad_h_node: T.Tensor((NO, 3 * E_NODE), "float32"),
    ):
        with T.Kernel(NO, T.ceildiv(MAX_E, BLOCK_D), threads=128) as (owner, tile):
            for b, j in T.Parallel(3, BLOCK_D):
                d = tile * BLOCK_D + j
                if d < E_EDGE:
                    acc = T.alloc_var("float32", init=0)
                    for a in T.serial(A):
                        acc += h_edge[owner, b * E_EDGE + a] * grad_q_edge[owner, a * E_EDGE + d]
                    if d < A:
                        for k in T.serial(E_EDGE):
                            acc += h_edge[owner, b * E_EDGE + k] * grad_q_edge[owner, d * E_EDGE + k]
                    grad_h_edge[owner, b * E_EDGE + d] = acc * (scale / 3.0)
                if d < E_NODE:
                    acc = T.alloc_var("float32", init=0)
                    for a in T.serial(A):
                        acc += h_node[owner, b * E_NODE + a] * grad_q_node[owner, a * E_NODE + d]
                    if d < A:
                        for k in T.serial(E_NODE):
                            acc += h_node[owner, b * E_NODE + k] * grad_q_node[owner, d * E_NODE + k]
                    grad_h_node[owner, b * E_NODE + d] = acc * (scale / 3.0)

    return grrg_backward


@tilelang.jit
def fused_sym_block_dual_hg_backward(
    M, E_EDGE, E_NODE, N_NODE_EXT, NO, THREADS=128,
):
    """Dual HG VJP with direct edge writes and node-space scatter-adds."""

    @T.prim_func
    def hg_backward(
        grad_h_edge: T.Tensor((NO, 3 * E_EDGE), "float32"),
        grad_h_node: T.Tensor((NO, 3 * E_NODE), "float32"),
        edge_ebd: T.Tensor((M, E_EDGE), "float32"),
        node_ebd_ext: T.Tensor((N_NODE_EXT, E_NODE), "float32"),
        h2: T.Tensor((M, 3), "float32"),
        sw: T.Tensor((M,), "float32"),
        owner: T.Tensor((M,), "int64"),
        n_ext2e_index: T.Tensor((M,), "int64"),
        grad_flat_h_edge: T.Tensor((M, 3 * E_EDGE), "float32"),
        grad_flat_h_node: T.Tensor((M, 3 * E_NODE), "float32"),
        grad_edge_ebd: T.Tensor((M, E_EDGE), "float32"),
        grad_node_ebd_ext: T.Tensor((N_NODE_EXT, E_NODE), "float32"),
        grad_h2: T.Tensor((M, 3), "float32"),
        grad_sw: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M, threads=THREADS) as (edge,):
            tx = T.get_thread_binding()
            o = owner[edge]
            node = n_ext2e_index[edge]
            w = sw[edge]
            acc_h = T.alloc_fragment((3,), "float32")
            acc_sw = T.alloc_var("float32", init=0)
            T.clear(acc_h)

            for d in T.serial(tx, E_EDGE, THREADS):
                q = T.alloc_var("float32", init=0)
                for b in T.serial(3):
                    r = grad_h_edge[o, b * E_EDGE + d]
                    grad_flat_h_edge[edge, b * E_EDGE + d] = r
                    q += r * h2[edge, b]
                    acc_h[b] += r * edge_ebd[edge, d]
                grad_edge_ebd[edge, d] = q * w
                acc_sw += q * edge_ebd[edge, d]

            for d in T.serial(tx, E_NODE, THREADS):
                q = T.alloc_var("float32", init=0)
                value = node_ebd_ext[node, d]
                for b in T.serial(3):
                    r = grad_h_node[o, b * E_NODE + d]
                    grad_flat_h_node[edge, b * E_NODE + d] = r
                    q += r * h2[edge, b]
                    acc_h[b] += r * value
                T.atomic_add(grad_node_ebd_ext[node, d], q * w)
                acc_sw += q * value

            sh_h = T.alloc_shared((3, THREADS), "float32")
            sh_sw = T.alloc_shared((THREADS,), "float32")
            for b in T.Parallel(3):
                sh_h[b, tx] = acc_h[b]
            sh_sw[tx] = acc_sw
            T.sync_threads()
            red_h = T.alloc_shared((3,), "float32")
            red_sw = T.alloc_shared((1,), "float32")
            T.reduce_sum(sh_h, red_h, dim=1)
            T.reduce_sum(sh_sw, red_sw, dim=0)
            if tx == 0:
                for b in T.serial(3):
                    grad_h2[edge, b] = red_h[b] * w
                grad_sw[edge] = red_sw[0]

    return hg_backward


@tilelang.jit
def fused_sym_block_projection_double_ts_output(
    N, D_EDGE, D_NODE, C,
    HAS_BASE=True, HAS_Q_EDGE=True, HAS_Q_NODE=True,
    HAS_WEIGHT=True, HAS_BIAS=True, HAS_RESIDUAL=True,
    BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
):
    """Build t/s and grad-grad-output using a logical K=2D projection."""
    D = D_EDGE + D_NODE

    @T.prim_func
    def double_backward_ts(
        grad_output: T.Tensor((N, C), "float32"),
        q_edge: T.Tensor((N, D_EDGE), "float32"),
        q_node: T.Tensor((N, D_NODE), "float32"),
        weight: T.Tensor((D, C), "float32"),
        residual: T.Tensor((C,), "float32"),
        preact: T.Tensor((N, C), "float32"),
        u_base: T.Tensor((N, C), "float32"),
        u_q_edge: T.Tensor((N, D_EDGE), "float32"),
        u_q_node: T.Tensor((N, D_NODE), "float32"),
        u_weight: T.Tensor((D, C), "float32"),
        u_bias: T.Tensor((C,), "float32"),
        u_residual: T.Tensor((C,), "float32"),
        threshold: T.float32,
        slope: T.float32,
        const_value: T.float32,
        p_out: T.Tensor((N, C), "float32"),
        t_out: T.Tensor((N, C), "float32"),
        s_out: T.Tensor((N, C), "float32"),
        grad_grad_output: T.Tensor((N, C), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_M), T.ceildiv(C, BLOCK_N), threads=128) as (bx, by):
            a_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float32")
            b_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float32")
            acc = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.clear(acc)

            # [u_Q | Q] @ [W; u_W], with both concatenations kept logical.
            if HAS_Q_EDGE or HAS_Q_NODE or HAS_WEIGHT:
                for ko in T.Pipelined(T.ceildiv(2 * D, BLOCK_K), num_stages=2):
                    for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                        n = bx * BLOCK_M + mi
                        k = ko * BLOCK_K + ki
                        if n < N and k < 2 * D:
                            if k < D:
                                if k < D_EDGE:
                                    if HAS_Q_EDGE:
                                        a_shared[mi, ki] = u_q_edge[n, k]
                                    else:
                                        a_shared[mi, ki] = 0
                                else:
                                    if HAS_Q_NODE:
                                        a_shared[mi, ki] = u_q_node[n, k - D_EDGE]
                                    else:
                                        a_shared[mi, ki] = 0
                            else:
                                d = k - D
                                if d < D_EDGE:
                                    if HAS_WEIGHT:
                                        a_shared[mi, ki] = q_edge[n, d]
                                    else:
                                        a_shared[mi, ki] = 0
                                else:
                                    if HAS_WEIGHT:
                                        a_shared[mi, ki] = q_node[n, d - D_EDGE]
                                    else:
                                        a_shared[mi, ki] = 0
                        else:
                            a_shared[mi, ki] = 0
                    for ki, ni in T.Parallel(BLOCK_K, BLOCK_N):
                        k = ko * BLOCK_K + ki
                        c = by * BLOCK_N + ni
                        if k < 2 * D and c < C:
                            if k < D:
                                b_shared[ki, ni] = weight[k, c]
                            else:
                                if HAS_WEIGHT:
                                    b_shared[ki, ni] = u_weight[k - D, c]
                                else:
                                    b_shared[ki, ni] = 0
                        else:
                            b_shared[ki, ni] = 0
                    T.sync_threads()
                    T.gemm(a_shared, b_shared, acc)
                    T.sync_threads()

            for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                n = bx * BLOCK_M + mi
                c = by * BLOCK_N + ni
                if n < N and c < C:
                    z = preact[n, c]
                    t = T.alloc_var("float32")
                    p = T.alloc_var("float32")
                    s = T.alloc_var("float32")
                    gg = T.alloc_var("float32")
                    value = T.alloc_var("float32")
                    deriv = T.alloc_var("float32")
                    second = T.alloc_var("float32")
                    if z >= threshold:
                        th = T.tanh(slope * (z - threshold))
                        value = th + const_value
                        deriv = slope * (1.0 - th * th)
                        second = -2.0 * slope * th * deriv
                    else:
                        sig = 1.0 / (1.0 + T.exp(-z))
                        sigp = sig * (1.0 - sig)
                        value = z * sig
                        deriv = sig + z * sigp
                        second = sigp * (2.0 + z * (1.0 - 2.0 * sig))
                    t = acc[mi, ni]
                    if HAS_BIAS:
                        t += u_bias[c]
                    g = grad_output[n, c]
                    p = g * residual[c] * deriv
                    s = t * g * residual[c] * second
                    if HAS_RESIDUAL:
                        s += u_residual[c] * g * deriv
                    gg = t * residual[c] * deriv
                    if HAS_RESIDUAL:
                        gg += u_residual[c] * value
                    if HAS_BASE:
                        gg += u_base[n, c]
                    p_out[n, c] = p
                    t_out[n, c] = t
                    s_out[n, c] = s
                    grad_grad_output[n, c] = gg

    return double_backward_ts


@tilelang.jit
def fused_sym_block_projection_double_q(
    N, D_EDGE, D_NODE, C, HAS_WEIGHT=True,
    BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
):
    """Compute [p|s] @ [u_W.T;W.T] as one logical K=2C GEMM."""
    D = D_EDGE + D_NODE

    @T.prim_func
    def double_backward_q(
        p: T.Tensor((N, C), "float32"),
        s: T.Tensor((N, C), "float32"),
        weight: T.Tensor((D, C), "float32"),
        u_weight: T.Tensor((D, C), "float32"),
        grad_q_edge: T.Tensor((N, D_EDGE), "float32"),
        grad_q_node: T.Tensor((N, D_NODE), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_M), T.ceildiv(D, BLOCK_N), threads=128) as (bx, by):
            lhs = T.alloc_shared((BLOCK_M, BLOCK_K), "float32")
            rhs = T.alloc_shared((BLOCK_N, BLOCK_K), "float32")
            acc = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(2 * C, BLOCK_K), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    n = bx * BLOCK_M + mi
                    k = ko * BLOCK_K + ki
                    if n < N and k < 2 * C:
                        if k < C:
                            if HAS_WEIGHT:
                                lhs[mi, ki] = p[n, k]
                            else:
                                lhs[mi, ki] = 0
                        else:
                            lhs[mi, ki] = s[n, k - C]
                    else:
                        lhs[mi, ki] = 0
                for ni, ki in T.Parallel(BLOCK_N, BLOCK_K):
                    d = by * BLOCK_N + ni
                    k = ko * BLOCK_K + ki
                    if d < D and k < 2 * C:
                        if k < C:
                            if HAS_WEIGHT:
                                rhs[ni, ki] = u_weight[d, k]
                            else:
                                rhs[ni, ki] = 0
                        else:
                            rhs[ni, ki] = weight[d, k - C]
                    else:
                        rhs[ni, ki] = 0
                T.sync_threads()
                T.gemm(lhs, rhs, acc, transpose_B=True)
                T.sync_threads()
            for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                n = bx * BLOCK_M + mi
                d = by * BLOCK_N + ni
                if n < N and d < D:
                    if d < D_EDGE:
                        grad_q_edge[n, d] = acc[mi, ni]
                    else:
                        grad_q_node[n, d - D_EDGE] = acc[mi, ni]

    return double_backward_q


@tilelang.jit
def fused_sym_block_projection_double_weight(
    N, D_EDGE, D_NODE, C, HAS_Q_EDGE=True, HAS_Q_NODE=True,
    BLOCK_D=32, BLOCK_C=32, BLOCK_N=32,
):
    """Compute [u_Q;Q].T @ [p;s] as a logical K=2N GEMM."""
    D = D_EDGE + D_NODE

    @T.prim_func
    def double_backward_weight(
        u_q_edge: T.Tensor((N, D_EDGE), "float32"),
        u_q_node: T.Tensor((N, D_NODE), "float32"),
        q_edge: T.Tensor((N, D_EDGE), "float32"),
        q_node: T.Tensor((N, D_NODE), "float32"),
        p: T.Tensor((N, C), "float32"),
        s: T.Tensor((N, C), "float32"),
        grad_weight: T.Tensor((D, C), "float32"),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(C, BLOCK_C), threads=128) as (bx, by):
            lhs = T.alloc_shared((BLOCK_N, BLOCK_D), "float32")
            rhs = T.alloc_shared((BLOCK_N, BLOCK_C), "float32")
            acc = T.alloc_fragment((BLOCK_D, BLOCK_C), "float32")
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(2 * N, BLOCK_N), num_stages=2):
                for ni, di in T.Parallel(BLOCK_N, BLOCK_D):
                    k = ko * BLOCK_N + ni
                    d = bx * BLOCK_D + di
                    if k < 2 * N and d < D:
                        if k < N:
                            if d < D_EDGE:
                                if HAS_Q_EDGE:
                                    lhs[ni, di] = u_q_edge[k, d]
                                else:
                                    lhs[ni, di] = 0
                            else:
                                if HAS_Q_NODE:
                                    lhs[ni, di] = u_q_node[k, d - D_EDGE]
                                else:
                                    lhs[ni, di] = 0
                        else:
                            if d < D_EDGE:
                                lhs[ni, di] = q_edge[k - N, d]
                            else:
                                lhs[ni, di] = q_node[k - N, d - D_EDGE]
                    else:
                        lhs[ni, di] = 0
                for ni, ci in T.Parallel(BLOCK_N, BLOCK_C):
                    k = ko * BLOCK_N + ni
                    c = by * BLOCK_C + ci
                    if k < 2 * N and c < C:
                        if k < N:
                            rhs[ni, ci] = p[k, c]
                        else:
                            rhs[ni, ci] = s[k - N, c]
                    else:
                        rhs[ni, ci] = 0
                T.sync_threads()
                T.gemm(lhs, rhs, acc, transpose_A=True)
                T.sync_threads()
            for di, ci in T.Parallel(BLOCK_D, BLOCK_C):
                d = bx * BLOCK_D + di
                c = by * BLOCK_C + ci
                if d < D and c < C:
                    grad_weight[d, c] = acc[di, ci]

    return double_backward_weight


@tilelang.jit
def fused_sym_block_projection_double_vector(N, C, THREADS=128):
    """Reduce second-order bias and residual gradients."""

    @T.prim_func
    def double_backward_vector(
        grad_output: T.Tensor((N, C), "float32"),
        preact: T.Tensor((N, C), "float32"),
        t: T.Tensor((N, C), "float32"),
        s: T.Tensor((N, C), "float32"),
        threshold: T.float32,
        slope: T.float32,
        grad_bias: T.Tensor((C,), "float32"),
        grad_residual: T.Tensor((C,), "float32"),
    ):
        with T.Kernel(C, threads=THREADS) as (c,):
            tx = T.get_thread_binding()
            sh_b = T.alloc_shared((THREADS,), "float32")
            sh_r = T.alloc_shared((THREADS,), "float32")
            acc_b = T.alloc_var("float32", init=0)
            acc_r = T.alloc_var("float32", init=0)
            for n in T.serial(tx, N, THREADS):
                z = preact[n, c]
                deriv = T.alloc_var("float32")
                if z >= threshold:
                    th = T.tanh(slope * (z - threshold))
                    deriv = slope * (1.0 - th * th)
                else:
                    sig = 1.0 / (1.0 + T.exp(-z))
                    deriv = sig * (1.0 + z * (1.0 - sig))
                acc_b += s[n, c]
                acc_r += t[n, c] * grad_output[n, c] * deriv
            sh_b[tx] = acc_b
            sh_r[tx] = acc_r
            T.sync_threads()
            red_b = T.alloc_shared((1,), "float32")
            red_r = T.alloc_shared((1,), "float32")
            T.reduce_sum(sh_b, red_b, dim=0)
            T.reduce_sum(sh_r, red_r, dim=0)
            if tx == 0:
                grad_bias[c] = red_b[0]
                grad_residual[c] = red_r[0]

    return double_backward_vector


@tilelang.jit
def fused_sym_block_dual_double_owner_uniform(
    M, E_EDGE, E_NODE, N_NODE_EXT, NO, A,
    HAS_EDGE=True, HAS_NODE=True, HAS_H2=True, HAS_SW=True,
    THREADS=128,
):
    """Dual-branch owner part of the geometry double backward."""
    assert M % NO == 0
    EDGES_PER_OWNER = M // NO
    EDGE_ACTIVE = HAS_H2 or HAS_EDGE or HAS_SW
    NODE_ACTIVE = HAS_H2 or HAS_NODE or HAS_SW

    @T.prim_func
    def double_backward_owner_uniform(
        grad_q_edge: T.Tensor((NO, A * E_EDGE), "float32"),
        grad_q_node: T.Tensor((NO, A * E_NODE), "float32"),
        h_edge: T.Tensor((NO, 3 * E_EDGE), "float32"),
        h_node: T.Tensor((NO, 3 * E_NODE), "float32"),
        edge_ebd: T.Tensor((M, E_EDGE), "float32"),
        node_ebd_ext: T.Tensor((N_NODE_EXT, E_NODE), "float32"),
        h2: T.Tensor((M, 3), "float32"),
        sw: T.Tensor((M,), "float32"),
        n_ext2e_index: T.Tensor((M,), "int64"),
        u_edge: T.Tensor((M, E_EDGE), "float32"),
        u_node: T.Tensor((N_NODE_EXT, E_NODE), "float32"),
        u_h2: T.Tensor((M, 3), "float32"),
        u_sw: T.Tensor((M,), "float32"),
        scale: T.float32,
        grad_grad_q_edge: T.Tensor((NO, A * E_EDGE), "float32"),
        grad_grad_q_node: T.Tensor((NO, A * E_NODE), "float32"),
        grad_h_edge: T.Tensor((NO, 3 * E_EDGE), "float32"),
        grad_h_node: T.Tensor((NO, 3 * E_NODE), "float32"),
    ):
        with T.Kernel(NO, threads=THREADS) as (owner,):
            q_edge = T.alloc_shared((3, E_EDGE), "float32")
            q_node = T.alloc_shared((3, E_NODE), "float32")
            for b, d in T.Parallel(3, E_EDGE):
                acc = T.alloc_var("float32", init=0)
                if EDGE_ACTIVE:
                    for r in T.serial(EDGES_PER_OWNER):
                        edge = owner * EDGES_PER_OWNER + r
                        if HAS_H2:
                            acc += u_h2[edge, b] * edge_ebd[edge, d] * sw[edge]
                        if HAS_EDGE:
                            acc += u_edge[edge, d] * h2[edge, b] * sw[edge]
                        if HAS_SW:
                            acc += u_sw[edge] * h2[edge, b] * edge_ebd[edge, d]
                q_edge[b, d] = acc
            for b, d in T.Parallel(3, E_NODE):
                acc = T.alloc_var("float32", init=0)
                if NODE_ACTIVE:
                    for r in T.serial(EDGES_PER_OWNER):
                        edge = owner * EDGES_PER_OWNER + r
                        node = n_ext2e_index[edge]
                        value = node_ebd_ext[node, d]
                        if HAS_H2:
                            acc += u_h2[edge, b] * value * sw[edge]
                        if HAS_NODE:
                            acc += u_node[node, d] * h2[edge, b] * sw[edge]
                        if HAS_SW:
                            acc += u_sw[edge] * h2[edge, b] * value
                q_node[b, d] = acc
            T.sync_threads()
            c = scale / 3.0
            for a, d in T.Parallel(A, E_EDGE):
                acc = T.alloc_var("float32", init=0)
                if EDGE_ACTIVE:
                    for b in T.serial(3):
                        acc += (
                            h_edge[owner, b * E_EDGE + a] * q_edge[b, d]
                            + q_edge[b, a] * h_edge[owner, b * E_EDGE + d]
                        )
                grad_grad_q_edge[owner, a * E_EDGE + d] = acc * c
            for a, d in T.Parallel(A, E_NODE):
                acc = T.alloc_var("float32", init=0)
                if NODE_ACTIVE:
                    for b in T.serial(3):
                        acc += (
                            h_node[owner, b * E_NODE + a] * q_node[b, d]
                            + q_node[b, a] * h_node[owner, b * E_NODE + d]
                        )
                grad_grad_q_node[owner, a * E_NODE + d] = acc * c
            for b, d in T.Parallel(3, E_EDGE):
                acc = T.alloc_var("float32", init=0)
                if EDGE_ACTIVE:
                    for a in T.serial(A):
                        acc += q_edge[b, a] * grad_q_edge[owner, a * E_EDGE + d]
                    if d < A:
                        for k in T.serial(E_EDGE):
                            acc += q_edge[b, k] * grad_q_edge[owner, d * E_EDGE + k]
                grad_h_edge[owner, b * E_EDGE + d] = acc * c
            for b, d in T.Parallel(3, E_NODE):
                acc = T.alloc_var("float32", init=0)
                if NODE_ACTIVE:
                    for a in T.serial(A):
                        acc += q_node[b, a] * grad_q_node[owner, a * E_NODE + d]
                    if d < A:
                        for k in T.serial(E_NODE):
                            acc += q_node[b, k] * grad_q_node[owner, d * E_NODE + k]
                grad_h_node[owner, b * E_NODE + d] = acc * c

    return double_backward_owner_uniform


@tilelang.jit
def fused_sym_block_dual_double_owner_segmented(
    M, E_EDGE, E_NODE, N_NODE_EXT, NO, A,
    HAS_EDGE=True, HAS_NODE=True, HAS_H2=True, HAS_SW=True,
    THREADS=128,
):
    """Segmented-owner variant of the dual geometry double backward."""
    EDGE_ACTIVE = HAS_H2 or HAS_EDGE or HAS_SW
    NODE_ACTIVE = HAS_H2 or HAS_NODE or HAS_SW

    @T.prim_func
    def double_backward_owner_segmented(
        grad_q_edge: T.Tensor((NO, A * E_EDGE), "float32"),
        grad_q_node: T.Tensor((NO, A * E_NODE), "float32"),
        h_edge: T.Tensor((NO, 3 * E_EDGE), "float32"),
        h_node: T.Tensor((NO, 3 * E_NODE), "float32"),
        edge_ebd: T.Tensor((M, E_EDGE), "float32"),
        node_ebd_ext: T.Tensor((N_NODE_EXT, E_NODE), "float32"),
        h2: T.Tensor((M, 3), "float32"),
        sw: T.Tensor((M,), "float32"),
        n_ext2e_index: T.Tensor((M,), "int64"),
        u_edge: T.Tensor((M, E_EDGE), "float32"),
        u_node: T.Tensor((N_NODE_EXT, E_NODE), "float32"),
        u_h2: T.Tensor((M, 3), "float32"),
        u_sw: T.Tensor((M,), "float32"),
        offsets: T.Tensor((NO + 1,), "int64"),
        order: T.Tensor((M,), "int64"),
        scale: T.float32,
        grad_grad_q_edge: T.Tensor((NO, A * E_EDGE), "float32"),
        grad_grad_q_node: T.Tensor((NO, A * E_NODE), "float32"),
        grad_h_edge: T.Tensor((NO, 3 * E_EDGE), "float32"),
        grad_h_node: T.Tensor((NO, 3 * E_NODE), "float32"),
    ):
        with T.Kernel(NO, threads=THREADS) as (owner,):
            q_edge = T.alloc_shared((3, E_EDGE), "float32")
            q_node = T.alloc_shared((3, E_NODE), "float32")
            for b, d in T.Parallel(3, E_EDGE):
                acc = T.alloc_var("float32", init=0)
                if EDGE_ACTIVE:
                    for r in T.serial(offsets[owner], offsets[owner + 1]):
                        edge = order[r]
                        if HAS_H2:
                            acc += u_h2[edge, b] * edge_ebd[edge, d] * sw[edge]
                        if HAS_EDGE:
                            acc += u_edge[edge, d] * h2[edge, b] * sw[edge]
                        if HAS_SW:
                            acc += u_sw[edge] * h2[edge, b] * edge_ebd[edge, d]
                q_edge[b, d] = acc
            for b, d in T.Parallel(3, E_NODE):
                acc = T.alloc_var("float32", init=0)
                if NODE_ACTIVE:
                    for r in T.serial(offsets[owner], offsets[owner + 1]):
                        edge = order[r]
                        node = n_ext2e_index[edge]
                        value = node_ebd_ext[node, d]
                        if HAS_H2:
                            acc += u_h2[edge, b] * value * sw[edge]
                        if HAS_NODE:
                            acc += u_node[node, d] * h2[edge, b] * sw[edge]
                        if HAS_SW:
                            acc += u_sw[edge] * h2[edge, b] * value
                q_node[b, d] = acc
            T.sync_threads()
            c = scale / 3.0
            for a, d in T.Parallel(A, E_EDGE):
                acc = T.alloc_var("float32", init=0)
                if EDGE_ACTIVE:
                    for b in T.serial(3):
                        acc += (
                            h_edge[owner, b * E_EDGE + a] * q_edge[b, d]
                            + q_edge[b, a] * h_edge[owner, b * E_EDGE + d]
                        )
                grad_grad_q_edge[owner, a * E_EDGE + d] = acc * c
            for a, d in T.Parallel(A, E_NODE):
                acc = T.alloc_var("float32", init=0)
                if NODE_ACTIVE:
                    for b in T.serial(3):
                        acc += (
                            h_node[owner, b * E_NODE + a] * q_node[b, d]
                            + q_node[b, a] * h_node[owner, b * E_NODE + d]
                        )
                grad_grad_q_node[owner, a * E_NODE + d] = acc * c
            for b, d in T.Parallel(3, E_EDGE):
                acc = T.alloc_var("float32", init=0)
                if EDGE_ACTIVE:
                    for a in T.serial(A):
                        acc += q_edge[b, a] * grad_q_edge[owner, a * E_EDGE + d]
                    if d < A:
                        for k in T.serial(E_EDGE):
                            acc += q_edge[b, k] * grad_q_edge[owner, d * E_EDGE + k]
                grad_h_edge[owner, b * E_EDGE + d] = acc * c
            for b, d in T.Parallel(3, E_NODE):
                acc = T.alloc_var("float32", init=0)
                if NODE_ACTIVE:
                    for a in T.serial(A):
                        acc += q_node[b, a] * grad_q_node[owner, a * E_NODE + d]
                    if d < A:
                        for k in T.serial(E_NODE):
                            acc += q_node[b, k] * grad_q_node[owner, d * E_NODE + k]
                grad_h_node[owner, b * E_NODE + d] = acc * c

    return double_backward_owner_segmented


@tilelang.jit
def fused_sym_block_dual_double_edge(
    M, E_EDGE, E_NODE, N_NODE_EXT, NO,
    HAS_EDGE=True, HAS_NODE=True, HAS_H2=True, HAS_SW=True,
    THREADS=128,
):
    """Edge part of the dual geometry double backward."""
    EDGE_ACTIVE = HAS_H2 or HAS_EDGE or HAS_SW
    NODE_ACTIVE = HAS_H2 or HAS_NODE or HAS_SW

    @T.prim_func
    def double_backward_edge(
        flat_r_edge: T.Tensor((M, 3 * E_EDGE), "float32"),
        flat_r_node: T.Tensor((M, 3 * E_NODE), "float32"),
        grad_h_edge: T.Tensor((NO, 3 * E_EDGE), "float32"),
        grad_h_node: T.Tensor((NO, 3 * E_NODE), "float32"),
        edge_ebd: T.Tensor((M, E_EDGE), "float32"),
        node_ebd_ext: T.Tensor((N_NODE_EXT, E_NODE), "float32"),
        h2: T.Tensor((M, 3), "float32"),
        sw: T.Tensor((M,), "float32"),
        owner: T.Tensor((M,), "int64"),
        n_ext2e_index: T.Tensor((M,), "int64"),
        u_edge: T.Tensor((M, E_EDGE), "float32"),
        u_node: T.Tensor((N_NODE_EXT, E_NODE), "float32"),
        u_h2: T.Tensor((M, 3), "float32"),
        u_sw: T.Tensor((M,), "float32"),
        scale: T.float32,
        grad_edge_ebd: T.Tensor((M, E_EDGE), "float32"),
        grad_node_ebd_ext: T.Tensor((N_NODE_EXT, E_NODE), "float32"),
        grad_h2_out: T.Tensor((M, 3), "float32"),
        grad_sw_out: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M, threads=THREADS) as (edge,):
            tx = T.get_thread_binding()
            o = owner[edge]
            node = n_ext2e_index[edge]
            w = sw[edge]
            acc_h = T.alloc_fragment((3,), "float32")
            acc_sw = T.alloc_var("float32", init=0)
            T.clear(acc_h)

            for d in T.serial(tx, E_EDGE, THREADS):
                out_value = T.alloc_var("float32", init=0)
                if EDGE_ACTIVE:
                    edge_value = edge_ebd[edge, d]
                    for b in T.serial(3):
                        r = flat_r_edge[edge, b * E_EDGE + d]
                        gh = grad_h_edge[o, b * E_EDGE + d] * scale
                        if HAS_H2:
                            out_value += u_h2[edge, b] * r * w
                            acc_sw += u_h2[edge, b] * edge_value * r
                        if HAS_EDGE:
                            acc_h[b] += u_edge[edge, d] * w * r
                            acc_sw += u_edge[edge, d] * h2[edge, b] * r
                        if HAS_SW:
                            out_value += u_sw[edge] * r * h2[edge, b]
                            acc_h[b] += u_sw[edge] * edge_value * r
                        out_value += gh * h2[edge, b] * w
                        acc_h[b] += gh * edge_value * w
                        acc_sw += gh * h2[edge, b] * edge_value
                grad_edge_ebd[edge, d] = out_value

            for d in T.serial(tx, E_NODE, THREADS):
                if NODE_ACTIVE:
                    node_value = node_ebd_ext[node, d]
                    out_value = T.alloc_var("float32", init=0)
                    for b in T.serial(3):
                        r = flat_r_node[edge, b * E_NODE + d]
                        gh = grad_h_node[o, b * E_NODE + d] * scale
                        if HAS_H2:
                            out_value += u_h2[edge, b] * r * w
                            acc_sw += u_h2[edge, b] * node_value * r
                        if HAS_NODE:
                            acc_h[b] += u_node[node, d] * w * r
                            acc_sw += u_node[node, d] * h2[edge, b] * r
                        if HAS_SW:
                            out_value += u_sw[edge] * r * h2[edge, b]
                            acc_h[b] += u_sw[edge] * node_value * r
                        out_value += gh * h2[edge, b] * w
                        acc_h[b] += gh * node_value * w
                        acc_sw += gh * h2[edge, b] * node_value
                    T.atomic_add(grad_node_ebd_ext[node, d], out_value)

            sh_h = T.alloc_shared((3, THREADS), "float32")
            sh_sw = T.alloc_shared((THREADS,), "float32")
            for b in T.Parallel(3):
                sh_h[b, tx] = acc_h[b]
            sh_sw[tx] = acc_sw
            T.sync_threads()
            red_h = T.alloc_shared((3,), "float32")
            red_sw = T.alloc_shared((1,), "float32")
            T.reduce_sum(sh_h, red_h, dim=1)
            T.reduce_sum(sh_sw, red_sw, dim=0)
            if tx == 0:
                for b in T.serial(3):
                    grad_h2_out[edge, b] = red_h[b]
                grad_sw_out[edge] = red_sw[0]

    return double_backward_edge


class FusedDualSymGeometryFunction(torch.autograd.Function):
    """Online node gather plus the two materialized grouped Gram products."""

    @staticmethod
    def forward(
        ctx,
        edge_ebd,
        node_ebd_ext,
        h2,
        sw,
        owner,
        n_ext2e_index,
        num_owner,
        scale_factor,
        axis_neuron,
        owner_metadata,
    ):
        edge_ebd = edge_ebd.contiguous()
        node_ebd_ext = node_ebd_ext.contiguous()
        h2 = h2.contiguous()
        sw = sw.contiguous()
        owner = owner.long().contiguous()
        n_ext2e_index = n_ext2e_index.long().contiguous()
        M, e_edge = edge_ebd.shape
        n_node_ext, e_node = node_ebd_ext.shape
        h_edge = torch.empty(
            (num_owner, 3 * e_edge), device=edge_ebd.device, dtype=edge_ebd.dtype
        )
        h_node = torch.empty(
            (num_owner, 3 * e_node), device=edge_ebd.device, dtype=edge_ebd.dtype
        )
        metadata = owner_metadata
        uniform, offsets, order = metadata
        if M == 0:
            h_edge.zero_()
            h_node.zero_()
        elif uniform:
            hg_forward_kernel = fused_sym_block_dual_hg_forward_uniform(
                M=M, E_EDGE=e_edge, E_NODE=e_node,
                N_NODE_EXT=n_node_ext, NO=num_owner,
            )
            hg_forward_kernel(
                edge_ebd, node_ebd_ext, h2, sw, n_ext2e_index,
                float(scale_factor), h_edge, h_node,
            )
        else:
            hg_forward_kernel = fused_sym_block_dual_hg_forward_segmented(
                M=M, E_EDGE=e_edge, E_NODE=e_node,
                N_NODE_EXT=n_node_ext, NO=num_owner,
            )
            hg_forward_kernel(
                edge_ebd, node_ebd_ext, h2, sw, n_ext2e_index,
                offsets, order, float(scale_factor), h_edge, h_node,
            )

        q_edge = torch.empty(
            (num_owner, axis_neuron * e_edge),
            device=edge_ebd.device, dtype=edge_ebd.dtype,
        )
        q_node = torch.empty(
            (num_owner, axis_neuron * e_node),
            device=edge_ebd.device, dtype=edge_ebd.dtype,
        )
        grrg_kernel = fused_sym_block_dual_grrg_forward(
            NO=num_owner, E_EDGE=e_edge, E_NODE=e_node, A=axis_neuron,
        )
        grrg_kernel(h_edge, h_node, q_edge, q_node)

        ctx.owner_metadata = metadata
        ctx.num_owner = num_owner
        ctx.scale_factor = scale_factor
        ctx.axis_neuron = axis_neuron
        ctx.save_for_backward(
            edge_ebd, node_ebd_ext, h2, sw, owner, n_ext2e_index,
            h_edge, h_node,
        )
        return q_edge, q_node

    @staticmethod
    def backward(ctx, grad_q_edge, grad_q_node):
        (
            edge_ebd, node_ebd_ext, h2, sw, owner, n_ext2e_index,
            h_edge, h_node,
        ) = ctx.saved_tensors
        grad_edge, grad_node, grad_h2, grad_sw = (
            FusedDualSymGeometryFunctionBackward.apply(
                grad_q_edge, grad_q_node,
                edge_ebd, node_ebd_ext, h2, sw,
                owner, n_ext2e_index, h_edge, h_node,
                ctx.num_owner, ctx.scale_factor, ctx.axis_neuron,
                ctx.owner_metadata,
            )
        )
        return (
            grad_edge, grad_node, grad_h2, grad_sw,
            None, None, None, None, None, None,
        )


class FusedDualSymGeometryFunctionBackward(torch.autograd.Function):
    """First derivative of the wider geometry block, with fused double backward."""

    @staticmethod
    def forward(
        ctx,
        grad_q_edge,
        grad_q_node,
        edge_ebd,
        node_ebd_ext,
        h2,
        sw,
        owner,
        n_ext2e_index,
        h_edge,
        h_node,
        num_owner,
        scale_factor,
        axis_neuron,
        owner_metadata,
    ):
        grad_q_edge = grad_q_edge.contiguous()
        grad_q_node = grad_q_node.contiguous()
        M, e_edge = edge_ebd.shape
        n_node_ext, e_node = node_ebd_ext.shape
        grad_h_edge = torch.empty_like(h_edge)
        grad_h_node = torch.empty_like(h_node)
        grrg_backward = fused_sym_block_dual_grrg_backward(
            NO=num_owner, E_EDGE=e_edge, E_NODE=e_node, A=axis_neuron,
        )
        grrg_backward(
            grad_q_edge, grad_q_node, h_edge, h_node,
            float(scale_factor), grad_h_edge, grad_h_node,
        )

        grad_flat_h_edge = torch.empty(
            (M, 3 * e_edge), device=edge_ebd.device, dtype=edge_ebd.dtype,
        )
        grad_flat_h_node = torch.empty(
            (M, 3 * e_node), device=edge_ebd.device, dtype=edge_ebd.dtype,
        )
        grad_edge = torch.empty_like(edge_ebd)
        grad_node = torch.zeros_like(node_ebd_ext)
        grad_h2 = torch.empty_like(h2)
        grad_sw = torch.empty_like(sw)
        if M:
            hg_backward = fused_sym_block_dual_hg_backward(
                M=M, E_EDGE=e_edge, E_NODE=e_node,
                N_NODE_EXT=n_node_ext, NO=num_owner,
            )
            hg_backward(
                grad_h_edge, grad_h_node,
                edge_ebd, node_ebd_ext, h2, sw,
                owner, n_ext2e_index,
                grad_flat_h_edge, grad_flat_h_node,
                grad_edge, grad_node, grad_h2, grad_sw,
            )
        else:
            grad_edge.zero_()
            grad_h2.zero_()
            grad_sw.zero_()

        ctx.set_materialize_grads(False)
        ctx.owner_metadata = owner_metadata
        ctx.num_owner = num_owner
        ctx.scale_factor = scale_factor
        ctx.axis_neuron = axis_neuron
        ctx.save_for_backward(
            grad_q_edge, grad_q_node,
            edge_ebd, node_ebd_ext, h2, sw,
            owner, n_ext2e_index, h_edge, h_node,
            grad_flat_h_edge, grad_flat_h_node,
        )
        return grad_edge, grad_node, grad_h2, grad_sw

    @staticmethod
    def backward(ctx, u_edge, u_node, u_h2, u_sw):
        (
            grad_q_edge, grad_q_node,
            edge_ebd, node_ebd_ext, h2, sw,
            owner, n_ext2e_index, h_edge, h_node,
            flat_r_edge, flat_r_node,
        ) = ctx.saved_tensors
        M, e_edge = edge_ebd.shape
        n_node_ext, e_node = node_ebd_ext.shape
        num_owner = ctx.num_owner
        if M == 0:
            return (
                torch.zeros_like(grad_q_edge), torch.zeros_like(grad_q_node),
                torch.zeros_like(edge_ebd), torch.zeros_like(node_ebd_ext),
                torch.zeros_like(h2), torch.zeros_like(sw),
                None, None, None, None, None, None, None, None,
            )
        has_edge = u_edge is not None
        has_node = u_node is not None
        has_h2 = u_h2 is not None
        has_sw = u_sw is not None
        u_edge = None if u_edge is None else u_edge.contiguous()
        u_node = None if u_node is None else u_node.contiguous()
        u_h2 = None if u_h2 is None else u_h2.contiguous()
        u_sw = None if u_sw is None else u_sw.contiguous()

        gg_q_edge = torch.empty_like(grad_q_edge)
        gg_q_node = torch.empty_like(grad_q_node)
        grad_h_edge = torch.empty_like(h_edge)
        grad_h_node = torch.empty_like(h_node)
        uniform, offsets, order = ctx.owner_metadata
        factory = (
            fused_sym_block_dual_double_owner_uniform
            if uniform else fused_sym_block_dual_double_owner_segmented
        )
        owner_kernel = factory(
            M=M, E_EDGE=e_edge, E_NODE=e_node,
            N_NODE_EXT=n_node_ext, NO=num_owner, A=ctx.axis_neuron,
            HAS_EDGE=has_edge, HAS_NODE=has_node,
            HAS_H2=has_h2, HAS_SW=has_sw,
        )
        metadata_args = () if uniform else (offsets, order)
        owner_kernel(
            grad_q_edge, grad_q_node, h_edge, h_node,
            edge_ebd, node_ebd_ext, h2, sw, n_ext2e_index,
            u_edge if has_edge else edge_ebd,
            u_node if has_node else node_ebd_ext,
            u_h2 if has_h2 else h2,
            u_sw if has_sw else sw,
            *metadata_args, float(ctx.scale_factor),
            gg_q_edge, gg_q_node, grad_h_edge, grad_h_node,
        )

        grad_edge = torch.empty_like(edge_ebd)
        grad_node = torch.zeros_like(node_ebd_ext)
        grad_h2_out = torch.empty_like(h2)
        grad_sw_out = torch.empty_like(sw)
        edge_kernel = fused_sym_block_dual_double_edge(
            M=M, E_EDGE=e_edge, E_NODE=e_node,
            N_NODE_EXT=n_node_ext, NO=num_owner,
            HAS_EDGE=has_edge, HAS_NODE=has_node,
            HAS_H2=has_h2, HAS_SW=has_sw,
        )
        edge_kernel(
            flat_r_edge, flat_r_node, grad_h_edge, grad_h_node,
            edge_ebd, node_ebd_ext, h2, sw, owner, n_ext2e_index,
            u_edge if has_edge else edge_ebd,
            u_node if has_node else node_ebd_ext,
            u_h2 if has_h2 else h2,
            u_sw if has_sw else sw,
            float(ctx.scale_factor),
            grad_edge, grad_node, grad_h2_out, grad_sw_out,
        )
        return (
            gg_q_edge, gg_q_node,
            grad_edge, grad_node, grad_h2_out, grad_sw_out,
            None, None, None, None, None, None, None, None,
        )


class FusedSymProjectionActResidualFunction(torch.autograd.Function):
    """No-cat projection, custom-SiLU, and trainable residual accumulation."""

    @staticmethod
    def forward(
        ctx, node_base, q_edge, q_node, weight, bias, residual,
        threshold, slope, const_value,
    ):
        node_base = node_base.contiguous()
        q_edge = q_edge.contiguous()
        q_node = q_node.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        residual = residual.contiguous()
        N, C = node_base.shape
        d_edge = q_edge.shape[1]
        d_node = q_node.shape[1]
        preact = torch.empty_like(node_base)
        out = torch.empty_like(node_base)
        projection_forward_kernel = fused_sym_block_projection_act_residual_forward(
            N=N, D_EDGE=d_edge, D_NODE=d_node, C=C,
        )
        projection_forward_kernel(
            node_base, q_edge, q_node, weight, bias, residual,
            float(threshold), float(slope), float(const_value), preact, out,
        )
        ctx.threshold = threshold
        ctx.slope = slope
        ctx.const_value = const_value
        ctx.save_for_backward(q_edge, q_node, weight, bias, residual, preact)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q_edge, q_node, weight, bias, residual, preact = ctx.saved_tensors
        outputs = FusedSymProjectionActResidualFunctionBackward.apply(
            grad_output, q_edge, q_node, weight, bias, residual, preact,
            ctx.threshold, ctx.slope, ctx.const_value,
        )
        return (*outputs, None, None, None)


class FusedSymProjectionActResidualFunctionBackward(torch.autograd.Function):
    """First derivative of the no-cat projection, including double backward."""

    @staticmethod
    def forward(
        ctx, grad_output, q_edge, q_node, weight, bias, residual, preact,
        threshold, slope, const_value,
    ):
        grad_output = grad_output.contiguous()
        N, C = grad_output.shape
        d_edge = q_edge.shape[1]
        d_node = q_node.shape[1]
        grad_q_edge = torch.empty_like(q_edge)
        grad_q_node = torch.empty_like(q_node)
        q_kernel = fused_sym_block_projection_backward_q(
            N=N, D_EDGE=d_edge, D_NODE=d_node, C=C,
        )
        q_kernel(
            grad_output, preact, weight, residual,
            float(threshold), float(slope), grad_q_edge, grad_q_node,
        )
        grad_weight = torch.empty_like(weight)
        w_kernel = fused_sym_block_projection_backward_weight(
            N=N, D_EDGE=d_edge, D_NODE=d_node, C=C,
        )
        w_kernel(
            grad_output, preact, q_edge, q_node, residual,
            float(threshold), float(slope), grad_weight,
        )
        grad_bias = torch.empty_like(bias)
        grad_residual = torch.empty_like(residual)
        v_kernel = fused_sym_block_projection_backward_vector(N=N, C=C)
        v_kernel(
            grad_output, preact, residual,
            float(threshold), float(slope), float(const_value),
            grad_bias, grad_residual,
        )
        ctx.set_materialize_grads(False)
        ctx.threshold = threshold
        ctx.slope = slope
        ctx.const_value = const_value
        ctx.save_for_backward(
            grad_output, q_edge, q_node, weight, bias, residual, preact,
        )
        grad_node_base = grad_output
        return (
            grad_node_base, grad_q_edge, grad_q_node,
            grad_weight, grad_bias, grad_residual,
        )

    @staticmethod
    def backward(
        ctx, u_base, u_q_edge, u_q_node, u_weight, u_bias, u_residual,
    ):
        grad_output, q_edge, q_node, weight, bias, residual, preact = ctx.saved_tensors
        N, C = grad_output.shape
        d_edge = q_edge.shape[1]
        d_node = q_node.shape[1]
        has_base = u_base is not None
        has_q_edge = u_q_edge is not None
        has_q_node = u_q_node is not None
        has_weight = u_weight is not None
        has_bias = u_bias is not None
        has_residual = u_residual is not None
        u_base = None if u_base is None else u_base.contiguous()
        u_q_edge = None if u_q_edge is None else u_q_edge.contiguous()
        u_q_node = None if u_q_node is None else u_q_node.contiguous()
        u_weight = None if u_weight is None else u_weight.contiguous()
        u_bias = None if u_bias is None else u_bias.contiguous()
        u_residual = None if u_residual is None else u_residual.contiguous()

        p = torch.empty_like(grad_output)
        t = torch.empty_like(grad_output)
        s = torch.empty_like(grad_output)
        grad_grad_output = torch.empty_like(grad_output)
        ts_kernel = fused_sym_block_projection_double_ts_output(
            N=N, D_EDGE=d_edge, D_NODE=d_node, C=C,
            HAS_BASE=has_base, HAS_Q_EDGE=has_q_edge, HAS_Q_NODE=has_q_node,
            HAS_WEIGHT=has_weight, HAS_BIAS=has_bias,
            HAS_RESIDUAL=has_residual,
        )
        ts_kernel(
            grad_output, q_edge, q_node, weight, residual, preact,
            u_base if has_base else grad_output,
            u_q_edge if has_q_edge else q_edge,
            u_q_node if has_q_node else q_node,
            u_weight if has_weight else weight,
            u_bias if has_bias else bias,
            u_residual if has_residual else residual,
            float(ctx.threshold), float(ctx.slope), float(ctx.const_value),
            p, t, s, grad_grad_output,
        )

        grad_q_edge = torch.empty_like(q_edge)
        grad_q_node = torch.empty_like(q_node)
        q_kernel = fused_sym_block_projection_double_q(
            N=N, D_EDGE=d_edge, D_NODE=d_node, C=C,
            HAS_WEIGHT=has_weight,
        )
        q_kernel(
            p, s, weight, u_weight if has_weight else weight,
            grad_q_edge, grad_q_node,
        )

        grad_weight = torch.empty_like(weight)
        w_kernel = fused_sym_block_projection_double_weight(
            N=N, D_EDGE=d_edge, D_NODE=d_node, C=C,
            HAS_Q_EDGE=has_q_edge, HAS_Q_NODE=has_q_node,
        )
        w_kernel(
            u_q_edge if has_q_edge else q_edge,
            u_q_node if has_q_node else q_node,
            q_edge, q_node, p, s, grad_weight,
        )

        grad_bias = torch.empty_like(bias)
        grad_residual = torch.empty_like(residual)
        v_kernel = fused_sym_block_projection_double_vector(N=N, C=C)
        v_kernel(
            grad_output, preact, t, s,
            float(ctx.threshold), float(ctx.slope),
            grad_bias, grad_residual,
        )
        return (
            grad_grad_output, grad_q_edge, grad_q_node,
            grad_weight, grad_bias, grad_residual,
            None, None, None, None,
        )


def fused_sym_block_dynamic(
    node_base,
    edge_ebd,
    node_ebd_ext,
    h2,
    sw,
    owner,
    n_ext2e_index,
    projection_weight,
    projection_bias,
    residual,
    num_owner,
    nb,
    nloc,
    scale_factor,
    axis_neuron,
    threshold,
    slope,
    const_value,
    owner_metadata,
):
    """Compose the two new twice-differentiable Sym sub-operators."""
    node_shape = node_base.shape
    node_base_flat = node_base.reshape(num_owner, node_shape[-1])
    node_ext_flat = node_ebd_ext.reshape(-1, node_ebd_ext.shape[-1])
    q_edge, q_node = FusedDualSymGeometryFunction.apply(
        edge_ebd, node_ext_flat, h2, sw, owner, n_ext2e_index,
        num_owner, scale_factor, axis_neuron, owner_metadata,
    )
    out = FusedSymProjectionActResidualFunction.apply(
        node_base_flat, q_edge, q_node,
        projection_weight, projection_bias, residual,
        threshold, slope, const_value,
    )
    return out.reshape(nb, nloc, node_shape[-1])


# ============================================================================
# Wider RepFlow edge block
#
# The two projections share the same logical input
# [central_node | neighbour_node | edge].  They are evaluated as one logical
# output-concatenated GEMM.  The owner metadata is supplied by forward_fused
# and is shared with the wider Sym block.
# ============================================================================


@tilelang.jit
def fused_edge_block_projection_forward(
    M, N_NODE, N_NODE_EXT, C_NODE, C_EDGE,
    BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
):
    D = 2 * C_NODE + C_EDGE
    C = C_NODE + C_EDGE

    @T.prim_func
    def edge_projection_forward(
        node_ebd: T.Tensor((N_NODE, C_NODE), "float32"),
        node_ebd_ext: T.Tensor((N_NODE_EXT, C_NODE), "float32"),
        edge_ebd: T.Tensor((M, C_EDGE), "float32"),
        owner: T.Tensor((M,), "int64"),
        n_ext2e_index: T.Tensor((M,), "int64"),
        node_weight: T.Tensor((D, C_NODE), "float32"),
        node_bias: T.Tensor((C_NODE,), "float32"),
        edge_weight: T.Tensor((D, C_EDGE), "float32"),
        edge_bias: T.Tensor((C_EDGE,), "float32"),
        edge_residual: T.Tensor((C_EDGE,), "float32"),
        threshold: T.float32,
        slope: T.float32,
        const_value: T.float32,
        node_preact: T.Tensor((M, C_NODE), "float32"),
        edge_preact: T.Tensor((M, C_EDGE), "float32"),
        edge_partial: T.Tensor((M, C_EDGE), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(C, BLOCK_N), threads=128) as (bx, by):
            lhs = T.alloc_shared((BLOCK_M, BLOCK_K), "float32")
            rhs = T.alloc_shared((BLOCK_K, BLOCK_N), "float32")
            acc = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(D, BLOCK_K), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    edge = bx * BLOCK_M + mi
                    d = ko * BLOCK_K + ki
                    if edge < M and d < D:
                        if d < C_NODE:
                            lhs[mi, ki] = node_ebd[owner[edge], d]
                        elif d < 2 * C_NODE:
                            lhs[mi, ki] = node_ebd_ext[n_ext2e_index[edge], d - C_NODE]
                        else:
                            lhs[mi, ki] = edge_ebd[edge, d - 2 * C_NODE]
                    else:
                        lhs[mi, ki] = 0
                for ki, ni in T.Parallel(BLOCK_K, BLOCK_N):
                    d = ko * BLOCK_K + ki
                    c = by * BLOCK_N + ni
                    if d < D and c < C:
                        if c < C_NODE:
                            rhs[ki, ni] = node_weight[d, c]
                        else:
                            rhs[ki, ni] = edge_weight[d, c - C_NODE]
                    else:
                        rhs[ki, ni] = 0
                T.sync_threads()
                T.gemm(lhs, rhs, acc)
                T.sync_threads()
            for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                edge = bx * BLOCK_M + mi
                c = by * BLOCK_N + ni
                if edge < M and c < C:
                    if c < C_NODE:
                        node_preact[edge, c] = acc[mi, ni] + node_bias[c]
                    else:
                        ce = c - C_NODE
                        z = acc[mi, ni] + edge_bias[ce]
                        value = T.alloc_var("float32")
                        if z >= threshold:
                            value = T.tanh(slope * (z - threshold)) + const_value
                        else:
                            sig = 1.0 / (1.0 + T.exp(-z))
                            value = z * sig
                        edge_preact[edge, ce] = z
                        edge_partial[edge, ce] = edge_ebd[edge, ce] + edge_residual[ce] * value

    return edge_projection_forward


@tilelang.jit
def fused_edge_block_node_reduce_forward_uniform(
    M, NO, C_NODE, THREADS=128,
):
    assert M % NO == 0
    EDGES_PER_OWNER = M // NO

    @T.prim_func
    def edge_node_reduce_forward_uniform(
        node_partial: T.Tensor((NO, C_NODE), "float32"),
        node_preact: T.Tensor((M, C_NODE), "float32"),
        sw: T.Tensor((M,), "float32"),
        node_residual: T.Tensor((C_NODE,), "float32"),
        scale: T.float32,
        threshold: T.float32,
        slope: T.float32,
        const_value: T.float32,
        node_output: T.Tensor((NO, C_NODE), "float32"),
    ):
        with T.Kernel(NO, T.ceildiv(C_NODE, THREADS), threads=THREADS) as (owner_id, tile):
            tx = T.get_thread_binding()
            c = tile * THREADS + tx
            if c < C_NODE:
                acc = T.alloc_var("float32", init=0)
                for r in T.serial(EDGES_PER_OWNER):
                    edge = owner_id * EDGES_PER_OWNER + r
                    z = node_preact[edge, c]
                    value = T.alloc_var("float32")
                    if z >= threshold:
                        value = T.tanh(slope * (z - threshold)) + const_value
                    else:
                        sig = 1.0 / (1.0 + T.exp(-z))
                        value = z * sig
                    acc += sw[edge] * value
                node_output[owner_id, c] = node_partial[owner_id, c] + node_residual[c] * scale * acc

    return edge_node_reduce_forward_uniform


@tilelang.jit
def fused_edge_block_node_reduce_forward_segmented(
    M, NO, C_NODE, THREADS=128,
):
    @T.prim_func
    def edge_node_reduce_forward_segmented(
        node_partial: T.Tensor((NO, C_NODE), "float32"),
        node_preact: T.Tensor((M, C_NODE), "float32"),
        sw: T.Tensor((M,), "float32"),
        node_residual: T.Tensor((C_NODE,), "float32"),
        offsets: T.Tensor((NO + 1,), "int64"),
        order: T.Tensor((M,), "int64"),
        scale: T.float32,
        threshold: T.float32,
        slope: T.float32,
        const_value: T.float32,
        node_output: T.Tensor((NO, C_NODE), "float32"),
    ):
        with T.Kernel(NO, T.ceildiv(C_NODE, THREADS), threads=THREADS) as (owner_id, tile):
            tx = T.get_thread_binding()
            c = tile * THREADS + tx
            if c < C_NODE:
                acc = T.alloc_var("float32", init=0)
                for r in T.serial(offsets[owner_id], offsets[owner_id + 1]):
                    edge = order[r]
                    z = node_preact[edge, c]
                    value = T.alloc_var("float32")
                    if z >= threshold:
                        value = T.tanh(slope * (z - threshold)) + const_value
                    else:
                        sig = 1.0 / (1.0 + T.exp(-z))
                        value = z * sig
                    acc += sw[edge] * value
                node_output[owner_id, c] = node_partial[owner_id, c] + node_residual[c] * scale * acc

    return edge_node_reduce_forward_segmented


@tilelang.jit
def fused_edge_block_backward_inputs(
    M, N_NODE, N_NODE_EXT, C_NODE, C_EDGE,
    HAS_GRAD_NODE=True, HAS_GRAD_EDGE=True,
    BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
):
    D = 2 * C_NODE + C_EDGE
    C = C_NODE + C_EDGE

    @T.prim_func
    def edge_backward_inputs(
        grad_node_output: T.Tensor((N_NODE, C_NODE), "float32"),
        grad_edge_output: T.Tensor((M, C_EDGE), "float32"),
        node_preact: T.Tensor((M, C_NODE), "float32"),
        edge_preact: T.Tensor((M, C_EDGE), "float32"),
        sw: T.Tensor((M,), "float32"),
        owner: T.Tensor((M,), "int64"),
        n_ext2e_index: T.Tensor((M,), "int64"),
        node_weight: T.Tensor((D, C_NODE), "float32"),
        node_residual: T.Tensor((C_NODE,), "float32"),
        edge_weight: T.Tensor((D, C_EDGE), "float32"),
        edge_residual: T.Tensor((C_EDGE,), "float32"),
        scale: T.float32,
        threshold: T.float32,
        slope: T.float32,
        grad_node_ebd: T.Tensor((N_NODE, C_NODE), "float32"),
        grad_node_ebd_ext: T.Tensor((N_NODE_EXT, C_NODE), "float32"),
        grad_edge_ebd: T.Tensor((M, C_EDGE), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(D, BLOCK_N), threads=128) as (bx, by):
            p = T.alloc_shared((BLOCK_M, BLOCK_K), "float32")
            w = T.alloc_shared((BLOCK_N, BLOCK_K), "float32")
            acc = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(C, BLOCK_K), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    edge = bx * BLOCK_M + mi
                    c = ko * BLOCK_K + ki
                    if edge < M and c < C:
                        if c < C_NODE:
                            if HAS_GRAD_NODE:
                                node_z = node_preact[edge, c]
                                node_deriv = T.alloc_var("float32")
                                if node_z >= threshold:
                                    node_th = T.tanh(slope * (node_z - threshold))
                                    node_deriv = slope * (1.0 - node_th * node_th)
                                else:
                                    node_sig = 1.0 / (1.0 + T.exp(-node_z))
                                    node_deriv = node_sig * (1.0 + node_z * (1.0 - node_sig))
                                p[mi, ki] = grad_node_output[owner[edge], c] * node_residual[c] * scale * sw[edge] * node_deriv
                            else:
                                p[mi, ki] = 0
                        else:
                            ce = c - C_NODE
                            if HAS_GRAD_EDGE:
                                edge_z = edge_preact[edge, ce]
                                edge_deriv = T.alloc_var("float32")
                                if edge_z >= threshold:
                                    edge_th = T.tanh(slope * (edge_z - threshold))
                                    edge_deriv = slope * (1.0 - edge_th * edge_th)
                                else:
                                    edge_sig = 1.0 / (1.0 + T.exp(-edge_z))
                                    edge_deriv = edge_sig * (1.0 + edge_z * (1.0 - edge_sig))
                                p[mi, ki] = grad_edge_output[edge, ce] * edge_residual[ce] * edge_deriv
                            else:
                                p[mi, ki] = 0
                    else:
                        p[mi, ki] = 0
                for ni, ki in T.Parallel(BLOCK_N, BLOCK_K):
                    d = by * BLOCK_N + ni
                    c = ko * BLOCK_K + ki
                    if d < D and c < C:
                        if c < C_NODE:
                            w[ni, ki] = node_weight[d, c]
                        else:
                            w[ni, ki] = edge_weight[d, c - C_NODE]
                    else:
                        w[ni, ki] = 0
                T.sync_threads()
                T.gemm(p, w, acc, transpose_B=True)
                T.sync_threads()
            for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                edge = bx * BLOCK_M + mi
                d = by * BLOCK_N + ni
                if edge < M and d < D:
                    if d < C_NODE:
                        T.atomic_add(grad_node_ebd[owner[edge], d], acc[mi, ni])
                    elif d < 2 * C_NODE:
                        T.atomic_add(grad_node_ebd_ext[n_ext2e_index[edge], d - C_NODE], acc[mi, ni])
                    else:
                        ce = d - 2 * C_NODE
                        direct = T.alloc_var("float32", init=0)
                        if HAS_GRAD_EDGE:
                            direct = grad_edge_output[edge, ce]
                        grad_edge_ebd[edge, ce] = acc[mi, ni] + direct

    return edge_backward_inputs


@tilelang.jit
def fused_edge_block_backward_weight_partials(
    M, N_NODE, N_NODE_EXT, C_NODE, C_EDGE, SPLIT_M=1,
    HAS_GRAD_NODE=True, HAS_GRAD_EDGE=True,
    BLOCK_D=32, BLOCK_C=32, BLOCK_M=32,
):
    D = 2 * C_NODE + C_EDGE
    C = C_NODE + C_EDGE
    TILES = (M + BLOCK_M - 1) // BLOCK_M
    TILES_PER_SPLIT = (TILES + SPLIT_M - 1) // SPLIT_M

    @T.prim_func
    def edge_backward_weight_partials(
        grad_node_output: T.Tensor((N_NODE, C_NODE), "float32"),
        grad_edge_output: T.Tensor((M, C_EDGE), "float32"),
        node_ebd: T.Tensor((N_NODE, C_NODE), "float32"),
        node_ebd_ext: T.Tensor((N_NODE_EXT, C_NODE), "float32"),
        edge_ebd: T.Tensor((M, C_EDGE), "float32"),
        node_preact: T.Tensor((M, C_NODE), "float32"),
        edge_preact: T.Tensor((M, C_EDGE), "float32"),
        sw: T.Tensor((M,), "float32"),
        owner: T.Tensor((M,), "int64"),
        n_ext2e_index: T.Tensor((M,), "int64"),
        node_residual: T.Tensor((C_NODE,), "float32"),
        edge_residual: T.Tensor((C_EDGE,), "float32"),
        scale: T.float32,
        threshold: T.float32,
        slope: T.float32,
        node_partials: T.Tensor((SPLIT_M, D, C_NODE), "float32"),
        edge_partials: T.Tensor((SPLIT_M, D, C_EDGE), "float32"),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(C, BLOCK_C), SPLIT_M, threads=128) as (bx, by, bs):
            x = T.alloc_shared((BLOCK_M, BLOCK_D), "float32")
            p = T.alloc_shared((BLOCK_M, BLOCK_C), "float32")
            acc = T.alloc_fragment((BLOCK_D, BLOCK_C), "float32")
            T.clear(acc)
            for tile in T.Pipelined(TILES_PER_SPLIT, num_stages=2):
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    edge = (bs * TILES_PER_SPLIT + tile) * BLOCK_M + mi
                    d = bx * BLOCK_D + di
                    if edge < M and d < D:
                        if d < C_NODE:
                            x[mi, di] = node_ebd[owner[edge], d]
                        elif d < 2 * C_NODE:
                            x[mi, di] = node_ebd_ext[n_ext2e_index[edge], d - C_NODE]
                        else:
                            x[mi, di] = edge_ebd[edge, d - 2 * C_NODE]
                    else:
                        x[mi, di] = 0
                for mi, ci in T.Parallel(BLOCK_M, BLOCK_C):
                    edge = (bs * TILES_PER_SPLIT + tile) * BLOCK_M + mi
                    c = by * BLOCK_C + ci
                    if edge < M and c < C:
                        if c < C_NODE:
                            if HAS_GRAD_NODE:
                                node_z = node_preact[edge, c]
                                node_deriv = T.alloc_var("float32")
                                if node_z >= threshold:
                                    node_th = T.tanh(slope * (node_z - threshold))
                                    node_deriv = slope * (1.0 - node_th * node_th)
                                else:
                                    node_sig = 1.0 / (1.0 + T.exp(-node_z))
                                    node_deriv = node_sig * (1.0 + node_z * (1.0 - node_sig))
                                p[mi, ci] = grad_node_output[owner[edge], c] * node_residual[c] * scale * sw[edge] * node_deriv
                            else:
                                p[mi, ci] = 0
                        else:
                            ce = c - C_NODE
                            if HAS_GRAD_EDGE:
                                edge_z = edge_preact[edge, ce]
                                edge_deriv = T.alloc_var("float32")
                                if edge_z >= threshold:
                                    edge_th = T.tanh(slope * (edge_z - threshold))
                                    edge_deriv = slope * (1.0 - edge_th * edge_th)
                                else:
                                    edge_sig = 1.0 / (1.0 + T.exp(-edge_z))
                                    edge_deriv = edge_sig * (1.0 + edge_z * (1.0 - edge_sig))
                                p[mi, ci] = grad_edge_output[edge, ce] * edge_residual[ce] * edge_deriv
                            else:
                                p[mi, ci] = 0
                    else:
                        p[mi, ci] = 0
                T.sync_threads()
                T.gemm(x, p, acc, transpose_A=True)
                T.sync_threads()
            for di, ci in T.Parallel(BLOCK_D, BLOCK_C):
                d = bx * BLOCK_D + di
                c = by * BLOCK_C + ci
                if d < D and c < C:
                    if c < C_NODE:
                        node_partials[bs, d, c] = acc[di, ci]
                    else:
                        edge_partials[bs, d, c - C_NODE] = acc[di, ci]

    return edge_backward_weight_partials


@tilelang.jit
def fused_edge_block_backward_weight_reduce(D, C_NODE, C_EDGE, SPLIT_M):
    @T.prim_func
    def edge_backward_weight_reduce(
        node_partials: T.Tensor((SPLIT_M, D, C_NODE), "float32"),
        edge_partials: T.Tensor((SPLIT_M, D, C_EDGE), "float32"),
        grad_node_weight: T.Tensor((D, C_NODE), "float32"),
        grad_edge_weight: T.Tensor((D, C_EDGE), "float32"),
    ):
        with T.Kernel(T.ceildiv(D * (C_NODE + C_EDGE), 256), threads=256) as (bx,):
            idx = bx * 256 + T.get_thread_binding()
            if idx < D * (C_NODE + C_EDGE):
                d = idx // (C_NODE + C_EDGE)
                c = idx % (C_NODE + C_EDGE)
                acc = T.alloc_var("float32", init=0)
                if c < C_NODE:
                    for s in T.serial(SPLIT_M):
                        acc += node_partials[s, d, c]
                    grad_node_weight[d, c] = acc
                else:
                    ce = c - C_NODE
                    for s in T.serial(SPLIT_M):
                        acc += edge_partials[s, d, ce]
                    grad_edge_weight[d, ce] = acc

    return edge_backward_weight_reduce


@tilelang.jit
def fused_edge_block_backward_vectors(
    M, N_NODE, C_NODE, C_EDGE,
    HAS_GRAD_NODE=True, HAS_GRAD_EDGE=True, THREADS=128,
):
    GRID = max(M, C_NODE, C_EDGE)

    @T.prim_func
    def edge_backward_vectors(
        grad_node_output: T.Tensor((N_NODE, C_NODE), "float32"),
        grad_edge_output: T.Tensor((M, C_EDGE), "float32"),
        node_preact: T.Tensor((M, C_NODE), "float32"),
        edge_preact: T.Tensor((M, C_EDGE), "float32"),
        sw: T.Tensor((M,), "float32"),
        owner: T.Tensor((M,), "int64"),
        node_residual: T.Tensor((C_NODE,), "float32"),
        edge_residual: T.Tensor((C_EDGE,), "float32"),
        scale: T.float32,
        threshold: T.float32,
        slope: T.float32,
        const_value: T.float32,
        grad_node_bias: T.Tensor((C_NODE,), "float32"),
        grad_node_residual: T.Tensor((C_NODE,), "float32"),
        grad_edge_bias: T.Tensor((C_EDGE,), "float32"),
        grad_edge_residual: T.Tensor((C_EDGE,), "float32"),
        grad_sw: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(GRID, threads=THREADS) as (block,):
            tx = T.get_thread_binding()
            sh0 = T.alloc_shared((THREADS,), "float32")
            sh1 = T.alloc_shared((THREADS,), "float32")
            if block < C_NODE:
                acc_b = T.alloc_var("float32", init=0)
                acc_r = T.alloc_var("float32", init=0)
                if HAS_GRAD_NODE:
                    for edge in T.serial(tx, M, THREADS):
                        node_z = node_preact[edge, block]
                        node_value = T.alloc_var("float32")
                        node_deriv = T.alloc_var("float32")
                        if node_z >= threshold:
                            node_th = T.tanh(slope * (node_z - threshold))
                            node_value = node_th + const_value
                            node_deriv = slope * (1.0 - node_th * node_th)
                        else:
                            node_sig = 1.0 / (1.0 + T.exp(-node_z))
                            node_value = node_z * node_sig
                            node_deriv = node_sig * (1.0 + node_z * (1.0 - node_sig))
                        node_g = grad_node_output[owner[edge], block]
                        acc_b += node_g * node_residual[block] * scale * sw[edge] * node_deriv
                        acc_r += node_g * scale * sw[edge] * node_value
                sh0[tx] = acc_b
                sh1[tx] = acc_r
                T.sync_threads()
                red0 = T.alloc_shared((1,), "float32")
                red1 = T.alloc_shared((1,), "float32")
                T.reduce_sum(sh0, red0, dim=0)
                T.reduce_sum(sh1, red1, dim=0)
                if tx == 0:
                    grad_node_bias[block] = red0[0]
                    grad_node_residual[block] = red1[0]
            T.sync_threads()
            if block < C_EDGE:
                acc_b = T.alloc_var("float32", init=0)
                acc_r = T.alloc_var("float32", init=0)
                if HAS_GRAD_EDGE:
                    for edge in T.serial(tx, M, THREADS):
                        edge_z = edge_preact[edge, block]
                        edge_value = T.alloc_var("float32")
                        edge_deriv = T.alloc_var("float32")
                        if edge_z >= threshold:
                            edge_th = T.tanh(slope * (edge_z - threshold))
                            edge_value = edge_th + const_value
                            edge_deriv = slope * (1.0 - edge_th * edge_th)
                        else:
                            edge_sig = 1.0 / (1.0 + T.exp(-edge_z))
                            edge_value = edge_z * edge_sig
                            edge_deriv = edge_sig * (1.0 + edge_z * (1.0 - edge_sig))
                        edge_g = grad_edge_output[edge, block]
                        acc_b += edge_g * edge_residual[block] * edge_deriv
                        acc_r += edge_g * edge_value
                sh0[tx] = acc_b
                sh1[tx] = acc_r
                T.sync_threads()
                red0 = T.alloc_shared((1,), "float32")
                red1 = T.alloc_shared((1,), "float32")
                T.reduce_sum(sh0, red0, dim=0)
                T.reduce_sum(sh1, red1, dim=0)
                if tx == 0:
                    grad_edge_bias[block] = red0[0]
                    grad_edge_residual[block] = red1[0]
            T.sync_threads()
            if block < M:
                acc_sw = T.alloc_var("float32", init=0)
                if HAS_GRAD_NODE:
                    for c in T.serial(tx, C_NODE, THREADS):
                        sw_z = node_preact[block, c]
                        sw_value = T.alloc_var("float32")
                        if sw_z >= threshold:
                            sw_value = T.tanh(slope * (sw_z - threshold)) + const_value
                        else:
                            sw_sig = 1.0 / (1.0 + T.exp(-sw_z))
                            sw_value = sw_z * sw_sig
                        acc_sw += grad_node_output[owner[block], c] * node_residual[c] * scale * sw_value
                sh0[tx] = acc_sw
                T.sync_threads()
                red0 = T.alloc_shared((1,), "float32")
                T.reduce_sum(sh0, red0, dim=0)
                if tx == 0:
                    grad_sw[block] = red0[0]

    return edge_backward_vectors


@tilelang.jit
def fused_edge_block_double_prepare(
    M, N_NODE, N_NODE_EXT, C_NODE, C_EDGE,
    HAS_GRAD_NODE=True, HAS_GRAD_EDGE=True,
    HAS_U_NODE=True, HAS_U_NODE_EXT=True, HAS_U_EDGE=True,
    HAS_U_SW=True, HAS_U_NODE_WEIGHT=True, HAS_U_NODE_BIAS=True,
    HAS_U_NODE_RESIDUAL=True, HAS_U_EDGE_WEIGHT=True,
    HAS_U_EDGE_BIAS=True, HAS_U_EDGE_RESIDUAL=True,
    BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
):
    D = 2 * C_NODE + C_EDGE
    C = C_NODE + C_EDGE

    @T.prim_func
    def edge_double_prepare(
        grad_node_output: T.Tensor((N_NODE, C_NODE), "float32"),
        grad_edge_output: T.Tensor((M, C_EDGE), "float32"),
        node_ebd: T.Tensor((N_NODE, C_NODE), "float32"),
        node_ebd_ext: T.Tensor((N_NODE_EXT, C_NODE), "float32"),
        edge_ebd: T.Tensor((M, C_EDGE), "float32"),
        sw: T.Tensor((M,), "float32"),
        owner: T.Tensor((M,), "int64"),
        n_ext2e_index: T.Tensor((M,), "int64"),
        node_weight: T.Tensor((D, C_NODE), "float32"),
        node_residual: T.Tensor((C_NODE,), "float32"),
        edge_weight: T.Tensor((D, C_EDGE), "float32"),
        edge_residual: T.Tensor((C_EDGE,), "float32"),
        node_preact: T.Tensor((M, C_NODE), "float32"),
        edge_preact: T.Tensor((M, C_EDGE), "float32"),
        u_node: T.Tensor((N_NODE, C_NODE), "float32"),
        u_node_ext: T.Tensor((N_NODE_EXT, C_NODE), "float32"),
        u_edge: T.Tensor((M, C_EDGE), "float32"),
        u_sw: T.Tensor((M,), "float32"),
        u_node_weight: T.Tensor((D, C_NODE), "float32"),
        u_node_bias: T.Tensor((C_NODE,), "float32"),
        u_node_residual: T.Tensor((C_NODE,), "float32"),
        u_edge_weight: T.Tensor((D, C_EDGE), "float32"),
        u_edge_bias: T.Tensor((C_EDGE,), "float32"),
        u_edge_residual: T.Tensor((C_EDGE,), "float32"),
        scale: T.float32,
        threshold: T.float32,
        slope: T.float32,
        const_value: T.float32,
        p_node: T.Tensor((M, C_NODE), "float32"),
        p_edge: T.Tensor((M, C_EDGE), "float32"),
        t_node: T.Tensor((M, C_NODE), "float32"),
        t_edge: T.Tensor((M, C_EDGE), "float32"),
        s_node: T.Tensor((M, C_NODE), "float32"),
        s_edge: T.Tensor((M, C_EDGE), "float32"),
        grad_grad_edge_output: T.Tensor((M, C_EDGE), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(C, BLOCK_N), threads=128) as (bx, by):
            lhs = T.alloc_shared((BLOCK_M, BLOCK_K), "float32")
            rhs = T.alloc_shared((BLOCK_K, BLOCK_N), "float32")
            acc = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.clear(acc)
            # uX @ W
            for ko in T.Pipelined(T.ceildiv(D, BLOCK_K), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    edge = bx * BLOCK_M + mi
                    d = ko * BLOCK_K + ki
                    if edge < M and d < D:
                        if d < C_NODE:
                            if HAS_U_NODE:
                                lhs[mi, ki] = u_node[owner[edge], d]
                            else:
                                lhs[mi, ki] = 0
                        elif d < 2 * C_NODE:
                            if HAS_U_NODE_EXT:
                                lhs[mi, ki] = u_node_ext[n_ext2e_index[edge], d - C_NODE]
                            else:
                                lhs[mi, ki] = 0
                        else:
                            if HAS_U_EDGE:
                                lhs[mi, ki] = u_edge[edge, d - 2 * C_NODE]
                            else:
                                lhs[mi, ki] = 0
                    else:
                        lhs[mi, ki] = 0
                for ki, ni in T.Parallel(BLOCK_K, BLOCK_N):
                    d = ko * BLOCK_K + ki
                    c = by * BLOCK_N + ni
                    if d < D and c < C:
                        if c < C_NODE:
                            rhs[ki, ni] = node_weight[d, c]
                        else:
                            rhs[ki, ni] = edge_weight[d, c - C_NODE]
                    else:
                        rhs[ki, ni] = 0
                T.sync_threads()
                T.gemm(lhs, rhs, acc)
                T.sync_threads()
            # X @ uW, with both heads logically concatenated.
            if HAS_U_NODE_WEIGHT or HAS_U_EDGE_WEIGHT:
                for ko in T.Pipelined(T.ceildiv(D, BLOCK_K), num_stages=2):
                    for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                        edge = bx * BLOCK_M + mi
                        d = ko * BLOCK_K + ki
                        if edge < M and d < D:
                            if d < C_NODE:
                                lhs[mi, ki] = node_ebd[owner[edge], d]
                            elif d < 2 * C_NODE:
                                lhs[mi, ki] = node_ebd_ext[n_ext2e_index[edge], d - C_NODE]
                            else:
                                lhs[mi, ki] = edge_ebd[edge, d - 2 * C_NODE]
                        else:
                            lhs[mi, ki] = 0
                    for ki, ni in T.Parallel(BLOCK_K, BLOCK_N):
                        d = ko * BLOCK_K + ki
                        c = by * BLOCK_N + ni
                        if d < D and c < C:
                            if c < C_NODE:
                                if HAS_U_NODE_WEIGHT:
                                    rhs[ki, ni] = u_node_weight[d, c]
                                else:
                                    rhs[ki, ni] = 0
                            else:
                                if HAS_U_EDGE_WEIGHT:
                                    rhs[ki, ni] = u_edge_weight[d, c - C_NODE]
                                else:
                                    rhs[ki, ni] = 0
                        else:
                            rhs[ki, ni] = 0
                    T.sync_threads()
                    T.gemm(lhs, rhs, acc)
                    T.sync_threads()
            for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                edge = bx * BLOCK_M + mi
                c = by * BLOCK_N + ni
                if edge < M and c < C:
                    t = T.alloc_var("float32")
                    z = T.alloc_var("float32")
                    p = T.alloc_var("float32")
                    s = T.alloc_var("float32")
                    gg = T.alloc_var("float32")
                    if c < C_NODE:
                        t = acc[mi, ni]
                        if HAS_U_NODE_BIAS:
                            t += u_node_bias[c]
                        z = node_preact[edge, c]
                        value = T.alloc_var("float32")
                        deriv = T.alloc_var("float32")
                        second = T.alloc_var("float32")
                        if z >= threshold:
                            th = T.tanh(slope * (z - threshold))
                            value = th + const_value
                            deriv = slope * (1.0 - th * th)
                            second = -2.0 * slope * th * deriv
                        else:
                            sig = 1.0 / (1.0 + T.exp(-z))
                            value = z * sig
                            deriv = sig * (1.0 + z * (1.0 - sig))
                            second = sig * (1.0 - sig) * (2.0 + z * (1.0 - 2.0 * sig))
                        g = T.alloc_var("float32", init=0)
                        if HAS_GRAD_NODE:
                            g = grad_node_output[owner[edge], c]
                        p = g * node_residual[c] * scale * sw[edge] * deriv
                        s = g * scale * sw[edge] * node_residual[c] * second * t
                        if HAS_U_NODE_RESIDUAL:
                            s += g * scale * sw[edge] * u_node_residual[c] * deriv
                        if HAS_U_SW:
                            s += g * scale * u_sw[edge] * node_residual[c] * deriv
                        p_node[edge, c] = p
                        t_node[edge, c] = t
                        s_node[edge, c] = s
                    else:
                        ce = c - C_NODE
                        t = acc[mi, ni]
                        if HAS_U_EDGE_BIAS:
                            t += u_edge_bias[ce]
                        z = edge_preact[edge, ce]
                        value = T.alloc_var("float32")
                        deriv = T.alloc_var("float32")
                        second = T.alloc_var("float32")
                        if z >= threshold:
                            th = T.tanh(slope * (z - threshold))
                            value = th + const_value
                            deriv = slope * (1.0 - th * th)
                            second = -2.0 * slope * th * deriv
                        else:
                            sig = 1.0 / (1.0 + T.exp(-z))
                            value = z * sig
                            deriv = sig * (1.0 + z * (1.0 - sig))
                            second = sig * (1.0 - sig) * (2.0 + z * (1.0 - 2.0 * sig))
                        g = T.alloc_var("float32", init=0)
                        if HAS_GRAD_EDGE:
                            g = grad_edge_output[edge, ce]
                        p = g * edge_residual[ce] * deriv
                        s = g * edge_residual[ce] * second * t
                        if HAS_U_EDGE_RESIDUAL:
                            s += g * u_edge_residual[ce] * deriv
                        p_edge[edge, ce] = p
                        t_edge[edge, ce] = t
                        s_edge[edge, ce] = s
                        gg = edge_residual[ce] * deriv * t
                        if HAS_U_EDGE:
                            gg += u_edge[edge, ce]
                        if HAS_U_EDGE_RESIDUAL:
                            gg += u_edge_residual[ce] * value
                        grad_grad_edge_output[edge, ce] = gg

    return edge_double_prepare


@tilelang.jit
def fused_edge_block_double_node_output_uniform(
    M, NO, C_NODE, HAS_U_NODE_PARTIAL=True,
    HAS_U_SW=True, HAS_U_NODE_RESIDUAL=True, THREADS=128,
):
    assert M % NO == 0
    EDGES_PER_OWNER = M // NO

    @T.prim_func
    def edge_double_node_output_uniform(
        node_preact: T.Tensor((M, C_NODE), "float32"),
        t_node: T.Tensor((M, C_NODE), "float32"),
        sw: T.Tensor((M,), "float32"),
        node_residual: T.Tensor((C_NODE,), "float32"),
        u_node_partial: T.Tensor((NO, C_NODE), "float32"),
        u_sw: T.Tensor((M,), "float32"),
        u_node_residual: T.Tensor((C_NODE,), "float32"),
        scale: T.float32,
        threshold: T.float32,
        slope: T.float32,
        const_value: T.float32,
        grad_grad_node_output: T.Tensor((NO, C_NODE), "float32"),
    ):
        with T.Kernel(NO, T.ceildiv(C_NODE, THREADS), threads=THREADS) as (owner_id, tile):
            c = tile * THREADS + T.get_thread_binding()
            if c < C_NODE:
                acc = T.alloc_var("float32", init=0)
                if HAS_U_NODE_PARTIAL:
                    acc = u_node_partial[owner_id, c]
                for r in T.serial(EDGES_PER_OWNER):
                    edge = owner_id * EDGES_PER_OWNER + r
                    z = node_preact[edge, c]
                    value = T.alloc_var("float32")
                    deriv = T.alloc_var("float32")
                    if z >= threshold:
                        th = T.tanh(slope * (z - threshold))
                        value = th + const_value
                        deriv = slope * (1.0 - th * th)
                    else:
                        sig = 1.0 / (1.0 + T.exp(-z))
                        value = z * sig
                        deriv = sig * (1.0 + z * (1.0 - sig))
                    acc += scale * sw[edge] * node_residual[c] * deriv * t_node[edge, c]
                    if HAS_U_NODE_RESIDUAL:
                        acc += scale * sw[edge] * u_node_residual[c] * value
                    if HAS_U_SW:
                        acc += scale * u_sw[edge] * node_residual[c] * value
                grad_grad_node_output[owner_id, c] = acc

    return edge_double_node_output_uniform


@tilelang.jit
def fused_edge_block_double_node_output_segmented(
    M, NO, C_NODE, HAS_U_NODE_PARTIAL=True,
    HAS_U_SW=True, HAS_U_NODE_RESIDUAL=True, THREADS=128,
):
    @T.prim_func
    def edge_double_node_output_segmented(
        node_preact: T.Tensor((M, C_NODE), "float32"),
        t_node: T.Tensor((M, C_NODE), "float32"),
        sw: T.Tensor((M,), "float32"),
        node_residual: T.Tensor((C_NODE,), "float32"),
        u_node_partial: T.Tensor((NO, C_NODE), "float32"),
        u_sw: T.Tensor((M,), "float32"),
        u_node_residual: T.Tensor((C_NODE,), "float32"),
        offsets: T.Tensor((NO + 1,), "int64"),
        order: T.Tensor((M,), "int64"),
        scale: T.float32,
        threshold: T.float32,
        slope: T.float32,
        const_value: T.float32,
        grad_grad_node_output: T.Tensor((NO, C_NODE), "float32"),
    ):
        with T.Kernel(NO, T.ceildiv(C_NODE, THREADS), threads=THREADS) as (owner_id, tile):
            c = tile * THREADS + T.get_thread_binding()
            if c < C_NODE:
                acc = T.alloc_var("float32", init=0)
                if HAS_U_NODE_PARTIAL:
                    acc = u_node_partial[owner_id, c]
                for r in T.serial(offsets[owner_id], offsets[owner_id + 1]):
                    edge = order[r]
                    z = node_preact[edge, c]
                    value = T.alloc_var("float32")
                    deriv = T.alloc_var("float32")
                    if z >= threshold:
                        th = T.tanh(slope * (z - threshold))
                        value = th + const_value
                        deriv = slope * (1.0 - th * th)
                    else:
                        sig = 1.0 / (1.0 + T.exp(-z))
                        value = z * sig
                        deriv = sig * (1.0 + z * (1.0 - sig))
                    acc += scale * sw[edge] * node_residual[c] * deriv * t_node[edge, c]
                    if HAS_U_NODE_RESIDUAL:
                        acc += scale * sw[edge] * u_node_residual[c] * value
                    if HAS_U_SW:
                        acc += scale * u_sw[edge] * node_residual[c] * value
                grad_grad_node_output[owner_id, c] = acc

    return edge_double_node_output_segmented


@tilelang.jit
def fused_edge_block_double_inputs(
    M, N_NODE, N_NODE_EXT, C_NODE, C_EDGE,
    HAS_U_NODE_WEIGHT=True, HAS_U_EDGE_WEIGHT=True,
    BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
):
    D = 2 * C_NODE + C_EDGE
    C = C_NODE + C_EDGE

    @T.prim_func
    def edge_double_inputs(
        p_node: T.Tensor((M, C_NODE), "float32"),
        p_edge: T.Tensor((M, C_EDGE), "float32"),
        s_node: T.Tensor((M, C_NODE), "float32"),
        s_edge: T.Tensor((M, C_EDGE), "float32"),
        node_weight: T.Tensor((D, C_NODE), "float32"),
        edge_weight: T.Tensor((D, C_EDGE), "float32"),
        u_node_weight: T.Tensor((D, C_NODE), "float32"),
        u_edge_weight: T.Tensor((D, C_EDGE), "float32"),
        owner: T.Tensor((M,), "int64"),
        n_ext2e_index: T.Tensor((M,), "int64"),
        grad_node_ebd: T.Tensor((N_NODE, C_NODE), "float32"),
        grad_node_ebd_ext: T.Tensor((N_NODE_EXT, C_NODE), "float32"),
        grad_edge_ebd: T.Tensor((M, C_EDGE), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(D, BLOCK_N), threads=128) as (bx, by):
            lhs = T.alloc_shared((BLOCK_M, BLOCK_K), "float32")
            rhs = T.alloc_shared((BLOCK_N, BLOCK_K), "float32")
            acc = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(2 * C, BLOCK_K), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    edge = bx * BLOCK_M + mi
                    k = ko * BLOCK_K + ki
                    if edge < M and k < 2 * C:
                        c = k % C
                        if c < C_NODE:
                            if k >= C:
                                lhs[mi, ki] = s_node[edge, c]
                            else:
                                lhs[mi, ki] = p_node[edge, c]
                        else:
                            ce = c - C_NODE
                            if k >= C:
                                lhs[mi, ki] = s_edge[edge, ce]
                            else:
                                lhs[mi, ki] = p_edge[edge, ce]
                    else:
                        lhs[mi, ki] = 0
                for ni, ki in T.Parallel(BLOCK_N, BLOCK_K):
                    d = by * BLOCK_N + ni
                    k = ko * BLOCK_K + ki
                    if d < D and k < 2 * C:
                        c = k % C
                        if c < C_NODE:
                            if k >= C:
                                rhs[ni, ki] = node_weight[d, c]
                            elif HAS_U_NODE_WEIGHT:
                                rhs[ni, ki] = u_node_weight[d, c]
                            else:
                                rhs[ni, ki] = 0
                        else:
                            ce = c - C_NODE
                            if k >= C:
                                rhs[ni, ki] = edge_weight[d, ce]
                            elif HAS_U_EDGE_WEIGHT:
                                rhs[ni, ki] = u_edge_weight[d, ce]
                            else:
                                rhs[ni, ki] = 0
                    else:
                        rhs[ni, ki] = 0
                T.sync_threads()
                T.gemm(lhs, rhs, acc, transpose_B=True)
                T.sync_threads()
            for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                edge = bx * BLOCK_M + mi
                d = by * BLOCK_N + ni
                if edge < M and d < D:
                    if d < C_NODE:
                        T.atomic_add(grad_node_ebd[owner[edge], d], acc[mi, ni])
                    elif d < 2 * C_NODE:
                        T.atomic_add(grad_node_ebd_ext[n_ext2e_index[edge], d - C_NODE], acc[mi, ni])
                    else:
                        grad_edge_ebd[edge, d - 2 * C_NODE] = acc[mi, ni]

    return edge_double_inputs


@tilelang.jit
def fused_edge_block_double_weight_partials(
    M, N_NODE, N_NODE_EXT, C_NODE, C_EDGE, SPLIT_M=1,
    HAS_U_NODE=True, HAS_U_NODE_EXT=True, HAS_U_EDGE=True,
    BLOCK_D=32, BLOCK_C=32, BLOCK_M=32,
):
    D = 2 * C_NODE + C_EDGE
    C = C_NODE + C_EDGE
    TILES = (M + BLOCK_M - 1) // BLOCK_M
    TILES_PER_SPLIT = (TILES + SPLIT_M - 1) // SPLIT_M

    @T.prim_func
    def edge_double_weight_partials(
        node_ebd: T.Tensor((N_NODE, C_NODE), "float32"),
        node_ebd_ext: T.Tensor((N_NODE_EXT, C_NODE), "float32"),
        edge_ebd: T.Tensor((M, C_EDGE), "float32"),
        u_node: T.Tensor((N_NODE, C_NODE), "float32"),
        u_node_ext: T.Tensor((N_NODE_EXT, C_NODE), "float32"),
        u_edge: T.Tensor((M, C_EDGE), "float32"),
        owner: T.Tensor((M,), "int64"),
        n_ext2e_index: T.Tensor((M,), "int64"),
        p_node: T.Tensor((M, C_NODE), "float32"),
        p_edge: T.Tensor((M, C_EDGE), "float32"),
        s_node: T.Tensor((M, C_NODE), "float32"),
        s_edge: T.Tensor((M, C_EDGE), "float32"),
        node_partials: T.Tensor((SPLIT_M, D, C_NODE), "float32"),
        edge_partials: T.Tensor((SPLIT_M, D, C_EDGE), "float32"),
    ):
        with T.Kernel(T.ceildiv(D, BLOCK_D), T.ceildiv(C, BLOCK_C), SPLIT_M, threads=128) as (bx, by, bs):
            lhs = T.alloc_shared((BLOCK_M, BLOCK_D), "float32")
            rhs = T.alloc_shared((BLOCK_M, BLOCK_C), "float32")
            acc = T.alloc_fragment((BLOCK_D, BLOCK_C), "float32")
            T.clear(acc)
            # [uX; X].T @ [P; S] keeps both heads in one logical GEMM.
            for ko in T.Pipelined(2 * TILES_PER_SPLIT, num_stages=2):
                local_tile = ko % TILES_PER_SPLIT
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    edge = (bs * TILES_PER_SPLIT + local_tile) * BLOCK_M + mi
                    d = bx * BLOCK_D + di
                    if edge < M and d < D:
                        if ko >= TILES_PER_SPLIT:
                            if d < C_NODE:
                                lhs[mi, di] = node_ebd[owner[edge], d]
                            elif d < 2 * C_NODE:
                                lhs[mi, di] = node_ebd_ext[n_ext2e_index[edge], d - C_NODE]
                            else:
                                lhs[mi, di] = edge_ebd[edge, d - 2 * C_NODE]
                        else:
                            if d < C_NODE:
                                if HAS_U_NODE:
                                    lhs[mi, di] = u_node[owner[edge], d]
                                else:
                                    lhs[mi, di] = 0
                            elif d < 2 * C_NODE:
                                if HAS_U_NODE_EXT:
                                    lhs[mi, di] = u_node_ext[n_ext2e_index[edge], d - C_NODE]
                                else:
                                    lhs[mi, di] = 0
                            else:
                                if HAS_U_EDGE:
                                    lhs[mi, di] = u_edge[edge, d - 2 * C_NODE]
                                else:
                                    lhs[mi, di] = 0
                    else:
                        lhs[mi, di] = 0
                for mi, ci in T.Parallel(BLOCK_M, BLOCK_C):
                    edge = (bs * TILES_PER_SPLIT + local_tile) * BLOCK_M + mi
                    c = by * BLOCK_C + ci
                    if edge < M and c < C:
                        if c < C_NODE:
                            if ko >= TILES_PER_SPLIT:
                                rhs[mi, ci] = s_node[edge, c]
                            else:
                                rhs[mi, ci] = p_node[edge, c]
                        else:
                            if ko >= TILES_PER_SPLIT:
                                rhs[mi, ci] = s_edge[edge, c - C_NODE]
                            else:
                                rhs[mi, ci] = p_edge[edge, c - C_NODE]
                    else:
                        rhs[mi, ci] = 0
                T.sync_threads()
                T.gemm(lhs, rhs, acc, transpose_A=True)
                T.sync_threads()
            for di, ci in T.Parallel(BLOCK_D, BLOCK_C):
                d = bx * BLOCK_D + di
                c = by * BLOCK_C + ci
                if d < D and c < C:
                    if c < C_NODE:
                        node_partials[bs, d, c] = acc[di, ci]
                    else:
                        edge_partials[bs, d, c - C_NODE] = acc[di, ci]

    return edge_double_weight_partials


@tilelang.jit
def fused_edge_block_double_weight_reduce(D, C_NODE, C_EDGE, SPLIT_M):
    """Reduce split-M second-order weight partials when SPLIT_M > 1."""

    @T.prim_func
    def edge_double_weight_reduce(
        node_partials: T.Tensor((SPLIT_M, D, C_NODE), "float32"),
        edge_partials: T.Tensor((SPLIT_M, D, C_EDGE), "float32"),
        grad_node_weight: T.Tensor((D, C_NODE), "float32"),
        grad_edge_weight: T.Tensor((D, C_EDGE), "float32"),
    ):
        with T.Kernel(T.ceildiv(D * (C_NODE + C_EDGE), 256), threads=256) as (bx,):
            idx = bx * 256 + T.get_thread_binding()
            if idx < D * (C_NODE + C_EDGE):
                d = idx // (C_NODE + C_EDGE)
                c = idx % (C_NODE + C_EDGE)
                acc = T.alloc_var("float32", init=0)
                if c < C_NODE:
                    for s in T.serial(SPLIT_M):
                        acc += node_partials[s, d, c]
                    grad_node_weight[d, c] = acc
                else:
                    ce = c - C_NODE
                    for s in T.serial(SPLIT_M):
                        acc += edge_partials[s, d, ce]
                    grad_edge_weight[d, ce] = acc

    return edge_double_weight_reduce


@tilelang.jit
def fused_edge_block_double_vectors(
    M, N_NODE, C_NODE, C_EDGE,
    HAS_GRAD_NODE=True, HAS_GRAD_EDGE=True,
    HAS_U_SW=True, HAS_U_NODE_RESIDUAL=True, THREADS=128,
):
    GRID = max(M, C_NODE, C_EDGE)

    @T.prim_func
    def edge_double_vectors(
        grad_node_output: T.Tensor((N_NODE, C_NODE), "float32"),
        grad_edge_output: T.Tensor((M, C_EDGE), "float32"),
        node_preact: T.Tensor((M, C_NODE), "float32"),
        edge_preact: T.Tensor((M, C_EDGE), "float32"),
        t_node: T.Tensor((M, C_NODE), "float32"),
        t_edge: T.Tensor((M, C_EDGE), "float32"),
        s_node: T.Tensor((M, C_NODE), "float32"),
        s_edge: T.Tensor((M, C_EDGE), "float32"),
        sw: T.Tensor((M,), "float32"),
        owner: T.Tensor((M,), "int64"),
        node_residual: T.Tensor((C_NODE,), "float32"),
        u_sw: T.Tensor((M,), "float32"),
        u_node_residual: T.Tensor((C_NODE,), "float32"),
        scale: T.float32,
        threshold: T.float32,
        slope: T.float32,
        const_value: T.float32,
        grad_node_bias: T.Tensor((C_NODE,), "float32"),
        grad_node_residual: T.Tensor((C_NODE,), "float32"),
        grad_edge_bias: T.Tensor((C_EDGE,), "float32"),
        grad_edge_residual: T.Tensor((C_EDGE,), "float32"),
        grad_sw: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(GRID, threads=THREADS) as (block,):
            tx = T.get_thread_binding()
            sh0 = T.alloc_shared((THREADS,), "float32")
            sh1 = T.alloc_shared((THREADS,), "float32")
            if block < C_NODE:
                acc_b = T.alloc_var("float32", init=0)
                acc_r = T.alloc_var("float32", init=0)
                if HAS_GRAD_NODE:
                    for edge in T.serial(tx, M, THREADS):
                        z = node_preact[edge, block]
                        value = T.alloc_var("float32")
                        deriv = T.alloc_var("float32")
                        if z >= threshold:
                            th = T.tanh(slope * (z - threshold))
                            value = th + const_value
                            deriv = slope * (1.0 - th * th)
                        else:
                            sig = 1.0 / (1.0 + T.exp(-z))
                            value = z * sig
                            deriv = sig * (1.0 + z * (1.0 - sig))
                        acc_b += s_node[edge, block]
                        acc_r += grad_node_output[owner[edge], block] * scale * sw[edge] * deriv * t_node[edge, block]
                        if HAS_U_SW:
                            acc_r += grad_node_output[owner[edge], block] * scale * u_sw[edge] * value
                sh0[tx] = acc_b
                sh1[tx] = acc_r
                T.sync_threads()
                red0 = T.alloc_shared((1,), "float32")
                red1 = T.alloc_shared((1,), "float32")
                T.reduce_sum(sh0, red0, dim=0)
                T.reduce_sum(sh1, red1, dim=0)
                if tx == 0:
                    grad_node_bias[block] = red0[0]
                    grad_node_residual[block] = red1[0]
            T.sync_threads()
            if block < C_EDGE:
                acc_b = T.alloc_var("float32", init=0)
                acc_r = T.alloc_var("float32", init=0)
                if HAS_GRAD_EDGE:
                    for edge in T.serial(tx, M, THREADS):
                        z = edge_preact[edge, block]
                        deriv = T.alloc_var("float32")
                        if z >= threshold:
                            th = T.tanh(slope * (z - threshold))
                            deriv = slope * (1.0 - th * th)
                        else:
                            sig = 1.0 / (1.0 + T.exp(-z))
                            deriv = sig * (1.0 + z * (1.0 - sig))
                        acc_b += s_edge[edge, block]
                        acc_r += grad_edge_output[edge, block] * deriv * t_edge[edge, block]
                sh0[tx] = acc_b
                sh1[tx] = acc_r
                T.sync_threads()
                red0 = T.alloc_shared((1,), "float32")
                red1 = T.alloc_shared((1,), "float32")
                T.reduce_sum(sh0, red0, dim=0)
                T.reduce_sum(sh1, red1, dim=0)
                if tx == 0:
                    grad_edge_bias[block] = red0[0]
                    grad_edge_residual[block] = red1[0]
            T.sync_threads()
            if block < M:
                acc_sw = T.alloc_var("float32", init=0)
                if HAS_GRAD_NODE:
                    for c in T.serial(tx, C_NODE, THREADS):
                        z = node_preact[block, c]
                        value = T.alloc_var("float32")
                        deriv = T.alloc_var("float32")
                        if z >= threshold:
                            th = T.tanh(slope * (z - threshold))
                            value = th + const_value
                            deriv = slope * (1.0 - th * th)
                        else:
                            sig = 1.0 / (1.0 + T.exp(-z))
                            value = z * sig
                            deriv = sig * (1.0 + z * (1.0 - sig))
                        acc_sw += grad_node_output[owner[block], c] * scale * node_residual[c] * deriv * t_node[block, c]
                        if HAS_U_NODE_RESIDUAL:
                            acc_sw += grad_node_output[owner[block], c] * scale * u_node_residual[c] * value
                sh0[tx] = acc_sw
                T.sync_threads()
                red0 = T.alloc_shared((1,), "float32")
                T.reduce_sum(sh0, red0, dim=0)
                if tx == 0:
                    grad_sw[block] = red0[0]

    return edge_double_vectors


class FusedEdgeBlockFunction(torch.autograd.Function):
    """Projection, activation, owner reduction, and both residual updates."""

    @staticmethod
    def forward(
        ctx,
        node_partial,
        node_ebd,
        node_ebd_ext,
        edge_ebd,
        sw,
        owner,
        n_ext2e_index,
        node_weight,
        node_bias,
        node_residual,
        edge_weight,
        edge_bias,
        edge_residual,
        num_owner,
        scale_factor,
        threshold,
        slope,
        const_value,
        owner_metadata,
    ):
        node_partial = node_partial.contiguous()
        node_ebd = node_ebd.contiguous()
        node_ebd_ext = node_ebd_ext.contiguous()
        edge_ebd = edge_ebd.contiguous()
        sw = sw.contiguous()
        owner = owner.long().contiguous()
        n_ext2e_index = n_ext2e_index.long().contiguous()
        node_weight = node_weight.contiguous()
        node_bias = node_bias.contiguous()
        node_residual = node_residual.contiguous()
        edge_weight = edge_weight.contiguous()
        edge_bias = edge_bias.contiguous()
        edge_residual = edge_residual.contiguous()
        M, c_edge = edge_ebd.shape
        n_node, c_node = node_ebd.shape
        n_node_ext = node_ebd_ext.shape[0]
        if node_weight.shape != (2 * c_node + c_edge, c_node):
            raise ValueError("node Edge projection has an unexpected shape")
        if edge_weight.shape != (2 * c_node + c_edge, c_edge):
            raise ValueError("edge Edge projection must preserve edge width")

        node_preact = torch.empty(
            (M, c_node), device=edge_ebd.device, dtype=edge_ebd.dtype,
        )
        edge_preact = torch.empty_like(edge_ebd)
        edge_partial = torch.empty_like(edge_ebd)
        if M:
            projection = fused_edge_block_projection_forward(
                M=M, N_NODE=n_node, N_NODE_EXT=n_node_ext,
                C_NODE=c_node, C_EDGE=c_edge,
            )
            projection(
                node_ebd, node_ebd_ext, edge_ebd, owner, n_ext2e_index,
                node_weight, node_bias, edge_weight, edge_bias, edge_residual,
                float(threshold), float(slope), float(const_value),
                node_preact, edge_preact, edge_partial,
            )
        else:
            edge_partial.copy_(edge_ebd)

        node_output = torch.empty_like(node_partial)
        uniform, offsets, order = owner_metadata
        if M == 0:
            node_output.copy_(node_partial)
        elif uniform:
            reduce_forward = fused_edge_block_node_reduce_forward_uniform(
                M=M, NO=num_owner, C_NODE=c_node,
            )
            reduce_forward(
                node_partial, node_preact, sw, node_residual,
                float(scale_factor), float(threshold), float(slope),
                float(const_value), node_output,
            )
        else:
            reduce_forward = fused_edge_block_node_reduce_forward_segmented(
                M=M, NO=num_owner, C_NODE=c_node,
            )
            reduce_forward(
                node_partial, node_preact, sw, node_residual, offsets, order,
                float(scale_factor), float(threshold), float(slope),
                float(const_value), node_output,
            )

        ctx.set_materialize_grads(False)
        ctx.owner_metadata = owner_metadata
        ctx.num_owner = num_owner
        ctx.scale_factor = scale_factor
        ctx.threshold = threshold
        ctx.slope = slope
        ctx.const_value = const_value
        ctx.save_for_backward(
            node_partial, node_ebd, node_ebd_ext, edge_ebd, sw,
            owner, n_ext2e_index, node_weight, node_bias, node_residual,
            edge_weight, edge_bias, edge_residual, node_preact, edge_preact,
        )
        return node_output, edge_partial

    @staticmethod
    def backward(ctx, grad_node_output, grad_edge_output):
        (
            node_partial, node_ebd, node_ebd_ext, edge_ebd, sw,
            owner, n_ext2e_index, node_weight, node_bias, node_residual,
            edge_weight, edge_bias, edge_residual, node_preact, edge_preact,
        ) = ctx.saved_tensors
        has_grad_node = grad_node_output is not None
        has_grad_edge = grad_edge_output is not None
        if not has_grad_node and not has_grad_edge:
            return (None,) * 19
        grad_node_arg = (
            grad_node_output.contiguous()
            if has_grad_node else node_partial.detach()
        )
        grad_edge_arg = (
            grad_edge_output.contiguous()
            if has_grad_edge else edge_ebd.detach()
        )
        grads = FusedEdgeBlockFunctionBackward.apply(
            grad_node_arg, grad_edge_arg,
            node_partial, node_ebd, node_ebd_ext, edge_ebd, sw,
            owner, n_ext2e_index, node_weight, node_bias, node_residual,
            edge_weight, edge_bias, edge_residual, node_preact, edge_preact,
            ctx.num_owner, ctx.scale_factor, ctx.threshold, ctx.slope,
            ctx.const_value, ctx.owner_metadata, has_grad_node, has_grad_edge,
        )
        (
            grad_node_partial, grad_node, grad_node_ext, grad_edge, grad_sw,
            grad_node_weight, grad_node_bias, grad_node_residual,
            grad_edge_weight, grad_edge_bias, grad_edge_residual,
        ) = grads
        return (
            grad_node_partial if has_grad_node else None,
            grad_node, grad_node_ext, grad_edge, grad_sw,
            None, None,
            grad_node_weight, grad_node_bias, grad_node_residual,
            grad_edge_weight, grad_edge_bias, grad_edge_residual,
            None, None, None, None, None, None,
        )


class FusedEdgeBlockFunctionBackward(torch.autograd.Function):
    """First derivative of the wider Edge block, with fused double backward."""

    @staticmethod
    def forward(
        ctx,
        grad_node_output,
        grad_edge_output,
        node_partial,
        node_ebd,
        node_ebd_ext,
        edge_ebd,
        sw,
        owner,
        n_ext2e_index,
        node_weight,
        node_bias,
        node_residual,
        edge_weight,
        edge_bias,
        edge_residual,
        node_preact,
        edge_preact,
        num_owner,
        scale_factor,
        threshold,
        slope,
        const_value,
        owner_metadata,
        has_grad_node,
        has_grad_edge,
    ):
        M, c_edge = edge_ebd.shape
        n_node, c_node = node_ebd.shape
        n_node_ext = node_ebd_ext.shape[0]
        d = 2 * c_node + c_edge
        grad_node = torch.zeros_like(node_ebd)
        grad_node_ext = torch.zeros_like(node_ebd_ext)
        grad_edge = torch.empty_like(edge_ebd)
        if M:
            inputs_kernel = fused_edge_block_backward_inputs(
                M=M, N_NODE=n_node, N_NODE_EXT=n_node_ext,
                C_NODE=c_node, C_EDGE=c_edge,
                HAS_GRAD_NODE=has_grad_node, HAS_GRAD_EDGE=has_grad_edge,
            )
            inputs_kernel(
                grad_node_output, grad_edge_output, node_preact, edge_preact,
                sw, owner, n_ext2e_index, node_weight, node_residual,
                edge_weight, edge_residual, float(scale_factor),
                float(threshold), float(slope),
                grad_node, grad_node_ext, grad_edge,
            )
        else:
            grad_edge.zero_()

        # Keep the split interface, but do not pay workspace/reduction overhead
        # unless a future tuned configuration explicitly selects SPLIT_M > 1.
        split_m = 1
        grad_node_weight = torch.empty_like(node_weight)
        grad_edge_weight = torch.empty_like(edge_weight)
        if M:
            weight_kernel = fused_edge_block_backward_weight_partials(
                M=M, N_NODE=n_node, N_NODE_EXT=n_node_ext,
                C_NODE=c_node, C_EDGE=c_edge, SPLIT_M=split_m,
                HAS_GRAD_NODE=has_grad_node, HAS_GRAD_EDGE=has_grad_edge,
            )
            if split_m == 1:
                node_partials = grad_node_weight.unsqueeze(0)
                edge_partials = grad_edge_weight.unsqueeze(0)
            else:
                node_partials = torch.empty(
                    (split_m, d, c_node), device=edge_ebd.device,
                    dtype=edge_ebd.dtype,
                )
                edge_partials = torch.empty(
                    (split_m, d, c_edge), device=edge_ebd.device,
                    dtype=edge_ebd.dtype,
                )
            weight_kernel(
                grad_node_output, grad_edge_output,
                node_ebd, node_ebd_ext, edge_ebd,
                node_preact, edge_preact, sw, owner, n_ext2e_index,
                node_residual, edge_residual, float(scale_factor),
                float(threshold), float(slope), node_partials, edge_partials,
            )
            if split_m > 1:
                reduce_kernel = fused_edge_block_backward_weight_reduce(
                    D=d, C_NODE=c_node, C_EDGE=c_edge, SPLIT_M=split_m,
                )
                reduce_kernel(
                    node_partials, edge_partials,
                    grad_node_weight, grad_edge_weight,
                )
        else:
            grad_node_weight.zero_()
            grad_edge_weight.zero_()

        grad_node_bias = torch.empty_like(node_bias)
        grad_node_residual = torch.empty_like(node_residual)
        grad_edge_bias = torch.empty_like(edge_bias)
        grad_edge_residual = torch.empty_like(edge_residual)
        grad_sw = torch.empty_like(sw)
        if M:
            vector_kernel = fused_edge_block_backward_vectors(
                M=M, N_NODE=n_node, C_NODE=c_node, C_EDGE=c_edge,
                HAS_GRAD_NODE=has_grad_node, HAS_GRAD_EDGE=has_grad_edge,
            )
            vector_kernel(
                grad_node_output, grad_edge_output, node_preact, edge_preact,
                sw, owner, node_residual, edge_residual, float(scale_factor),
                float(threshold), float(slope), float(const_value),
                grad_node_bias, grad_node_residual,
                grad_edge_bias, grad_edge_residual, grad_sw,
            )
        else:
            grad_node_bias.zero_()
            grad_node_residual.zero_()
            grad_edge_bias.zero_()
            grad_edge_residual.zero_()
            grad_sw.zero_()

        ctx.set_materialize_grads(False)
        ctx.owner_metadata = owner_metadata
        ctx.num_owner = num_owner
        ctx.scale_factor = scale_factor
        ctx.threshold = threshold
        ctx.slope = slope
        ctx.const_value = const_value
        ctx.has_grad_node = has_grad_node
        ctx.has_grad_edge = has_grad_edge
        ctx.save_for_backward(
            grad_node_output, grad_edge_output,
            node_partial, node_ebd, node_ebd_ext, edge_ebd, sw,
            owner, n_ext2e_index, node_weight, node_bias, node_residual,
            edge_weight, edge_bias, edge_residual, node_preact, edge_preact,
        )
        grad_node_partial = grad_node_output
        return (
            grad_node_partial, grad_node, grad_node_ext, grad_edge, grad_sw,
            grad_node_weight, grad_node_bias, grad_node_residual,
            grad_edge_weight, grad_edge_bias, grad_edge_residual,
        )

    @staticmethod
    def backward(
        ctx,
        u_node_partial,
        u_node,
        u_node_ext,
        u_edge,
        u_sw,
        u_node_weight,
        u_node_bias,
        u_node_residual,
        u_edge_weight,
        u_edge_bias,
        u_edge_residual,
    ):
        (
            grad_node_output, grad_edge_output,
            node_partial, node_ebd, node_ebd_ext, edge_ebd, sw,
            owner, n_ext2e_index, node_weight, node_bias, node_residual,
            edge_weight, edge_bias, edge_residual, node_preact, edge_preact,
        ) = ctx.saved_tensors
        M, c_edge = edge_ebd.shape
        n_node, c_node = node_ebd.shape
        n_node_ext = node_ebd_ext.shape[0]
        d = 2 * c_node + c_edge

        has_u_node_partial = u_node_partial is not None
        has_u_node = u_node is not None
        has_u_node_ext = u_node_ext is not None
        has_u_edge = u_edge is not None
        has_u_sw = u_sw is not None
        has_u_node_weight = u_node_weight is not None
        has_u_node_bias = u_node_bias is not None
        has_u_node_residual = u_node_residual is not None
        has_u_edge_weight = u_edge_weight is not None
        has_u_edge_bias = u_edge_bias is not None
        has_u_edge_residual = u_edge_residual is not None

        u_node_partial_arg = (
            u_node_partial.contiguous() if has_u_node_partial else node_partial
        )
        u_node_arg = u_node.contiguous() if has_u_node else node_ebd
        u_node_ext_arg = (
            u_node_ext.contiguous() if has_u_node_ext else node_ebd_ext
        )
        u_edge_arg = u_edge.contiguous() if has_u_edge else edge_ebd
        u_sw_arg = u_sw.contiguous() if has_u_sw else sw
        u_node_weight_arg = (
            u_node_weight.contiguous() if has_u_node_weight else node_weight
        )
        u_node_bias_arg = (
            u_node_bias.contiguous() if has_u_node_bias else node_bias
        )
        u_node_residual_arg = (
            u_node_residual.contiguous()
            if has_u_node_residual else node_residual
        )
        u_edge_weight_arg = (
            u_edge_weight.contiguous() if has_u_edge_weight else edge_weight
        )
        u_edge_bias_arg = (
            u_edge_bias.contiguous() if has_u_edge_bias else edge_bias
        )
        u_edge_residual_arg = (
            u_edge_residual.contiguous()
            if has_u_edge_residual else edge_residual
        )

        p_node = torch.empty((M, c_node), device=edge_ebd.device, dtype=edge_ebd.dtype)
        p_edge = torch.empty_like(edge_ebd)
        t_node = torch.empty_like(p_node)
        t_edge = torch.empty_like(edge_ebd)
        s_node = torch.empty_like(p_node)
        s_edge = torch.empty_like(edge_ebd)
        gg_edge_output = torch.empty_like(edge_ebd)
        if M:
            prepare_kernel = fused_edge_block_double_prepare(
                M=M, N_NODE=n_node, N_NODE_EXT=n_node_ext,
                C_NODE=c_node, C_EDGE=c_edge,
                HAS_GRAD_NODE=ctx.has_grad_node,
                HAS_GRAD_EDGE=ctx.has_grad_edge,
                HAS_U_NODE=has_u_node,
                HAS_U_NODE_EXT=has_u_node_ext,
                HAS_U_EDGE=has_u_edge,
                HAS_U_SW=has_u_sw,
                HAS_U_NODE_WEIGHT=has_u_node_weight,
                HAS_U_NODE_BIAS=has_u_node_bias,
                HAS_U_NODE_RESIDUAL=has_u_node_residual,
                HAS_U_EDGE_WEIGHT=has_u_edge_weight,
                HAS_U_EDGE_BIAS=has_u_edge_bias,
                HAS_U_EDGE_RESIDUAL=has_u_edge_residual,
            )
            prepare_kernel(
                grad_node_output, grad_edge_output,
                node_ebd, node_ebd_ext, edge_ebd, sw,
                owner, n_ext2e_index, node_weight, node_residual,
                edge_weight, edge_residual, node_preact, edge_preact,
                u_node_arg, u_node_ext_arg, u_edge_arg, u_sw_arg,
                u_node_weight_arg, u_node_bias_arg, u_node_residual_arg,
                u_edge_weight_arg, u_edge_bias_arg, u_edge_residual_arg,
                float(ctx.scale_factor), float(ctx.threshold),
                float(ctx.slope), float(ctx.const_value),
                p_node, p_edge, t_node, t_edge, s_node, s_edge,
                gg_edge_output,
            )
        else:
            gg_edge_output.zero_()

        gg_node_output = torch.empty_like(grad_node_output)
        uniform, offsets, order = ctx.owner_metadata
        if M == 0:
            if has_u_node_partial:
                gg_node_output.copy_(u_node_partial_arg)
            else:
                gg_node_output.zero_()
        else:
            node_output_factory = (
                fused_edge_block_double_node_output_uniform
                if uniform else fused_edge_block_double_node_output_segmented
            )
            node_output_kernel = node_output_factory(
                M=M, NO=ctx.num_owner, C_NODE=c_node,
                HAS_U_NODE_PARTIAL=has_u_node_partial,
                HAS_U_SW=has_u_sw,
                HAS_U_NODE_RESIDUAL=has_u_node_residual,
            )
            metadata_args = () if uniform else (offsets, order)
            node_output_kernel(
                node_preact, t_node, sw, node_residual,
                u_node_partial_arg, u_sw_arg, u_node_residual_arg,
                *metadata_args, float(ctx.scale_factor),
                float(ctx.threshold), float(ctx.slope),
                float(ctx.const_value), gg_node_output,
            )

        grad_node = torch.zeros_like(node_ebd)
        grad_node_ext = torch.zeros_like(node_ebd_ext)
        grad_edge = torch.empty_like(edge_ebd)
        if M:
            inputs_kernel = fused_edge_block_double_inputs(
                M=M, N_NODE=n_node, N_NODE_EXT=n_node_ext,
                C_NODE=c_node, C_EDGE=c_edge,
                HAS_U_NODE_WEIGHT=has_u_node_weight,
                HAS_U_EDGE_WEIGHT=has_u_edge_weight,
            )
            inputs_kernel(
                p_node, p_edge, s_node, s_edge,
                node_weight, edge_weight,
                u_node_weight_arg, u_edge_weight_arg,
                owner, n_ext2e_index, grad_node, grad_node_ext, grad_edge,
            )
        else:
            grad_edge.zero_()

        split_m = 1
        grad_node_weight = torch.empty_like(node_weight)
        grad_edge_weight = torch.empty_like(edge_weight)
        if M:
            weight_kernel = fused_edge_block_double_weight_partials(
                M=M, N_NODE=n_node, N_NODE_EXT=n_node_ext,
                C_NODE=c_node, C_EDGE=c_edge, SPLIT_M=split_m,
                HAS_U_NODE=has_u_node,
                HAS_U_NODE_EXT=has_u_node_ext,
                HAS_U_EDGE=has_u_edge,
            )
            if split_m == 1:
                node_partials = grad_node_weight.unsqueeze(0)
                edge_partials = grad_edge_weight.unsqueeze(0)
            else:
                node_partials = torch.empty(
                    (split_m, d, c_node), device=edge_ebd.device,
                    dtype=edge_ebd.dtype,
                )
                edge_partials = torch.empty(
                    (split_m, d, c_edge), device=edge_ebd.device,
                    dtype=edge_ebd.dtype,
                )
            weight_kernel(
                node_ebd, node_ebd_ext, edge_ebd,
                u_node_arg, u_node_ext_arg, u_edge_arg,
                owner, n_ext2e_index, p_node, p_edge, s_node, s_edge,
                node_partials, edge_partials,
            )
            if split_m > 1:
                reduce_kernel = fused_edge_block_double_weight_reduce(
                    D=d, C_NODE=c_node, C_EDGE=c_edge, SPLIT_M=split_m,
                )
                reduce_kernel(
                    node_partials, edge_partials,
                    grad_node_weight, grad_edge_weight,
                )
        else:
            grad_node_weight.zero_()
            grad_edge_weight.zero_()

        grad_node_bias = torch.empty_like(node_bias)
        grad_node_residual = torch.empty_like(node_residual)
        grad_edge_bias = torch.empty_like(edge_bias)
        grad_edge_residual = torch.empty_like(edge_residual)
        grad_sw = torch.empty_like(sw)
        if M:
            vector_kernel = fused_edge_block_double_vectors(
                M=M, N_NODE=n_node, C_NODE=c_node, C_EDGE=c_edge,
                HAS_GRAD_NODE=ctx.has_grad_node,
                HAS_GRAD_EDGE=ctx.has_grad_edge,
                HAS_U_SW=has_u_sw,
                HAS_U_NODE_RESIDUAL=has_u_node_residual,
            )
            vector_kernel(
                grad_node_output, grad_edge_output,
                node_preact, edge_preact, t_node, t_edge, s_node, s_edge,
                sw, owner, node_residual, u_sw_arg, u_node_residual_arg,
                float(ctx.scale_factor), float(ctx.threshold),
                float(ctx.slope), float(ctx.const_value),
                grad_node_bias, grad_node_residual,
                grad_edge_bias, grad_edge_residual, grad_sw,
            )
        else:
            grad_node_bias.zero_()
            grad_node_residual.zero_()
            grad_edge_bias.zero_()
            grad_edge_residual.zero_()
            grad_sw.zero_()

        return (
            gg_node_output, gg_edge_output,
            None, grad_node, grad_node_ext, grad_edge, grad_sw,
            None, None,
            grad_node_weight, grad_node_bias, grad_node_residual,
            grad_edge_weight, grad_edge_bias, grad_edge_residual,
            None, None,
            None, None, None, None, None, None, None, None,
        )


def fused_edge_block_dynamic(
    node_partial,
    node_ebd,
    node_ebd_ext,
    edge_ebd,
    sw,
    owner,
    n_ext2e_index,
    node_weight,
    node_bias,
    node_residual,
    edge_weight,
    edge_bias,
    edge_residual,
    num_owner,
    scale_factor,
    threshold,
    slope,
    const_value,
    owner_metadata,
):
    """Run the wider twice-differentiable Edge block without materializing cat."""
    node_shape = node_partial.shape
    node_partial_flat = node_partial.reshape(num_owner, node_shape[-1])
    node_flat = node_ebd.reshape(num_owner, node_ebd.shape[-1])
    node_ext_flat = node_ebd_ext.reshape(-1, node_ebd_ext.shape[-1])
    node_output, edge_output = FusedEdgeBlockFunction.apply(
        node_partial_flat, node_flat, node_ext_flat, edge_ebd, sw,
        owner, n_ext2e_index,
        node_weight, node_bias, node_residual,
        edge_weight, edge_bias, edge_residual,
        num_owner, scale_factor, threshold, slope, const_value,
        owner_metadata,
    )
    return node_output.reshape(node_shape), edge_output
