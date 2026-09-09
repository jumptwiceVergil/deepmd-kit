"""Rebuilt legacy tiled baseline, vendored from repository commit 6ff0569.

Only reachable functions/classes were retained; inert comments/docstrings removed.
The edge input bias fragment is cleared per K tile to fix the known legacy bug.
This is labelled legacy_rebuilt, NOT an exact historical timing snapshot.
No source files from the installed package are modified.
"""
from deepmd.pt.model.descriptor import utils_tilelang as _installed
# Supply imports/constants from the same installation, then override the full
# reachable implementation below. Do not copy module identity fields.
globals().update({k: v for k, v in vars(_installed).items() if not k.startswith("__")})
from collections import OrderedDict
_sym_owner_cache = OrderedDict()
BASELINE_COMMIT = "6ff0569"
BASELINE_FIXES = ["edge bias_tile reset for each K tile"]
# Snapshot SHA256: 1083146b64a1ad22181c43df426c85833f663a2949790b4fa78eb16c84f43e99
def _sym_owner_metadata(owner, num_owner):
    if num_owner <= 0 or owner.ndim != 1:
        raise ValueError('owner must be one-dimensional and num_owner positive')
    if owner.dtype not in (torch.int32, torch.int64):
        raise TypeError('owner must use int32 or int64 indices')
    base = owner
    while base._base is not None:
        base = base._base
    key = (id(base), owner.data_ptr(), owner.numel(), owner.stride(), owner._version, num_owner, owner.device)
    cached = _sym_owner_cache.get(key)
    if cached is not None and cached[0]() is base:
        _sym_owner_cache.move_to_end(key)
        return cached[1]
    if owner.numel() and bool(((owner < 0) | (owner >= num_owner)).any()):
        raise ValueError('owner indices must be in [0, num_owner)')
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
        order = torch.arange(count, device=owner.device, dtype=torch.int64) if sorted_owner else torch.argsort(owner, stable=True)
        result = (False, offsets, order)
    _sym_owner_cache[key] = (weakref.ref(base), result)
    _sym_owner_cache.move_to_end(key)
    while len(_sym_owner_cache) > _SYM_OWNER_CACHE_SIZE:
        _sym_owner_cache.popitem(last=False)
    return result

@tilelang.jit
def fused_cal_hg_dynamic_forward_segmented(M, E, NO, BLOCK_N=64):

    @T.prim_func
    def kernel(edge: T.Tensor((M, E), 'float32'), sw: T.Tensor((M,), 'float32'), h: T.Tensor((M, 3), 'float32'), offsets: T.Tensor((NO + 1,), 'int64'), order: T.Tensor((M,), 'int64'), scale: T.float32, out: T.Tensor((NO, 3 * E), 'float32')):
        with T.Kernel(NO, T.ceildiv(3 * E, BLOCK_N), threads=64) as (o, tile):
            for j in T.Parallel(BLOCK_N):
                col = tile * BLOCK_N + j
                if col < 3 * E:
                    acc = T.alloc_var('float32', init=0)
                    for r in T.serial(offsets[o], offsets[o + 1]):
                        m = order[r]
                        acc += edge[m, col % E] * sw[m] * h[m, col // E]
                    out[o, col] = acc * scale
    return kernel

@tilelang.jit
def fused_cal_hg_dynamic_forward_v1(M, N, E, NO, dtype='float32', accum_dtype='float32', BLOCK_N: int=64):
    assert M % NO == 0
    EDGES_PER_OWNER = M // NO

    @T.prim_func
    def hg_kernel(flat_edge_ebd: T.Tensor((M, E), dtype), flat_sw: T.Tensor((M,), dtype), flat_h2: T.Tensor((M, 3), dtype), scale_factor: T.float32, out: T.Tensor((NO, N), accum_dtype)):
        with T.Kernel(NO, T.ceildiv(N, BLOCK_N), threads=64) as (bx, by):
            meta_shared = T.alloc_shared((EDGES_PER_OWNER, 4), dtype)
            for r in T.Parallel(EDGES_PER_OWNER):
                edge_idx = bx * EDGES_PER_OWNER + r
                meta_shared[r, 0] = flat_sw[edge_idx]
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
                    for r in T.serial(EDGES_PER_OWNER):
                        edge_idx = bx * EDGES_PER_OWNER + r
                        acc[j] += flat_edge_ebd[edge_idx, e_idx] * meta_shared[r, 0] * meta_shared[r, h2_idx + 1]
                    acc[j] = acc[j] * scale_factor
            for j in T.Parallel(BLOCK_N):
                col = by * BLOCK_N + j
                if col < N:
                    out[bx, col] = acc[j]
    return hg_kernel

@tilelang.jit
def fused_call_grrg_forward_v0(NB, NLOC, E, AXIS, dtype='float32', accum_dtype='float32', BLOCK_M=4, BLOCK_N=32):

    @T.prim_func
    def grrg_kernel(h2g2: T.Tensor((NB, NLOC, 3, E), dtype), out: T.Tensor((NB, NLOC, AXIS * E), accum_dtype)):
        NUM_TILE_M = T.ceildiv(AXIS, BLOCK_M)
        NUM_TILE_N = T.ceildiv(E, BLOCK_N)
        with T.Kernel(NB, NLOC, NUM_TILE_M * NUM_TILE_N, threads=128) as (bx, by, bz):
            tile_m = bz // NUM_TILE_N
            tile_n = bz % NUM_TILE_N
            for a, e in T.Parallel(BLOCK_M, BLOCK_N):
                axis_idx = tile_m * BLOCK_M + a
                e_idx = tile_n * BLOCK_N + e
                if axis_idx < AXIS and e_idx < E:
                    acc = T.alloc_var(accum_dtype, 0)
                    for k in T.serial(3):
                        acc += h2g2[bx, by, k, axis_idx] * h2g2[bx, by, k, e_idx]
                    acc = acc / 3.0 ** 1
                    out[bx, by, axis_idx * E + e_idx] = acc
    return grrg_kernel

class FusedSymmetrizationOpDynamic(torch.autograd.Function):

    @staticmethod
    def forward(ctx, flat_edge_ebd: torch.Tensor, flat_h2: torch.Tensor, flat_sw: torch.Tensor, owner: torch.Tensor, num_owner: int, nb: int, nloc: int, scale_factor: float, axis_neuron: int) -> torch.Tensor:
        n_edge, e_dim = flat_edge_ebd.shape
        h2g2 = torch.empty((num_owner, 3 * e_dim), device=flat_edge_ebd.device, dtype=flat_edge_ebd.dtype)
        uniform, offsets, order = _sym_owner_metadata(owner, num_owner)
        if n_edge == 0:
            h2g2.zero_()
        elif uniform:
            hg_kernel = fused_cal_hg_dynamic_forward_v1(M=n_edge, N=3 * e_dim, E=e_dim, NO=num_owner)
            hg_kernel(flat_edge_ebd, flat_sw, flat_h2, scale_factor, h2g2)
        else:
            hg_kernel = fused_cal_hg_dynamic_forward_segmented(M=n_edge, E=e_dim, NO=num_owner)
            hg_kernel(flat_edge_ebd, flat_sw, flat_h2, offsets, order, scale_factor, h2g2)
        h2g2 = h2g2.reshape(nb, nloc, 3, e_dim)
        grrg = torch.empty((nb, nloc, axis_neuron * e_dim), device=h2g2.device, dtype=h2g2.dtype)
        grrg_kernel = fused_call_grrg_forward_v0(NB=nb, NLOC=nloc, E=e_dim, AXIS=axis_neuron)
        grrg_kernel(h2g2, grrg)
        ctx.save_for_backward(flat_edge_ebd, flat_h2, flat_sw, owner, h2g2)
        ctx.nb = nb
        ctx.nloc = nloc
        ctx.num_owner = num_owner
        ctx.scale_factor = scale_factor
        ctx.axis_neuron = axis_neuron
        return grrg

    @staticmethod
    def backward(ctx, grad_grrg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, float, int]:
        flat_edge_ebd, flat_h2, flat_sw, owner, h2g2 = ctx.saved_tensors
        nb = ctx.nb
        nloc = ctx.nloc
        num_owner = ctx.num_owner
        scale_factor = ctx.scale_factor
        axis_neuron = ctx.axis_neuron
        assert nb == 1, 'only support nb=1.'
        grad_flat_edge_ebd, grad_h2, grad_flat_sw = FusedSymmetrizationOpDynamicBackward.apply(grad_grrg, flat_edge_ebd, flat_h2, flat_sw, owner, h2g2, nb, nloc, num_owner, scale_factor, axis_neuron)
        return (grad_flat_edge_ebd, grad_h2, grad_flat_sw, None, None, None, None, None, None)

@tilelang.jit
def fused_call_hg_dynamic_backward(M, E, NB, NLOC, dtype='float32', accum_dtype='float32', BLOCK_D: T.int32=64):

    @T.prim_func
    def hg_backward(grad_h2g2: T.Tensor((NB, NLOC, 3, E), dtype), flat_edge_ebd: T.Tensor((M, E), dtype), flat_h2: T.Tensor((M, 3), dtype), flat_sw: T.Tensor((M,), dtype), owner: T.Tensor((M,), 'int64'), grad_flat_h2g2: T.Tensor((M, 3, E), accum_dtype), grad_flat_edge_ebd: T.Tensor((M, E), accum_dtype), grad_h2: T.Tensor((M, 3), accum_dtype), grad_flat_sw: T.Tensor((M,), accum_dtype)):
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
def fused_call_grrg_backward(NB, NLOC, E, A, dtype='float32', THREADS=128):

    @T.prim_func
    def grrg_backward(grad_grrg: T.Tensor((NB, NLOC, A * E), dtype), h2g2: T.Tensor((NB, NLOC, 3, E), dtype), grad_h2g2: T.Tensor((NB, NLOC, 3, E), dtype), scale_factor: T.float32):
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
def fused_call_grrg_backward_general(NB, NLOC, E, A, dtype='float32', THREADS=128):

    @T.prim_func
    def kernel(grad_grrg: T.Tensor((NB, NLOC, A * E), dtype), h2g2: T.Tensor((NB, NLOC, 3, E), dtype), grad_h2g2: T.Tensor((NB, NLOC, 3, E), dtype), scale_factor: T.float32):
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
def fused_symmetrization_double_backward_owner(M, E, NO, A, dtype='float32', accum_dtype='float32', THREADS=128):
    assert M % NO == 0
    EDGES_PER_OWNER = M // NO

    @T.prim_func
    def double_backward_owner(grad_grrg: T.Tensor((1, NO, A * E), dtype), h2g2: T.Tensor((1, NO, 3, E), dtype), flat_edge_ebd: T.Tensor((M, E), dtype), flat_h2: T.Tensor((M, 3), dtype), flat_sw: T.Tensor((M,), dtype), grad_grad_edge: T.Tensor((M, E), dtype), grad_grad_h2: T.Tensor((M, 3), dtype), grad_grad_sw: T.Tensor((M,), dtype), scale_factor: T.float32, grad_grad_grrg: T.Tensor((1, NO, A * E), accum_dtype), grad_h2g2: T.Tensor((1, NO, 3, E), accum_dtype)):
        with T.Kernel(NO, threads=THREADS) as (owner_idx,):
            grad_q = T.alloc_shared((3, E), accum_dtype)
            for b, d in T.Parallel(3, E):
                acc = T.alloc_var(accum_dtype, init=0)
                for r in T.serial(EDGES_PER_OWNER):
                    edge_idx = owner_idx * EDGES_PER_OWNER + r
                    edge = flat_edge_ebd[edge_idx, d]
                    h = flat_h2[edge_idx, b]
                    sw = flat_sw[edge_idx]
                    acc += grad_grad_h2[edge_idx, b] * edge * sw + grad_grad_edge[edge_idx, d] * h * sw + grad_grad_sw[edge_idx] * h * edge
                grad_q[b, d] = acc
            T.sync_threads()
            c = scale_factor / 3.0
            for a, d in T.Parallel(A, E):
                acc = T.alloc_var(accum_dtype, init=0)
                for b in T.serial(3):
                    acc += h2g2[0, owner_idx, b, a] * grad_q[b, d] + grad_q[b, a] * h2g2[0, owner_idx, b, d]
                grad_grad_grrg[0, owner_idx, a * E + d] = acc * c
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
def fused_symmetrization_double_backward_edge(M, E, NO, dtype='float32', accum_dtype='float32', THREADS=128):

    @T.prim_func
    def double_backward_edge(grad_flat_h2g2: T.Tensor((M, 3, E), dtype), grad_h2g2: T.Tensor((1, NO, 3, E), dtype), flat_edge_ebd: T.Tensor((M, E), dtype), flat_h2: T.Tensor((M, 3), dtype), flat_sw: T.Tensor((M,), dtype), owner: T.Tensor((M,), 'int64'), grad_grad_edge: T.Tensor((M, E), dtype), grad_grad_h2: T.Tensor((M, 3), dtype), grad_grad_sw: T.Tensor((M,), dtype), scale_factor: T.float32, grad_flat_edge_ebd: T.Tensor((M, E), accum_dtype), grad_flat_h2: T.Tensor((M, 3), accum_dtype), grad_flat_sw: T.Tensor((M,), accum_dtype)):
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
                partial_h[b, d] = grad_grad_edge[edge_idx, d] * sw * r + grad_grad_sw[edge_idx] * edge * r + gh * edge * sw
                partial_sw[b, d] = grad_grad_h2[edge_idx, b] * edge * r + grad_grad_edge[edge_idx, d] * h * r + gh * h * edge
            for d in T.Parallel(E):
                acc = T.alloc_var(accum_dtype, init=0)
                for b in T.serial(3):
                    r = grad_flat_h2g2[edge_idx, b, d]
                    gh = grad_h2g2[0, owner_idx, b, d] * scale_factor
                    acc += grad_grad_h2[edge_idx, b] * r * sw + grad_grad_sw[edge_idx] * r * flat_h2[edge_idx, b] + gh * flat_h2[edge_idx, b] * sw
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
def fused_symmetrization_double_backward_owner_segmented(M, E, NO, A, dtype='float32', accum_dtype='float32', THREADS=128):

    @T.prim_func
    def double_backward_owner(grad_grrg: T.Tensor((1, NO, A * E), dtype), h2g2: T.Tensor((1, NO, 3, E), dtype), flat_edge_ebd: T.Tensor((M, E), dtype), flat_h2: T.Tensor((M, 3), dtype), flat_sw: T.Tensor((M,), dtype), grad_grad_edge: T.Tensor((M, E), dtype), grad_grad_h2: T.Tensor((M, 3), dtype), grad_grad_sw: T.Tensor((M,), dtype), offsets: T.Tensor((NO + 1,), 'int64'), order: T.Tensor((M,), 'int64'), scale_factor: T.float32, grad_grad_grrg: T.Tensor((1, NO, A * E), accum_dtype), grad_h2g2: T.Tensor((1, NO, 3, E), accum_dtype)):
        with T.Kernel(NO, threads=THREADS) as (owner_idx,):
            grad_q = T.alloc_shared((3, E), accum_dtype)
            for b, d in T.Parallel(3, E):
                acc = T.alloc_var(accum_dtype, init=0)
                for r in T.serial(offsets[owner_idx], offsets[owner_idx + 1]):
                    edge_idx = order[r]
                    edge = flat_edge_ebd[edge_idx, d]
                    h = flat_h2[edge_idx, b]
                    sw = flat_sw[edge_idx]
                    acc += grad_grad_h2[edge_idx, b] * edge * sw + grad_grad_edge[edge_idx, d] * h * sw + grad_grad_sw[edge_idx] * h * edge
                grad_q[b, d] = acc
            T.sync_threads()
            c = scale_factor / 3.0
            for a, d in T.Parallel(A, E):
                acc = T.alloc_var(accum_dtype, init=0)
                for b in T.serial(3):
                    acc += h2g2[0, owner_idx, b, a] * grad_q[b, d] + grad_q[b, a] * h2g2[0, owner_idx, b, d]
                grad_grad_grrg[0, owner_idx, a * E + d] = acc * c
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
    def forward(ctx, grad_grrg: torch.Tensor, flat_edge_ebd: torch.Tensor, flat_h2: torch.Tensor, flat_sw: torch.Tensor, owner: torch.Tensor, h2g2: torch.Tensor, nb: int, nloc: int, num_owner: int, scale_factor: float, axis_neuron: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M, E = flat_edge_ebd.shape
        assert flat_edge_ebd.shape[-1] == h2g2.shape[-1]
        assert grad_grrg.shape[-1] == axis_neuron * E
        grad_grrg = grad_grrg.contiguous()
        h2g2 = h2g2.contiguous()
        grad_h2g2 = torch.empty((nb, nloc, 3, E), dtype=grad_grrg.dtype, device=grad_grrg.device)
        regular_extents = E > 0 and E & E - 1 == 0 and (axis_neuron > 0) and (axis_neuron & axis_neuron - 1 == 0)
        grrg_factory = fused_call_grrg_backward if regular_extents else fused_call_grrg_backward_general
        grrg_backward_kernel = grrg_factory(NB=nb, NLOC=nloc, E=E, A=axis_neuron)
        grrg_backward_kernel(grad_grrg, h2g2, grad_h2g2, float(scale_factor))
        ctx.owner_metadata = _sym_owner_metadata(owner, num_owner)
        owner = owner.long()
        grad_flat_h2g2 = torch.empty((M, 3, E), dtype=grad_h2g2.dtype, device=grad_h2g2.device)
        grad_flat_edge_ebd = torch.empty_like(flat_edge_ebd)
        grad_h2 = torch.empty_like(flat_h2)
        grad_flat_sw = torch.empty_like(flat_sw)
        if M:
            hg_backward_kernel = fused_call_hg_dynamic_backward(M=M, E=E, NB=nb, NLOC=nloc)
            hg_backward_kernel(grad_h2g2, flat_edge_ebd, flat_h2, flat_sw, owner, grad_flat_h2g2, grad_flat_edge_ebd, grad_h2, grad_flat_sw)
        ctx.save_for_backward(grad_grrg, flat_edge_ebd, flat_h2, flat_sw, owner, h2g2, grad_flat_h2g2)
        ctx.nb = nb
        ctx.nloc = nloc
        ctx.num_owner = num_owner
        ctx.scale_factor = scale_factor
        ctx.axis_neuron = axis_neuron
        return (grad_flat_edge_ebd, grad_h2, grad_flat_sw)

    @staticmethod
    def backward(ctx, grad_grad_edge: torch.Tensor, grad_grad_h2: torch.Tensor, grad_grad_sw: torch.Tensor):
        grad_grrg, flat_edge_ebd, flat_h2, flat_sw, owner, h2g2, grad_flat_h2g2 = ctx.saved_tensors
        num_owner = ctx.num_owner
        scale_factor = ctx.scale_factor
        axis_neuron = ctx.axis_neuron
        M, E = flat_edge_ebd.shape
        dtype = str(grad_grrg.dtype).replace('torch.', '')
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
        if M == 0:
            return (torch.zeros_like(grad_grrg), torch.zeros_like(flat_edge_ebd), torch.zeros_like(flat_h2), torch.zeros_like(flat_sw), None, None, None, None, None, None, None)
        uniform, offsets, order = ctx.owner_metadata
        factory = fused_symmetrization_double_backward_owner if uniform else fused_symmetrization_double_backward_owner_segmented
        owner_kernel = factory(M=M, E=E, NO=num_owner, A=axis_neuron, dtype=dtype)
        metadata_args = () if uniform else (offsets, order)
        owner_kernel(grad_grrg, h2g2, flat_edge_ebd, flat_h2, flat_sw, grad_grad_edge, grad_grad_h2, grad_grad_sw, *metadata_args, float(scale_factor), grad_grad_grrg, grad_h2g2)
        grad_flat_edge_ebd = torch.empty_like(flat_edge_ebd)
        grad_flat_h2 = torch.empty_like(flat_h2)
        grad_flat_sw = torch.empty_like(flat_sw)
        edge_kernel = fused_symmetrization_double_backward_edge(M=M, E=E, NO=num_owner, dtype=dtype)
        edge_kernel(grad_flat_h2g2, grad_h2g2, flat_edge_ebd, flat_h2, flat_sw, owner, grad_grad_edge, grad_grad_h2, grad_grad_sw, float(scale_factor), grad_flat_edge_ebd, grad_flat_h2, grad_flat_sw)
        return (grad_grad_grrg, grad_flat_edge_ebd, grad_flat_h2, grad_flat_sw, None, None, None, None, None, None, None)

@tilelang.jit
def fused_edge_update_forward(N_EDGES: int, N_NODES_LOC: int, N_NODES_EXT: int, NODE_DIM: int, EDGE_DIM: int, OUT_DIM: int, BLK_M: int=128, BLK_N: int=64, BLK_K: int=64):

    @T.prim_func
    def kernel(node_ebd: T.Tensor((N_NODES_LOC, NODE_DIM), 'float32'), node_ebd_ext: T.Tensor((N_NODES_EXT, NODE_DIM), 'float32'), flat_edge_ebd: T.Tensor((N_EDGES, EDGE_DIM), 'float32'), n2e_index: T.Tensor((N_EDGES,), 'int64'), n_ext2e_index: T.Tensor((N_EDGES,), 'int64'), node: T.Tensor((NODE_DIM, OUT_DIM), 'float32'), node_ext: T.Tensor((NODE_DIM, OUT_DIM), 'float32'), edge: T.Tensor((EDGE_DIM, OUT_DIM), 'float32'), bias: T.Tensor((OUT_DIM,), 'float32'), out: T.Tensor((N_EDGES, OUT_DIM), 'float32')):
        with T.Kernel(T.ceildiv(N_EDGES, BLK_M), T.ceildiv(OUT_DIM, BLK_N), threads=128) as (bx, by):
            A = T.alloc_shared((BLK_M, BLK_K), 'float32')
            B = T.alloc_shared((BLK_K, BLK_N), 'float32')
            acc = T.alloc_fragment((BLK_M, BLK_N), 'float32')
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(NODE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    edge_id = bx * BLK_M + i
                    k_id = k * BLK_K + j
                    if edge_id < N_EDGES and k_id < NODE_DIM:
                        src = n2e_index[edge_id]
                        A[i, j] = node_ebd[src, k_id]
                    else:
                        A[i, j] = T.float32(0)
                for i, j in T.Parallel(BLK_K, BLK_N):
                    k_id = k * BLK_K + i
                    out_id = by * BLK_N + j
                    if k_id < NODE_DIM and out_id < OUT_DIM:
                        B[i, j] = node[k_id, out_id]
                    else:
                        B[i, j] = T.float32(0)
                T.gemm(A, B, acc)
            for k in T.Pipelined(T.ceildiv(NODE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    edge_id = bx * BLK_M + i
                    k_id = k * BLK_K + j
                    if edge_id < N_EDGES and k_id < NODE_DIM:
                        src = n_ext2e_index[edge_id]
                        A[i, j] = node_ebd_ext[src, k_id]
                    else:
                        A[i, j] = T.float32(0)
                for i, j in T.Parallel(BLK_K, BLK_N):
                    k_id = k * BLK_K + i
                    out_id = by * BLK_N + j
                    if k_id < NODE_DIM and out_id < OUT_DIM:
                        B[i, j] = node_ext[k_id, out_id]
                    else:
                        B[i, j] = T.float32(0)
                T.gemm(A, B, acc)
            for k in T.Pipelined(T.ceildiv(EDGE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    edge_id = bx * BLK_M + i
                    k_id = k * BLK_K + j
                    if edge_id < N_EDGES and k_id < EDGE_DIM:
                        A[i, j] = flat_edge_ebd[edge_id, k_id]
                    else:
                        A[i, j] = T.float32(0)
                for i, j in T.Parallel(BLK_K, BLK_N):
                    k_id = k * BLK_K + i
                    out_id = by * BLK_N + j
                    if k_id < EDGE_DIM and out_id < OUT_DIM:
                        B[i, j] = edge[k_id, out_id]
                    else:
                        B[i, j] = T.float32(0)
                T.gemm(A, B, acc)
            for i, j in T.Parallel(BLK_M, BLK_N):
                if bx * BLK_M + i < N_EDGES and by * BLK_N + j < OUT_DIM:
                    out[bx * BLK_M + i, by * BLK_N + j] = acc[i, j] + bias[by * BLK_N + j]
    return kernel

class FusedEdgeUpdateFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, node_ebd: torch.Tensor, node_ebd_ext: torch.Tensor, flat_edge_ebd: torch.Tensor, n2e_index: torch.Tensor, n_ext2e_index: torch.Tensor, node: torch.Tensor, node_ext: torch.Tensor, edge: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        n_nodes_loc = node_ebd.shape[0]
        n_nodes_ext = node_ebd_ext.shape[0]
        n_edges = flat_edge_ebd.shape[0]
        node_dim = node_ebd.shape[-1]
        edge_dim = flat_edge_ebd.shape[-1]
        out_dim = node.shape[-1]
        edge_forward_kernel = fused_edge_update_forward(N_EDGES=n_edges, N_NODES_LOC=n_nodes_loc, N_NODES_EXT=n_nodes_ext, NODE_DIM=node_dim, EDGE_DIM=edge_dim, OUT_DIM=out_dim)
        out = torch.empty((n_edges, out_dim), device=node_ebd.device, dtype=node_ebd.dtype)
        edge_forward_kernel(node_ebd, node_ebd_ext, flat_edge_ebd, n2e_index, n_ext2e_index, node, node_ext, edge, bias, out)
        ctx.save_for_backward(node_ebd, node_ebd_ext, flat_edge_ebd, n2e_index, n_ext2e_index, node, node_ext, edge)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_ebd, node_ebd_ext, flat_edge_ebd, n2e_index, n_ext2e_index, node_weight, node_ext_weight, edge_weight = ctx.saved_tensors
        output = FusedEdgeUpdateFunctionBackward.apply(grad_out, node_ebd, node_ebd_ext, flat_edge_ebd, n2e_index, n_ext2e_index, node_weight, node_ext_weight, edge_weight)
        grad_node, grad_node_ext, grad_edge_ebd, _, _, grad_node_weight, grad_node_ext_weight, grad_edge_weight, grad_bias = output
        return (grad_node, grad_node_ext, grad_edge_ebd, None, None, grad_node_weight, grad_node_ext_weight, grad_edge_weight, grad_bias)

@tilelang.jit
def fused_edge_update_weight_backward_v1(E, K, D_edge, N_node, D_node, N_ext, D_ext, dtype='float32', accum_dtype='float32', BLOCK_E=32, BLOCK_D=16, BLOCK_K=32):
    MAX_D = max(D_edge, D_node, D_ext)

    @T.prim_func
    def weight_backward(grad_out: T.Tensor((E, K), dtype), flat_edge_ebd: T.Tensor((E, D_edge), dtype), node_ebd: T.Tensor((N_node, D_node), dtype), node_ebd_ext: T.Tensor((N_ext, D_ext), dtype), n2e_index: T.Tensor((E,), 'int64'), n_ext2e_index: T.Tensor((E,), 'int64'), grad_edge_weight: T.Tensor((D_edge, K), accum_dtype), grad_node_weight: T.Tensor((D_node, K), accum_dtype), grad_node_ext_weight: T.Tensor((D_ext, K), accum_dtype)):
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
def fused_edge_update_input_backward_v1(E, K, D_edge, D_node, D_ext, N_node, N_ext, dtype='float32', accum_dtype='float32', BLOCK_E=32, BLOCK_D=16, BLOCK_K=32):
    MAX_D = max(D_edge, D_node, D_ext)

    @T.prim_func
    def input_backward(grad_out: T.Tensor((E, K), dtype), edge_weight: T.Tensor((D_edge, K), dtype), node_weight: T.Tensor((D_node, K), dtype), node_ext_weight: T.Tensor((D_ext, K), dtype), n2e_index: T.Tensor((E,), 'int64'), n_ext2e_index: T.Tensor((E,), 'int64'), grad_edge_ebd: T.Tensor((E, D_edge), accum_dtype), grad_node: T.Tensor((N_node, D_node), accum_dtype), grad_node_ext: T.Tensor((N_ext, D_ext), accum_dtype), grad_bias: T.Tensor((K,), accum_dtype)):
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
                # Rebuilt baseline correctness fix; not a performance optimization.
                T.clear(bias_tile)
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
def fused_edge_update_double_backward_inputs(E, K, D_edge, D_node, D_ext, N_node, N_ext, dtype='float32', accum_dtype='float32', THREADS=128, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32):

    @T.prim_func
    def double_backward_inputs(grad_out: T.Tensor((E, K), dtype), node_ebd: T.Tensor((N_node, D_node), dtype), node_ebd_ext: T.Tensor((N_ext, D_ext), dtype), flat_edge_ebd: T.Tensor((E, D_edge), dtype), n2e_index: T.Tensor((E,), 'int64'), n_ext2e_index: T.Tensor((E,), 'int64'), node_weight: T.Tensor((D_node, K), dtype), node_ext_weight: T.Tensor((D_ext, K), dtype), edge_weight: T.Tensor((D_edge, K), dtype), grad_grad_node: T.Tensor((N_node, D_node), dtype), grad_grad_node_ext: T.Tensor((N_ext, D_ext), dtype), grad_grad_edge_ebd: T.Tensor((E, D_edge), dtype), grad_grad_node_weight: T.Tensor((D_node, K), dtype), grad_grad_node_ext_weight: T.Tensor((D_ext, K), dtype), grad_grad_edge_weight: T.Tensor((D_edge, K), dtype), grad_grad_bias: T.Tensor((K,), dtype), grad_grad_out: T.Tensor((E, K), accum_dtype), grad_node_ebd: T.Tensor((N_node, D_node), accum_dtype), grad_node_ebd_ext: T.Tensor((N_ext, D_ext), accum_dtype), grad_flat_edge_ebd: T.Tensor((E, D_edge), accum_dtype)):
        with T.Kernel(T.ceildiv(E, BLOCK_M), T.ceildiv(max(K, D_edge, D_node, D_ext), BLOCK_N), threads=THREADS) as (bx, by):
            lhs_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            rhs_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
            acc_output = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            acc_input = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            T.clear(acc_output)
            if by * BLOCK_N < K:
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
                    grad_grad_out[m, n] = acc_output[mi, ni] + grad_grad_bias[n]
            if by * BLOCK_N < D_edge:
                T.clear(acc_input)
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
                            rhs_shared[ri, ni] = grad_grad_edge_weight[n, r]
                        else:
                            rhs_shared[ri, ni] = 0
                    T.sync_threads()
                    T.gemm(lhs_shared, rhs_shared, acc_input)
                    T.sync_threads()
                T.sync_threads()
                for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                    m = bx * BLOCK_M + mi
                    n = by * BLOCK_N + ni
                    if m < E and n < D_edge:
                        grad_flat_edge_ebd[m, n] = acc_input[mi, ni]
                T.sync_threads()
            if by * BLOCK_N < D_node:
                T.clear(acc_input)
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
                            rhs_shared[ri, ni] = grad_grad_node_weight[n, r]
                        else:
                            rhs_shared[ri, ni] = 0
                    T.sync_threads()
                    T.gemm(lhs_shared, rhs_shared, acc_input)
                    T.sync_threads()
                T.sync_threads()
                for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                    m = bx * BLOCK_M + mi
                    n = by * BLOCK_N + ni
                    if m < E and n < D_node:
                        T.atomic_add(grad_node_ebd[n2e_index[m], n], acc_input[mi, ni])
                T.sync_threads()
            if by * BLOCK_N < D_ext:
                T.clear(acc_input)
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
                            rhs_shared[ri, ni] = grad_grad_node_ext_weight[n, r]
                        else:
                            rhs_shared[ri, ni] = 0
                    T.sync_threads()
                    T.gemm(lhs_shared, rhs_shared, acc_input)
                    T.sync_threads()
                T.sync_threads()
                for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                    m = bx * BLOCK_M + mi
                    n = by * BLOCK_N + ni
                    if m < E and n < D_ext:
                        T.atomic_add(grad_node_ebd_ext[n_ext2e_index[m], n], acc_input[mi, ni])
                T.sync_threads()
    return double_backward_inputs

@tilelang.jit
def fused_edge_update_double_backward_weights(E, K, D_edge, D_node, D_ext, N_node, N_ext, dtype='float32', accum_dtype='float32', BLOCK_D=32, BLOCK_K=16, THREADS=128, BLOCK_M=32):
    MAX_D = max(D_edge, D_node, D_ext)

    @T.prim_func
    def double_backward_weights(grad_out: T.Tensor((E, K), dtype), n2e_index: T.Tensor((E,), 'int64'), n_ext2e_index: T.Tensor((E,), 'int64'), grad_grad_node: T.Tensor((N_node, D_node), dtype), grad_grad_node_ext: T.Tensor((N_ext, D_ext), dtype), grad_grad_edge_ebd: T.Tensor((E, D_edge), dtype), grad_node_weight: T.Tensor((D_node, K), accum_dtype), grad_node_ext_weight: T.Tensor((D_ext, K), accum_dtype), grad_edge_weight: T.Tensor((D_edge, K), accum_dtype)):
        with T.Kernel(T.ceildiv(MAX_D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=THREADS) as (bx, by):
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
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
    def forward(ctx, grad_out: torch.Tensor, node_ebd: torch.Tensor, node_ebd_ext: torch.Tensor, flat_edge_ebd: torch.Tensor, n2e_index: torch.Tensor, n_ext2e_index: torch.Tensor, node_weight: torch.Tensor, node_ext_weight: torch.Tensor, edge_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        E, K = grad_out.shape
        N_node, D_node = node_ebd.shape
        N_ext, D_ext = node_ebd_ext.shape
        E_edge, D_edge = flat_edge_ebd.shape
        grad_edge_weight = torch.empty((D_edge, K), device=grad_out.device, dtype=grad_out.dtype)
        grad_node_weight = torch.empty((D_node, K), device=grad_out.device, dtype=grad_out.dtype)
        grad_node_ext_weight = torch.empty((D_ext, K), device=grad_out.device, dtype=grad_out.dtype)
        weight_kernel = fused_edge_update_weight_backward_v1(E=E, K=K, D_edge=D_edge, N_node=N_node, D_node=D_node, N_ext=N_ext, D_ext=D_ext)
        weight_kernel(grad_out, flat_edge_ebd, node_ebd, node_ebd_ext, n2e_index, n_ext2e_index, grad_edge_weight, grad_node_weight, grad_node_ext_weight)
        grad_edge_ebd = torch.empty((E, D_edge), device=grad_out.device, dtype=grad_out.dtype)
        grad_node = torch.zeros((N_node, D_node), device=grad_out.device, dtype=grad_out.dtype)
        grad_node_ext = torch.zeros((N_ext, D_ext), device=grad_out.device, dtype=grad_out.dtype)
        grad_bias = torch.zeros((K,), device=grad_out.device, dtype=grad_out.dtype)
        input_kernel = fused_edge_update_input_backward_v1(E=E, K=K, D_edge=D_edge, D_node=D_node, D_ext=D_ext, N_node=N_node, N_ext=N_ext)
        input_kernel(grad_out, edge_weight, node_weight, node_ext_weight, n2e_index, n_ext2e_index, grad_edge_ebd, grad_node, grad_node_ext, grad_bias)
        ctx.save_for_backward(grad_out, node_ebd, node_ebd_ext, flat_edge_ebd, n2e_index, n_ext2e_index, node_weight, node_ext_weight, edge_weight)
        return (grad_node, grad_node_ext, grad_edge_ebd, None, None, grad_node_weight, grad_node_ext_weight, grad_edge_weight, grad_bias)

    @staticmethod
    def backward(ctx, grad_grad_node: torch.Tensor, grad_grad_node_ext: torch.Tensor, grad_grad_edge_ebd: torch.Tensor, grad_grad_n2e_index: torch.Tensor, grad_grad_n_ext2e_index: torch.Tensor, grad_grad_node_weight: torch.Tensor, grad_grad_node_ext_weight: torch.Tensor, grad_grad_edge_weight: torch.Tensor, grad_grad_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_out, node_ebd, node_ebd_ext, flat_edge_ebd, n2e_index, n_ext2e_index, node_weight, node_ext_weight, edge_weight = ctx.saved_tensors
        E, K = grad_out.shape
        N_node, D_node = node_ebd.shape
        N_ext, D_ext = node_ebd_ext.shape
        _, D_edge = flat_edge_ebd.shape
        dtype = str(grad_out.dtype).replace('torch.', '')
        optional_grads = (('grad_grad_node', grad_grad_node, node_ebd), ('grad_grad_node_ext', grad_grad_node_ext, node_ebd_ext), ('grad_grad_edge_ebd', grad_grad_edge_ebd, flat_edge_ebd), ('grad_grad_node_weight', grad_grad_node_weight, node_weight), ('grad_grad_node_ext_weight', grad_grad_node_ext_weight, node_ext_weight), ('grad_grad_edge_weight', grad_grad_edge_weight, edge_weight))
        materialized = {name: torch.zeros_like(reference) if value is None else value.contiguous() for name, value, reference in optional_grads}
        if grad_grad_bias is None:
            grad_grad_bias = torch.zeros((K,), device=grad_out.device, dtype=grad_out.dtype)
        else:
            grad_grad_bias = grad_grad_bias.contiguous()
        grad_grad_node = materialized['grad_grad_node']
        grad_grad_node_ext = materialized['grad_grad_node_ext']
        grad_grad_edge_ebd = materialized['grad_grad_edge_ebd']
        grad_grad_node_weight = materialized['grad_grad_node_weight']
        grad_grad_node_ext_weight = materialized['grad_grad_node_ext_weight']
        grad_grad_edge_weight = materialized['grad_grad_edge_weight']
        grad_grad_out = torch.empty_like(grad_out)
        grad_grad_node_ebd = torch.zeros_like(node_ebd)
        grad_grad_node_ebd_ext = torch.zeros_like(node_ebd_ext)
        grad_grad_flat_edge_ebd = torch.empty_like(flat_edge_ebd)
        input_kernel = fused_edge_update_double_backward_inputs(E=E, K=K, D_edge=D_edge, D_node=D_node, D_ext=D_ext, N_node=N_node, N_ext=N_ext, dtype=dtype)
        input_kernel(grad_out, node_ebd, node_ebd_ext, flat_edge_ebd, n2e_index, n_ext2e_index, node_weight, node_ext_weight, edge_weight, grad_grad_node, grad_grad_node_ext, grad_grad_edge_ebd, grad_grad_node_weight, grad_grad_node_ext_weight, grad_grad_edge_weight, grad_grad_bias, grad_grad_out, grad_grad_node_ebd, grad_grad_node_ebd_ext, grad_grad_flat_edge_ebd)
        grad_grad_node_weight_input = torch.empty_like(node_weight)
        grad_grad_node_ext_weight_input = torch.empty_like(node_ext_weight)
        grad_grad_edge_weight_input = torch.empty_like(edge_weight)
        weight_kernel = fused_edge_update_double_backward_weights(E=E, K=K, D_edge=D_edge, D_node=D_node, D_ext=D_ext, N_node=N_node, N_ext=N_ext, dtype=dtype)
        weight_kernel(grad_out, n2e_index, n_ext2e_index, grad_grad_node, grad_grad_node_ext, grad_grad_edge_ebd, grad_grad_node_weight_input, grad_grad_node_ext_weight_input, grad_grad_edge_weight_input)
        return (grad_grad_out, grad_grad_node_ebd, grad_grad_node_ebd_ext, grad_grad_flat_edge_ebd, None, None, grad_grad_node_weight_input, grad_grad_node_ext_weight_input, grad_grad_edge_weight_input)

@tilelang.jit
def fused_angle_update_forward(N_ANGLE: int, N_NODE: int, N_EDGE: int, ANGLE_DIM: int, NODE_DIM: int, EDGE_DIM: int, OUT_DIM: int, BLK_M: int=128, BLK_N: int=64, BLK_K: int=64):

    @T.prim_func
    def fused_angle_update(flat_angle_ebd: T.Tensor((N_ANGLE, ANGLE_DIM), 'float32'), node_ebd: T.Tensor((N_NODE, NODE_DIM), 'float32'), flat_edge_ebd: T.Tensor((N_EDGE, EDGE_DIM), 'float32'), n2a_index: T.Tensor((N_ANGLE,), 'int64'), eij2a_index: T.Tensor((N_ANGLE,), 'int64'), eik2a_index: T.Tensor((N_ANGLE,), 'int64'), angle_weight: T.Tensor((ANGLE_DIM, OUT_DIM), 'float32'), node_weight: T.Tensor((NODE_DIM, OUT_DIM), 'float32'), edge_ik_weight: T.Tensor((EDGE_DIM, OUT_DIM), 'float32'), edge_ij_weight: T.Tensor((EDGE_DIM, OUT_DIM), 'float32'), bias: T.Tensor((OUT_DIM,), 'float32'), out: T.Tensor((N_ANGLE, OUT_DIM), 'float32')):
        with T.Kernel(T.ceildiv(N_ANGLE, BLK_M), T.ceildiv(OUT_DIM, BLK_N), threads=128) as (bx, by):
            A = T.alloc_shared((BLK_M, BLK_K), 'float32')
            B = T.alloc_shared((BLK_K, BLK_N), 'float32')
            acc = T.alloc_fragment((BLK_M, BLK_N), 'float32')
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(ANGLE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    row_idx = bx * BLK_M + i
                    col_idx = k * BLK_K + j
                    if row_idx < N_ANGLE and col_idx < ANGLE_DIM:
                        A[i, j] = flat_angle_ebd[row_idx, col_idx]
                    else:
                        A[i, j] = T.float32(0)
                for i, j in T.Parallel(BLK_K, BLK_N):
                    row_idx = k * BLK_K + i
                    col_idx = by * BLK_N + j
                    if row_idx < ANGLE_DIM and col_idx < OUT_DIM:
                        B[i, j] = angle_weight[row_idx, col_idx]
                    else:
                        B[i, j] = T.float32(0)
                T.gemm(A, B, acc)
            for k in T.Pipelined(T.ceildiv(NODE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    angle_idx = bx * BLK_M + i
                    col_idx = k * BLK_K + j
                    if angle_idx < N_ANGLE and col_idx < NODE_DIM:
                        node_idx = n2a_index[angle_idx]
                        A[i, j] = node_ebd[node_idx, col_idx]
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
            for k in T.Pipelined(T.ceildiv(EDGE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    angle_idx = bx * BLK_M + i
                    col_idx = k * BLK_K + j
                    if angle_idx < N_ANGLE and col_idx < EDGE_DIM:
                        edge_idx = eik2a_index[angle_idx]
                        A[i, j] = flat_edge_ebd[edge_idx, col_idx]
                    else:
                        A[i, j] = T.float32(0)
                for i, j in T.Parallel(BLK_K, BLK_N):
                    row_idx = k * BLK_K + i
                    col_idx = by * BLK_N + j
                    if row_idx < EDGE_DIM and col_idx < OUT_DIM:
                        B[i, j] = edge_ik_weight[row_idx, col_idx]
                    else:
                        B[i, j] = T.float32(0)
                T.gemm(A, B, acc)
            for k in T.Pipelined(T.ceildiv(EDGE_DIM, BLK_K), num_stages=2):
                for i, j in T.Parallel(BLK_M, BLK_K):
                    angle_idx = bx * BLK_M + i
                    col_idx = k * BLK_K + j
                    if angle_idx < N_ANGLE and col_idx < EDGE_DIM:
                        edge_idx = eij2a_index[angle_idx]
                        A[i, j] = flat_edge_ebd[edge_idx, col_idx]
                    else:
                        A[i, j] = T.float32(0)
                for i, j in T.Parallel(BLK_K, BLK_N):
                    row_idx = k * BLK_K + i
                    col_idx = by * BLK_N + j
                    if row_idx < EDGE_DIM and col_idx < OUT_DIM:
                        B[i, j] = edge_ij_weight[row_idx, col_idx]
                    else:
                        B[i, j] = T.float32(0)
                T.gemm(A, B, acc)
            for i, j in T.Parallel(BLK_M, BLK_N):
                if bx * BLK_M + i < N_ANGLE and by * BLK_N + j < OUT_DIM:
                    out[bx * BLK_M + i, by * BLK_N + j] = T.cast(acc[i, j], 'float32') + bias[by * BLK_N + j]
    return fused_angle_update

class FusedAngleUpdateFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, flat_angle_ebd: torch.Tensor, node_ebd: torch.Tensor, flat_edge_ebd: torch.Tensor, n2a_index: torch.Tensor, eij2a_index: torch.Tensor, eik2a_index: torch.Tensor, sub_angle: torch.Tensor, sub_node: torch.Tensor, sub_edge_ik: torch.Tensor, sub_edge_ij: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
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
        kernel = fused_angle_update_forward(N_ANGLE=n_angle, N_NODE=n_node, N_EDGE=n_edge, ANGLE_DIM=angle_dim, NODE_DIM=node_dim, EDGE_DIM=edge_dim, OUT_DIM=out_dim)
        kernel(flat_angle_ebd, flat_node_ebd, flat_edge_ebd, n2a_index, eij2a_index, eik2a_index, sub_angle, sub_node, sub_edge_ik, sub_edge_ij, bias, result_update)
        ctx.save_for_backward(flat_angle_ebd, flat_node_ebd, flat_edge_ebd, n2a_index, eij2a_index, eik2a_index, sub_angle, sub_node, sub_edge_ik, sub_edge_ij)
        ctx.node_ebd_shape = node_ebd.shape
        return result_update

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_angle_ebd, flat_node_ebd, flat_edge_ebd, n2a_index, eij2a_index, eik2a_index, sub_angle, sub_node, sub_edge_ik, sub_edge_ij = ctx.saved_tensors
        node_ebd_shape = ctx.node_ebd_shape
        output = FusedAngleUpdateFunctionBackward.apply(grad_output, flat_angle_ebd, flat_node_ebd, flat_edge_ebd, n2a_index, eij2a_index, eik2a_index, sub_angle, sub_node, sub_edge_ik, sub_edge_ij, node_ebd_shape)
        grad_flat_angle_ebd, grad_node_ebd, grad_flat_edge_ebd, grad_sub_angle, grad_sub_node, grad_sub_edge_ik, grad_sub_edge_ij, grad_bias = output
        return (grad_flat_angle_ebd, grad_node_ebd, grad_flat_edge_ebd, None, None, None, grad_sub_angle, grad_sub_node, grad_sub_edge_ik, grad_sub_edge_ij, grad_bias)

@tilelang.jit
def fused_angle_update_backward_inputs(M, K, A, N, EK, N_NODE, N_EDGE, dtype='float32', accum_dtype='float32', THREADS=128, BLOCK_M=32, BLOCK_D=16, BLOCK_K=32):
    MAX_D = max(A, N, EK)

    @T.prim_func
    def backward_inputs(grad_output: T.Tensor((M, K), dtype), n2a_index: T.Tensor((M,), 'int64'), eij2a_index: T.Tensor((M,), 'int64'), eik2a_index: T.Tensor((M,), 'int64'), sub_angle: T.Tensor((A, K), dtype), sub_node: T.Tensor((N, K), dtype), sub_edge_ik: T.Tensor((EK, K), dtype), sub_edge_ij: T.Tensor((EK, K), dtype), grad_flat_angle: T.Tensor((M, A), accum_dtype), grad_flat_node: T.Tensor((N_NODE, N), accum_dtype), grad_flat_edge: T.Tensor((N_EDGE, EK), accum_dtype), grad_bias: T.Tensor((K,), accum_dtype)):
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
    return backward_inputs

@tilelang.jit
def fused_angle_update_backward_weights(M, K, A, N, EK, N_NODE, N_EDGE, dtype='float32', accum_dtype='float32', BLOCK_D=32, BLOCK_K=16, THREADS=128, BLOCK_M=32):
    MAX_D = max(A, N, EK)

    @T.prim_func
    def backward_weights(grad_output: T.Tensor((M, K), dtype), flat_angle_ebd: T.Tensor((M, A), dtype), flat_node_ebd: T.Tensor((N_NODE, N), dtype), flat_edge_ebd: T.Tensor((N_EDGE, EK), dtype), n2a_index: T.Tensor((M,), 'int64'), eij2a_index: T.Tensor((M,), 'int64'), eik2a_index: T.Tensor((M,), 'int64'), grad_sub_angle: T.Tensor((A, K), accum_dtype), grad_sub_node: T.Tensor((N, K), accum_dtype), grad_sub_edge_ik: T.Tensor((EK, K), accum_dtype), grad_sub_edge_ij: T.Tensor((EK, K), accum_dtype)):
        with T.Kernel(T.ceildiv(MAX_D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=THREADS) as (bx, by):
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            acc_angle = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc_angle)
            acc_node = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc_node)
            acc_ik = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc_ik)
            acc_ij = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
            T.clear(acc_ij)
            for mo in T.Pipelined(T.ceildiv(M, BLOCK_M), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = mo * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    d = bx * BLOCK_D + di
                    m = mo * BLOCK_M + mi
                    if d < A and m < M:
                        feature_shared[mi, di] = flat_angle_ebd[m, d]
                    else:
                        feature_shared[mi, di] = 0
                T.sync_threads()
                if bx * BLOCK_D < A:
                    T.gemm(feature_shared, grad_shared, acc_angle, transpose_A=True)
                T.sync_threads()
            T.sync_threads()
            for mo in T.Pipelined(T.ceildiv(M, BLOCK_M), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = mo * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    d = bx * BLOCK_D + di
                    m = mo * BLOCK_M + mi
                    if d < N and m < M:
                        feature_shared[mi, di] = flat_node_ebd[n2a_index[m], d]
                    else:
                        feature_shared[mi, di] = 0
                T.sync_threads()
                if bx * BLOCK_D < N:
                    T.gemm(feature_shared, grad_shared, acc_node, transpose_A=True)
                T.sync_threads()
            T.sync_threads()
            for mo in T.Pipelined(T.ceildiv(M, BLOCK_M), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = mo * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    d = bx * BLOCK_D + di
                    m = mo * BLOCK_M + mi
                    if d < EK and m < M:
                        feature_shared[mi, di] = flat_edge_ebd[eik2a_index[m], d]
                    else:
                        feature_shared[mi, di] = 0
                T.sync_threads()
                if bx * BLOCK_D < EK:
                    T.gemm(feature_shared, grad_shared, acc_ik, transpose_A=True)
                T.sync_threads()
            T.sync_threads()
            for mo in T.Pipelined(T.ceildiv(M, BLOCK_M), num_stages=2):
                for mi, ki in T.Parallel(BLOCK_M, BLOCK_K):
                    m = mo * BLOCK_M + mi
                    k = by * BLOCK_K + ki
                    if m < M and k < K:
                        grad_shared[mi, ki] = grad_output[m, k]
                    else:
                        grad_shared[mi, ki] = 0
                T.sync_threads()
                for mi, di in T.Parallel(BLOCK_M, BLOCK_D):
                    d = bx * BLOCK_D + di
                    m = mo * BLOCK_M + mi
                    if d < EK and m < M:
                        feature_shared[mi, di] = flat_edge_ebd[eij2a_index[m], d]
                    else:
                        feature_shared[mi, di] = 0
                T.sync_threads()
                if bx * BLOCK_D < EK:
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
def fused_angle_update_double_backward_inputs(M, K, A, N, EK, N_NODE, N_EDGE, dtype='float32', accum_dtype='float32', THREADS=128, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32):

    @T.prim_func
    def double_backward_inputs(grad_output: T.Tensor((M, K), dtype), flat_angle_ebd: T.Tensor((M, A), dtype), flat_node_ebd: T.Tensor((N_NODE, N), dtype), flat_edge_ebd: T.Tensor((N_EDGE, EK), dtype), n2a_index: T.Tensor((M,), 'int64'), eij2a_index: T.Tensor((M,), 'int64'), eik2a_index: T.Tensor((M,), 'int64'), sub_angle: T.Tensor((A, K), dtype), sub_node: T.Tensor((N, K), dtype), sub_edge_ik: T.Tensor((EK, K), dtype), sub_edge_ij: T.Tensor((EK, K), dtype), gg_flat_angle: T.Tensor((M, A), dtype), gg_flat_node: T.Tensor((N_NODE, N), dtype), gg_flat_edge: T.Tensor((N_EDGE, EK), dtype), gg_sub_angle: T.Tensor((A, K), dtype), gg_sub_node: T.Tensor((N, K), dtype), gg_sub_edge_ik: T.Tensor((EK, K), dtype), gg_sub_edge_ij: T.Tensor((EK, K), dtype), gg_bias: T.Tensor((K,), dtype), grad_grad_output: T.Tensor((M, K), accum_dtype), grad_flat_angle: T.Tensor((M, A), accum_dtype), grad_flat_node: T.Tensor((N_NODE, N), accum_dtype), grad_flat_edge: T.Tensor((N_EDGE, EK), accum_dtype)):
        with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(max(K, A, N, EK, EK), BLOCK_N), threads=THREADS) as (bx, by):
            lhs_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            rhs_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
            acc_output = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            acc_input = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            T.clear(acc_output)
            if by * BLOCK_N < K:
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
                    grad_grad_output[m, n] = acc_output[mi, ni] + gg_bias[n]
            if by * BLOCK_N < A:
                T.clear(acc_input)
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
                            rhs_shared[ri, ni] = gg_sub_angle[n, r]
                        else:
                            rhs_shared[ri, ni] = 0
                    T.sync_threads()
                    T.gemm(lhs_shared, rhs_shared, acc_input)
                    T.sync_threads()
                T.sync_threads()
                for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                    m = bx * BLOCK_M + mi
                    n = by * BLOCK_N + ni
                    if m < M and n < A:
                        grad_flat_angle[m, n] = acc_input[mi, ni]
                T.sync_threads()
            if by * BLOCK_N < N:
                T.clear(acc_input)
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
                            rhs_shared[ri, ni] = gg_sub_node[n, r]
                        else:
                            rhs_shared[ri, ni] = 0
                    T.sync_threads()
                    T.gemm(lhs_shared, rhs_shared, acc_input)
                    T.sync_threads()
                T.sync_threads()
                for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                    m = bx * BLOCK_M + mi
                    n = by * BLOCK_N + ni
                    if m < M and n < N:
                        T.atomic_add(grad_flat_node[n2a_index[m], n], acc_input[mi, ni])
                T.sync_threads()
            if by * BLOCK_N < EK:
                T.clear(acc_input)
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
                            rhs_shared[ri, ni] = gg_sub_edge_ik[n, r]
                        else:
                            rhs_shared[ri, ni] = 0
                    T.sync_threads()
                    T.gemm(lhs_shared, rhs_shared, acc_input)
                    T.sync_threads()
                T.sync_threads()
                for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                    m = bx * BLOCK_M + mi
                    n = by * BLOCK_N + ni
                    if m < M and n < EK:
                        T.atomic_add(grad_flat_edge[eik2a_index[m], n], acc_input[mi, ni])
                T.sync_threads()
            if by * BLOCK_N < EK:
                T.clear(acc_input)
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
                            rhs_shared[ri, ni] = gg_sub_edge_ij[n, r]
                        else:
                            rhs_shared[ri, ni] = 0
                    T.sync_threads()
                    T.gemm(lhs_shared, rhs_shared, acc_input)
                    T.sync_threads()
                T.sync_threads()
                for mi, ni in T.Parallel(BLOCK_M, BLOCK_N):
                    m = bx * BLOCK_M + mi
                    n = by * BLOCK_N + ni
                    if m < M and n < EK:
                        T.atomic_add(grad_flat_edge[eij2a_index[m], n], acc_input[mi, ni])
                T.sync_threads()
    return double_backward_inputs

@tilelang.jit
def fused_angle_update_double_backward_weights(M, K, A, N, EK, N_NODE, N_EDGE, dtype='float32', accum_dtype='float32', BLOCK_D=32, BLOCK_K=16, THREADS=128, BLOCK_M=32):
    MAX_D = max(A, N, EK)

    @T.prim_func
    def double_backward_weights(grad_output: T.Tensor((M, K), dtype), n2a_index: T.Tensor((M,), 'int64'), eij2a_index: T.Tensor((M,), 'int64'), eik2a_index: T.Tensor((M,), 'int64'), gg_flat_angle: T.Tensor((M, A), dtype), gg_flat_node: T.Tensor((N_NODE, N), dtype), gg_flat_edge: T.Tensor((N_EDGE, EK), dtype), grad_sub_angle: T.Tensor((A, K), accum_dtype), grad_sub_node: T.Tensor((N, K), accum_dtype), grad_sub_edge_ik: T.Tensor((EK, K), accum_dtype), grad_sub_edge_ij: T.Tensor((EK, K), accum_dtype)):
        with T.Kernel(T.ceildiv(MAX_D, BLOCK_D), T.ceildiv(K, BLOCK_K), threads=THREADS) as (bx, by):
            feature_shared = T.alloc_shared((BLOCK_M, BLOCK_D), dtype)
            grad_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            acc = T.alloc_fragment((BLOCK_D, BLOCK_K), accum_dtype)
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
    def forward(ctx, grad_output: torch.Tensor, flat_angle_ebd: torch.Tensor, flat_node_ebd: torch.Tensor, flat_edge_ebd: torch.Tensor, n2a_index: torch.Tensor, eij2a_index: torch.Tensor, eik2a_index: torch.Tensor, sub_angle: torch.Tensor, sub_node: torch.Tensor, sub_edge_ik: torch.Tensor, sub_edge_ij: torch.Tensor, node_ebd_shape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_output = grad_output.contiguous()
        M, K = grad_output.shape
        _, A = flat_angle_ebd.shape
        N_NODE, N = flat_node_ebd.shape
        N_EDGE, EK = flat_edge_ebd.shape
        dtype = str(grad_output.dtype).replace('torch.', '')
        grad_flat_angle_ebd = torch.empty_like(flat_angle_ebd)
        grad_flat_node_ebd = torch.zeros_like(flat_node_ebd)
        grad_flat_edge_ebd = torch.zeros_like(flat_edge_ebd)
        grad_bias = torch.zeros((K,), device=grad_output.device, dtype=grad_output.dtype)
        input_kernel = fused_angle_update_backward_inputs(M=M, K=K, A=A, N=N, EK=EK, N_NODE=N_NODE, N_EDGE=N_EDGE, dtype=dtype)
        input_kernel(grad_output, n2a_index, eij2a_index, eik2a_index, sub_angle, sub_node, sub_edge_ik, sub_edge_ij, grad_flat_angle_ebd, grad_flat_node_ebd, grad_flat_edge_ebd, grad_bias)
        grad_node_ebd = grad_flat_node_ebd.reshape(node_ebd_shape)
        grad_sub_angle = torch.empty_like(sub_angle)
        grad_sub_node = torch.empty_like(sub_node)
        grad_sub_edge_ik = torch.empty_like(sub_edge_ik)
        grad_sub_edge_ij = torch.empty_like(sub_edge_ij)
        weight_kernel = fused_angle_update_backward_weights(M=M, K=K, A=A, N=N, EK=EK, N_NODE=N_NODE, N_EDGE=N_EDGE, dtype=dtype)
        weight_kernel(grad_output, flat_angle_ebd, flat_node_ebd, flat_edge_ebd, n2a_index, eij2a_index, eik2a_index, grad_sub_angle, grad_sub_node, grad_sub_edge_ik, grad_sub_edge_ij)
        ctx.save_for_backward(grad_output, flat_angle_ebd, flat_node_ebd, flat_edge_ebd, n2a_index, eij2a_index, eik2a_index, sub_angle, sub_node, sub_edge_ik, sub_edge_ij)
        return (grad_flat_angle_ebd, grad_node_ebd, grad_flat_edge_ebd, grad_sub_angle, grad_sub_node, grad_sub_edge_ik, grad_sub_edge_ij, grad_bias)

    @staticmethod
    def backward(ctx, grad_grad_flat_angle_ebd, grad_grad_node_ebd, grad_grad_flat_edge_ebd, grad_grad_sub_angle, grad_grad_sub_node, grad_grad_sub_edge_ik, grad_grad_sub_edge_ij, grad_grad_bias):
        grad_output, flat_angle_ebd, flat_node_ebd, flat_edge_ebd, n2a_index, eij2a_index, eik2a_index, sub_angle, sub_node, sub_edge_ik, sub_edge_ij = ctx.saved_tensors
        M, K = grad_output.shape
        _, A = flat_angle_ebd.shape
        N_NODE, N = flat_node_ebd.shape
        N_EDGE, EK = flat_edge_ebd.shape
        dtype = str(grad_output.dtype).replace('torch.', '')
        grad_grad_flat_angle_ebd = torch.zeros_like(flat_angle_ebd) if grad_grad_flat_angle_ebd is None else grad_grad_flat_angle_ebd.contiguous()
        grad_grad_flat_node_ebd = torch.zeros_like(flat_node_ebd) if grad_grad_node_ebd is None else grad_grad_node_ebd.reshape_as(flat_node_ebd).contiguous()
        grad_grad_flat_edge_ebd = torch.zeros_like(flat_edge_ebd) if grad_grad_flat_edge_ebd is None else grad_grad_flat_edge_ebd.contiguous()
        grad_grad_sub_angle = torch.zeros_like(sub_angle) if grad_grad_sub_angle is None else grad_grad_sub_angle.contiguous()
        grad_grad_sub_node = torch.zeros_like(sub_node) if grad_grad_sub_node is None else grad_grad_sub_node.contiguous()
        grad_grad_sub_edge_ik = torch.zeros_like(sub_edge_ik) if grad_grad_sub_edge_ik is None else grad_grad_sub_edge_ik.contiguous()
        grad_grad_sub_edge_ij = torch.zeros_like(sub_edge_ij) if grad_grad_sub_edge_ij is None else grad_grad_sub_edge_ij.contiguous()
        grad_grad_bias = torch.zeros((K,), device=grad_output.device, dtype=grad_output.dtype) if grad_grad_bias is None else grad_grad_bias.contiguous()
        grad_grad_output = torch.empty_like(grad_output)
        grad_flat_angle_ebd = torch.empty_like(flat_angle_ebd)
        grad_flat_node_ebd = torch.zeros_like(flat_node_ebd)
        grad_flat_edge_ebd = torch.zeros_like(flat_edge_ebd)
        input_kernel = fused_angle_update_double_backward_inputs(M=M, K=K, A=A, N=N, EK=EK, N_NODE=N_NODE, N_EDGE=N_EDGE, dtype=dtype)
        input_kernel(grad_output, flat_angle_ebd, flat_node_ebd, flat_edge_ebd, n2a_index, eij2a_index, eik2a_index, sub_angle, sub_node, sub_edge_ik, sub_edge_ij, grad_grad_flat_angle_ebd, grad_grad_flat_node_ebd, grad_grad_flat_edge_ebd, grad_grad_sub_angle, grad_grad_sub_node, grad_grad_sub_edge_ik, grad_grad_sub_edge_ij, grad_grad_bias, grad_grad_output, grad_flat_angle_ebd, grad_flat_node_ebd, grad_flat_edge_ebd)
        grad_sub_angle = torch.empty_like(sub_angle)
        grad_sub_node = torch.empty_like(sub_node)
        grad_sub_edge_ik = torch.empty_like(sub_edge_ik)
        grad_sub_edge_ij = torch.empty_like(sub_edge_ij)
        weight_kernel = fused_angle_update_double_backward_weights(M=M, K=K, A=A, N=N, EK=EK, N_NODE=N_NODE, N_EDGE=N_EDGE, dtype=dtype)
        weight_kernel(grad_output, n2a_index, eij2a_index, eik2a_index, grad_grad_flat_angle_ebd, grad_grad_flat_node_ebd, grad_grad_flat_edge_ebd, grad_sub_angle, grad_sub_node, grad_sub_edge_ik, grad_sub_edge_ij)
        return (grad_grad_output, grad_flat_angle_ebd, grad_flat_node_ebd, grad_flat_edge_ebd, None, None, None, grad_sub_angle, grad_sub_node, grad_sub_edge_ik, grad_sub_edge_ij, None)
