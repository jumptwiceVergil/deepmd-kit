import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def fused_symmetrization_op_dynamic_forward(
    M,  # n_edge
    N,  # 3 * e_dim
    E,  # e_dim
    NO = 12,
    dtype = "float32",
    accum_dtype = "float32",
    BLOCK_N: int = 64,
):
    @T.prim_func
    def kernel(
        flat_edge_ebd: T.Buffer((M, E), dtype),
        flat_sw: T.Buffer((M,), dtype),
        flat_h2: T.Buffer((M, 3), dtype),
        owner: T.Buffer((M,), "int64"),
        num_owner: T.int32,
        scale_factor: T.float32,
        out: T.Buffer((NO, N), accum_dtype),
    ):
        with T.Kernel(num_owner, T.ceildiv(N, BLOCK_N), threads=64) as (bx, by):
            acc = T.alloc_fragment((BLOCK_N,), accum_dtype)

            T.clear(acc)

            for j in T.Parallel(BLOCK_N):
                col = by * BLOCK_N + j
                if col < N:
                    # col 映射到 [3, E]
                    h2_idx = col // E
                    e_idx = col % E

                    value = 0.0
                    for r in T.serial(NO):
                        edge_idx = bx * NO + r

                        value += (
                            flat_edge_ebd[edge_idx, e_idx]
                            * flat_sw[edge_idx]
                            * flat_h2[edge_idx, h2_idx]
                        )

                    acc[j] = value * scale_factor

            for j in T.Parallel(BLOCK_N):
                col = by * BLOCK_N + j
                if col < N:
                    out[bx, col] = acc[j]

class FusedSymmetrizationOpDynamic:
    @staticmethod
    def forward(
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
            dtype=flat_edge_ebd.device,
        )

        sym_kernel = fused_symmetrization_op_dynamic_forward(n_edge, 3*e_dim, e_dim)
        sym_kernel(flat_edge_ebd, flat_sw, flat_h2, owner, num_owner, scale_factor, h2g2)
        h2g2 = h2g2.reshape(nb, nloc, 3, e_dim)

        # nb x nloc x 3 x e_dim
        nb, nloc, _, e_dim = h2g2.shape
        # nb x nloc x 3 x axis
        h2g2m = h2g2[..., :axis_neuron]
        # nb x nloc x axis x e_dim
        g1_13 = torch.matmul(torch.transpose(h2g2m, -1, -2), h2g2) / (3.0**1)
        # nb x nloc x (axis x e_dim)
        grrg = g1_13.view(nb, nloc, axis_neuron * e_dim)
        return grrg
    
    @staticmethod
    def backward:
        return FusedSymmetrizationOpDynamicBackward.apply()

class FusedSymmetrizationOpDynamicBackward:
    @staticmethod
    def forward:
        pass

    @staticmethod
    def backward:
        pass
