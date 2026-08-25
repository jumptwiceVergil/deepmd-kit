import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def test_write_kernel(
    NB,
    NLOC,
    E,
    AXIS,
    dtype="float32",
    accum_dtype="float32",
    BLOCK_N=32,
):
    @T.prim_func
    def kernel(
        out: T.Buffer((NB, NLOC, AXIS * E), accum_dtype),
    ):
        with T.Kernel(
            NB,
            NLOC,
            AXIS,
            threads=BLOCK_N,
        ) as (bx, by, bz):

            for e in T.Parallel(BLOCK_N):
                e_idx = e

                if e_idx < E:
                    out[
                        bx,
                        by,
                        bz * E + e_idx,
                    ] = 0.0

    return kernel


NB = 1
NLOC = 12
E = 128
AXIS = 4

out = torch.empty(
    (NB, NLOC, AXIS * E),
    device="cuda",
    dtype=torch.float32,
)

kernel = test_write_kernel(
    NB=NB,
    NLOC=NLOC,
    E=E,
    AXIS=AXIS,
)

print("kernel compiled")

kernel(out)

torch.cuda.synchronize()

print("kernel executed")
print(out.shape)
print(out.abs().max())
