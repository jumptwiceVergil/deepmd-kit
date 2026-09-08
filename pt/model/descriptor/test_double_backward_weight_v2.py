"""Check weight-side double VJPs; run with --device cuda on the target GPU.

CPU mode interprets kernel indexing and mathematics, not GPU scheduling.
The strict CUDA tolerances intentionally do not hide TF32 rounding errors.
"""
import argparse
import ast
import itertools
from pathlib import Path

import torch

from test_angle_weight_v2 import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    source = Path(__file__).with_name("utils_tilelang.py")
    names = {f"fused_{kind}_update_double_backward_weights_v2{suffix}"
             for kind in ("edge", "angle") for suffix in ("", "_reduce")}
    if args.device == "cpu":
        helper = load_file("double_helpers", source.with_name("test_tilelang_double_backward.py"))
        interpreter = helper.LoopInterpreter()
        interpreter.serial = lambda n: range(n)
        nodes = [n for n in ast.parse(source.read_text(encoding="utf-8")).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        for node in nodes:
            node.decorator_list = []
        namespace = {"T": interpreter}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        dtype = torch.float64
    else:
        if not torch.cuda.is_available():
            parser.error("CUDA is required")
        namespace = vars(load_file("double_weight_v2", source))
        torch.backends.cuda.matmul.allow_tf32 = False
        dtype = torch.float32
    failures = []
    for kind, shape, splits in itertools.product(
            ("edge", "angle"),
            ((7, 9, 3, 5, 4), (35, 65, 17, 33, 19),
             (132, 64, 32, 32, 32)), (1, 3, 8)):
        torch.manual_seed(47)
        m, k, a, n, e = shape
        nn, ne = 5, 11
        def rand(*shape):
            return (torch.randn(shape, dtype=dtype, device=args.device) * .2).requires_grad_()
        ni, ij, ik = (torch.randint(size, (m,), device=args.device) for size in (nn, ne, ne))
        if kind == "edge":
            xs = [rand(nn, n), rand(ne, e), rand(m, a)]
            dims = dict(E=m, K=k, D_edge=a, D_node=n, D_ext=e, N_node=nn, N_ext=ne)
            rdims = dict(K=k, D_edge=a, D_node=n, D_ext=e)
            widths = (n, e, a)
            indices = (ni, ij, None)
        else:
            xs = [rand(m, a), rand(nn, n), rand(ne, e)]
            dims = dict(M=m, K=k, A=a, N=n, EK=e, N_NODE=nn, N_EDGE=ne)
            rdims = dict(K=k, A=a, N=n, EK=e)
            widths = (a, n, e, e)
            indices = (None, ni, ik, ij)
        weights = [rand(d, k) for d in widths]
        features = xs if kind == "edge" else [*xs, xs[2]]
        y = sum((x if ix is None else x[ix]) @ w
                for x, ix, w in zip(features, indices, weights))
        g = rand(m, k)
        ux = [torch.randn_like(x) * .2 for x in xs]
        first = torch.autograd.grad(y, xs, g, create_graph=True, retain_graph=True)
        reference = torch.autograd.grad(first, weights, ux, create_graph=True, retain_graph=True)
        workspace = torch.full((splits, sum(widths), k), float("nan"), dtype=dtype, device=args.device)
        outputs = [torch.full_like(w, float("nan")) for w in weights]
        prefix = f"fused_{kind}_update_double_backward_weights_v2"
        partial = namespace[prefix](**dims, SPLIT_M=splits)
        reduce = namespace[prefix + "_reduce"](**rdims, SPLIT_M=splits)
        pargs = ((g, ni, ij, *ux, workspace) if kind == "edge"
                 else (g, *ux, ni, ij, ik, workspace))
        if args.device == "cuda":
            partial(*pargs)
            reduce(workspace, *outputs)
            torch.cuda.synchronize()
        else:
            grid = ((sum(widths)+31)//32, (k+15)//16)
            for interpreter.grid in itertools.product(range(grid[0]), range(grid[1]), range(splits)):
                partial(*pargs)
            for interpreter.grid in itertools.product(range(grid[0]), range(grid[1])):
                reduce(workspace, *outputs)
        assert torch.isfinite(workspace).all(), "Invalid/unwritten workspace"
        for i, (actual, expected) in enumerate(zip(outputs, reference)):
            try:
                torch.testing.assert_close(actual, expected,
                                           atol=2e-4 if args.device == "cuda" else 1e-10,
                                           rtol=2e-3 if args.device == "cuda" else 1e-10)
            except AssertionError as error:
                failures.append(f"{kind} {shape} S={splits} output={i}: {error}")
        print(f"Checked {kind} {shape} S={splits}", flush=True)
    if failures:
        raise AssertionError("\n".join(failures))
    print("All 18 checks passed." + (" CPU interpretation only." if args.device == "cpu" else ""))


if __name__ == "__main__":
    main()
