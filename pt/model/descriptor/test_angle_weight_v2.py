"""Validate both v2 kernels: python test_angle_weight_v2.py --device cuda.

CPU mode interprets the actual loop bodies and does not validate CUDA scheduling.
"""
import argparse
import ast
import importlib.util
import itertools
from pathlib import Path

import torch


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    source = Path(__file__).with_name("utils_tilelang.py")
    names = ("fused_angle_update_backward_weights_v2", "fused_angle_update_backward_weights_v2_reduce")
    if args.device == "cpu":
        helper = load_file("double_backward_test_helpers", source.with_name("test_tilelang_double_backward.py"))
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
        namespace = vars(load_file("angle_weight_v2_kernels", source))
        torch.backends.cuda.matmul.allow_tf32 = False
        dtype = torch.float32
    failures = []
    for (m, k, a, n, e), splits in itertools.product(
            ((7, 9, 3, 5, 4), (35, 65, 17, 33, 19), (132, 64, 32, 32, 32)), (1, 3, 8)):
        torch.manual_seed(23)
        nn, ne = 5, 11
        def rand(*shape):
            return torch.randn(shape, dtype=dtype, device=args.device) * .2
        angle, node, edge, g = rand(m, a), rand(nn, n), rand(ne, e), rand(m, k)
        ni, ij, ik = (torch.randint(size, (m,), device=args.device) for size in (nn, ne, ne))
        weights = [rand(d, k).requires_grad_() for d in (a, n, e, e)]
        y = angle @ weights[0] + node[ni] @ weights[1] + edge[ik] @ weights[2] + edge[ij] @ weights[3]
        reference = torch.autograd.grad(y, weights, g, create_graph=True, retain_graph=True)
        workspace = torch.full((splits, a+n+2*e, k), float("nan"), device=args.device, dtype=dtype)
        outputs = [torch.full_like(w, float("nan")) for w in weights]
        partial = namespace[names[0]](M=m, K=k, A=a, N=n, EK=e, N_NODE=nn, N_EDGE=ne, SPLIT_M=splits)
        reduce = namespace[names[1]](K=k, A=a, N=n, EK=e, SPLIT_M=splits)
        partial_args = (g, angle, node, edge, ni, ij, ik, workspace)
        if args.device == "cuda":
            partial(*partial_args)
            reduce(workspace, *outputs)
            torch.cuda.synchronize()
        else:
            grid = ((a+n+2*e+31)//32, (k+15)//16)
            for interpreter.grid in itertools.product(range(grid[0]), range(grid[1]), range(splits)):
                partial(*partial_args)
            for interpreter.grid in itertools.product(range(grid[0]), range(grid[1])):
                reduce(workspace, *outputs)
        assert torch.isfinite(workspace).all(), "Unwritten or invalid workspace values"
        for label, actual, expected in zip(("angle", "node", "ik", "ij"), outputs, reference):
            try:
                torch.testing.assert_close(actual, expected, atol=2e-4 if args.device == "cuda" else 1e-10,
                                           rtol=2e-3 if args.device == "cuda" else 1e-10)
            except AssertionError as error:
                failures.append(f"{m,k,a,n,e} S={splits} {label}: {error}")
        print(f"Checked {m,k,a,n,e} SPLIT_M={splits}", flush=True)
    if failures:
        raise AssertionError("\n".join(failures))
    print("All checks passed." + (" CPU interpretation only." if args.device == "cpu" else ""))


if __name__ == "__main__":
    main()
