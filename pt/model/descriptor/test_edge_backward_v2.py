"""Edge backward v2 and its double backward through the actual autograd class.

python test_edge_backward_v2.py --device cuda
python test_edge_backward_v2.py --device cpu

CPU interprets the kernel bodies; it does not validate TileLang GPU compilation.
Keep test_tilelang_double_backward_optional.py in the same directory.
"""
import argparse
import ast
import importlib.util
import itertools
from pathlib import Path

import torch

from test_tilelang_double_backward_optional import LoopInterpreter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    source = Path(__file__).with_name("utils_tilelang.py")
    names = {
        "fused_edge_update_input_backward_v2",
        "fused_edge_update_weight_backward_v2",
        "fused_edge_update_weight_backward_v2_reduce",
        "fused_edge_update_double_backward_inputs",
        "fused_edge_update_double_backward_weights_v2",
        "fused_edge_update_double_backward_weights_v2_reduce",
    }
    clsname = "FusedEdgeUpdateFunctionBackward"
    if args.device == "cpu":
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name in names | {clsname}]
        for n in nodes:
            n.decorator_list = []
        interpreter = LoopInterpreter()
        interpreter.Parallel = lambda *sizes: (range(sizes[0]) if len(sizes) == 1
                                               else itertools.product(*(range(v) for v in sizes)))
        namespace = {"T": interpreter, "torch": torch}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)

        def wrap(name, factory):
            def make(**kw):
                kernel = factory(**kw)
                if name == "fused_edge_update_input_backward_v2":
                    grid = ((kw["E"]+31)//32, (kw["D_edge"]+kw["D_node"]+kw["D_ext"]+31)//32)
                elif name.endswith("_inputs"):
                    grid = ((kw["E"]+31)//32, (max(kw["K"], kw["D_edge"], kw["D_node"], kw["D_ext"])+31)//32)
                else:
                    grid = ((kw["D_edge"]+kw["D_node"]+kw["D_ext"]+31)//32, (kw["K"]+15)//16)
                    if not name.endswith("_reduce"):
                        grid += (kw["SPLIT_M"],)
                def launch(*values):
                    for interpreter.grid in itertools.product(*(range(v) for v in grid)):
                        kernel(*values)
                return launch
            return make
        for name in names:
            namespace[name] = wrap(name, namespace[name])
        dtype = torch.float64
    else:
        if not torch.cuda.is_available():
            parser.error("CUDA is required")
        spec = importlib.util.spec_from_file_location("edge_backward_v2_kernels", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace = vars(module)
        torch.backends.cuda.matmul.allow_tf32 = False
        dtype = torch.float32

    def check(actual, expected):
        torch.testing.assert_close(actual, expected,
                                   atol=2e-4 if args.device == "cuda" else 1e-10,
                                   rtol=2e-3 if args.device == "cuda" else 1e-10)

    # Non-tile-aligned dimensions plus both training output widths.
    for e, k, de, dn, dx in ((7, 9, 3, 5, 4), (35, 65, 17, 33, 19),
                             (132, 64, 64, 128, 128), (132, 128, 64, 128, 128)):
        torch.manual_seed(93)
        nn, nx = 12, 15
        def rand(*shape):
            return (torch.randn(shape, dtype=dtype, device=args.device) * .2).requires_grad_()
        node, ext, edge = rand(nn, dn), rand(nx, dx), rand(e, de)
        wn, wx, we = rand(dn, k), rand(dx, k), rand(de, k)
        bias, g = rand(k), rand(e, k)
        # Unordered duplicates; trailing rows never receive contributions.
        ni = torch.randint(nn-2, (e,), device=args.device)
        xi = torch.randint(nx-3, (e,), device=args.device)
        y = node[ni] @ wn + ext[xi] @ wx + edge @ we + bias
        targets = (node, ext, edge, wn, wx, we, bias)
        reference = torch.autograd.grad(y, targets, g, create_graph=True, retain_graph=True)
        outputs = namespace[clsname].apply(g, node, ext, edge, ni, xi, wn, wx, we)
        actual = tuple(outputs[i] for i in (0, 1, 2, 5, 6, 7, 8))
        for i, (a, r) in enumerate(zip(actual, reference)):
            try:
                check(a, r)
            except AssertionError:
                print(f"FAIL first gradient output={i}, shape={e,k,de,dn,dx}", flush=True)
                raise
        # Bias must not carry one K tile's accumulation into the next.
        check(actual[-1], g.sum(0))
        assert torch.count_nonzero(actual[0][-2:]) == 0
        assert torch.count_nonzero(actual[1][-3:]) == 0
        second_targets = (g, node, ext, edge, wn, wx, we)
        for label, selected in (("all", range(7)), ("inputs", range(3)),
                                ("weights", range(3, 6)), ("bias", (6,))):
            selected = tuple(selected)
            vectors = tuple(torch.randn_like(actual[i]) * .2 for i in selected)
            expected = torch.autograd.grad(
                tuple(reference[i] for i in selected), second_targets, vectors,
                retain_graph=True, allow_unused=True,
            )
            got = torch.autograd.grad(
                tuple(actual[i] for i in selected), second_targets, vectors,
                retain_graph=True, allow_unused=True,
            )
            for i, (a, r, target) in enumerate(zip(got, expected, second_targets)):
                if a is None:
                    a = torch.zeros_like(target)
                if r is None:
                    r = torch.zeros_like(target)
                try:
                    check(a, r)
                except AssertionError:
                    print(f"FAIL double gradient output={i}, mode={label}, shape={e,k,de,dn,dx}", flush=True)
                    raise
        print(f"PASS first/double backward {e,k,de,dn,dx}", flush=True)
    print("All checks passed." + (" CPU interpretation only." if args.device == "cpu" else ""))


if __name__ == "__main__":
    main()
