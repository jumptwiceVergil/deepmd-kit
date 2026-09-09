"""Optional-cotangent double backward checks, with NaN-poisoned absent inputs.

python test_tilelang_double_backward_optional.py --device cuda
python test_tilelang_double_backward_optional.py --device cpu --exhaustive

CPU mode interprets the actual kernel loops, testing formulas/indexing only.
CUDA mode compiles and executes the actual TileLang kernels in float32.
"""
import argparse
import ast
import contextlib
import importlib.util
import itertools
from types import SimpleNamespace
from pathlib import Path

import torch


class LoopInterpreter:
    grid = ()
    Tensor = staticmethod(lambda *args: None)
    prim_func = staticmethod(lambda f: f)
    ceildiv = staticmethod(lambda a, b: (a + b - 1) // b)
    Parallel = staticmethod(lambda *sizes: itertools.product(*(range(x) for x in sizes)))
    Pipelined = staticmethod(lambda n, **kwargs: range(n))
    alloc_shared = staticmethod(lambda shape, dtype: torch.zeros(shape, dtype=torch.float64))
    alloc_fragment = alloc_shared
    clear = staticmethod(lambda a: a.zero_())
    sync_threads = staticmethod(lambda: None)
    serial = staticmethod(lambda n: range(n))
    atomic_add = staticmethod(lambda a, b: a.add_(b))
    gemm = staticmethod(lambda a, b, c, transpose_A=False:
                        c.add_((a.T if transpose_A else a) @ b))

    @contextlib.contextmanager
    def Kernel(self, *dims, **kwargs):
        yield self.grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--exhaustive", action="store_true", help="Test every presence mask on the small shape")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA-enabled PyTorch and a GPU are required")
    source = Path(__file__).with_name("utils_tilelang.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {f"fused_{kind}_update_double_backward_{side}"
             for kind in ("edge", "angle") for side in ("inputs", "weights_v2", "weights_v2_reduce")}
    definitions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name in ("FusedEdgeUpdateFunctionBackward", "FusedAngleUpdateFunctionBackward")}
    interpreter = LoopInterpreter()
    if args.device == "cpu":
        for node in definitions.values():
            node.decorator_list = []
        namespace = {"T": interpreter, "torch": torch}
        exec(compile(ast.Module(body=[*definitions.values(), *classes.values()], type_ignores=[]), str(source), "exec"), namespace)
    else:
        spec = importlib.util.spec_from_file_location("double_backward_kernels", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace = vars(module)
        torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.float64 if args.device == "cpu" else torch.float32

    cases = []
    for kind in ("edge", "angle"):
        count = 7 if kind == "edge" else 8
        masks = range(1 << count) if args.exhaustive else [0, (1 << count)-1, *[1 << i for i in range(count)], 5, 42]
        shapes = ((7, 9, 3, 5, 4),) if args.exhaustive else ((7, 9, 3, 5, 4), (35, 65, 17, 33, 19))
        cases.extend((kind, shape, mask) for shape in shapes for mask in masks)
    for kind, shape, mask in cases:
        only_input_cotangents = False
        torch.manual_seed(47)
        rows, k, da, dn, de = shape
        nn, ne = 5, 11
        def rand(*shape):
            return (torch.randn(shape, device=args.device, dtype=dtype) * .2).requires_grad_()
        def index(n):
            return torch.randint(n, (rows,), device=args.device)
        tensors = {}
        if kind == "edge":
            dims = dict(E=rows, K=k, D_edge=da, D_node=dn, D_ext=de, N_node=nn, N_ext=ne)
            tensors.update(n2e_index=index(nn), n_ext2e_index=index(ne))
            # X, Ux, W, Uw, output input gradient, output weight gradient, index, shape
            groups = [
                ("flat_edge_ebd", "grad_grad_edge_ebd", "edge_weight", "grad_grad_edge_weight", "grad_flat_edge_ebd", "grad_edge_weight", None, (rows, da)),
                ("node_ebd", "grad_grad_node", "node_weight", "grad_grad_node_weight", "grad_node_ebd", "grad_node_weight", "n2e_index", (nn, dn)),
                ("node_ebd_ext", "grad_grad_node_ext", "node_ext_weight", "grad_grad_node_ext_weight", "grad_node_ebd_ext", "grad_node_ext_weight", "n_ext2e_index", (ne, de)),
            ]
            gname, uname, oname = "grad_out", "grad_grad_bias", "grad_grad_out"
        else:
            dims = dict(M=rows, K=k, A=da, N=dn, EK=de, N_NODE=nn, N_EDGE=ne)
            tensors.update(n2a_index=index(nn), eik2a_index=index(ne), eij2a_index=index(ne))
            groups = [
                ("flat_angle_ebd", "gg_flat_angle", "sub_angle", "gg_sub_angle", "grad_flat_angle", "grad_sub_angle", None, (rows, da)),
                ("flat_node_ebd", "gg_flat_node", "sub_node", "gg_sub_node", "grad_flat_node", "grad_sub_node", "n2a_index", (nn, dn)),
                ("flat_edge_ebd", "gg_flat_edge", "sub_edge_ik", "gg_sub_edge_ik", "grad_flat_edge", "grad_sub_edge_ik", "eik2a_index", (ne, de)),
                ("flat_edge_ebd", "gg_flat_edge", "sub_edge_ij", "gg_sub_edge_ij", "grad_flat_edge", "grad_sub_edge_ij", "eij2a_index", (ne, de)),
            ]
            gname, uname, oname = "grad_output", "gg_bias", "grad_grad_output"
        targets, vectors, outputs = {}, {}, {}
        bias = rand(k)
        y = bias.expand(rows, k)
        for x, ux, w, uw, gx, gw, ix, xs in groups:
            if x not in tensors:
                tensors[x], tensors[ux] = rand(*xs), rand(*xs).detach()
                targets[x], vectors[x], outputs[x] = tensors[x], tensors[ux], gx
                tensors[gx] = torch.zeros_like(tensors[x])
            tensors[w] = rand(xs[1], k)
            tensors[uw] = torch.zeros_like(tensors[w]) if only_input_cotangents else rand(xs[1], k).detach()
            targets[w], vectors[w], outputs[w] = tensors[w], tensors[uw], gw
            tensors[gw] = torch.zeros_like(tensors[w])
            gathered = tensors[x] if ix is None else tensors[x][tensors[ix]]
            y = y + gathered @ tensors[w]
        tensors[gname] = rand(rows, k)
        tensors[uname] = torch.zeros_like(bias) if only_input_cotangents else rand(k).detach()
        tensors[oname] = torch.zeros_like(y)

        flag_map = {"edge":{"grad_grad_node":"HAS_NODE","grad_grad_node_ext":"HAS_EXT","grad_grad_edge_ebd":"HAS_EDGE","grad_grad_node_weight":"HAS_NODE_WEIGHT","grad_grad_node_ext_weight":"HAS_EXT_WEIGHT","grad_grad_edge_weight":"HAS_EDGE_WEIGHT","grad_grad_bias":"HAS_BIAS"},"angle":{"gg_flat_angle":"HAS_ANGLE","gg_flat_node":"HAS_NODE","gg_flat_edge":"HAS_EDGE","gg_sub_angle":"HAS_ANGLE_WEIGHT","gg_sub_node":"HAS_NODE_WEIGHT","gg_sub_edge_ik":"HAS_IK_WEIGHT","gg_sub_edge_ij":"HAS_IJ_WEIGHT","gg_bias":"HAS_BIAS"}}
        flags = {flag: bool(mask & (1 << i))
                 for i, flag in enumerate(flag_map[kind].values())}
        # References use actual zeros; device kernel arguments use NaN poison.
        for tensor_name, flag in flag_map[kind].items():
            if not flags[flag]:
                tensors[tensor_name].zero_()

        first = torch.autograd.grad(y, [*targets.values(), bias], tensors[gname], create_graph=True)
        phi = sum((v * u).sum() for v, u in zip(first, [*vectors.values(), tensors[uname]]))
        reference = torch.autograd.grad(phi, [tensors[gname], *targets.values()])

        for tensor_name, flag in flag_map[kind].items():
            if not flags[flag]:
                tensors[tensor_name] = torch.full_like(tensors[tensor_name], float("nan"))
        splits = 3
        total_d = da + dn + (de if kind == "edge" else 2*de)
        tensors["workspace"] = torch.full((splits, total_d, k), float("nan"), dtype=dtype, device=args.device)
        for side in ("inputs", "weights_v2", "weights_v2_reduce"):
            name = f"fused_{kind}_update_double_backward_{side}"
            node = definitions[name]
            inner = next(n for n in node.body if isinstance(n, ast.FunctionDef))
            call_args = [tensors[p.arg] for p in inner.args.args]
            accepted = {p.arg for p in node.args.args}
            kwargs = {key: value for key, value in {**dims, **flags, "SPLIT_M": splits}.items() if key in accepted}
            kernel = namespace[name](**kwargs)
            if args.device == "cuda":
                kernel(*call_args)
            else:
                if side == "inputs":
                    grid = ((rows+31)//32, (max(k, da, dn, de)+31)//32)
                else:
                    grid = ((total_d+31)//32, (k+15)//16)
                    if side == "weights_v2":
                        grid += (splits,)
                with torch.no_grad():
                    for interpreter.grid in itertools.product(*(range(v) for v in grid)):
                        kernel(*call_args)
        assert torch.isfinite(tensors["workspace"]).all()
        for result_name, expected in zip([oname, *outputs.values()], reference):
            try:
                torch.testing.assert_close(tensors[result_name], expected,
                                           atol=2e-4 if args.device == "cuda" else 1e-10,
                                           rtol=2e-3 if args.device == "cuda" else 1e-10)
            except AssertionError:
                print(f"FAIL {kind} {shape} mask={mask} output={result_name}")
                raise
        # Also execute the actual Python backward method with None arguments.
        # This checks dispatch, placeholders, flags, and output ordering.
        clsname = "FusedEdgeUpdateFunctionBackward" if kind == "edge" else "FusedAngleUpdateFunctionBackward"
        backward = next(n for n in classes[clsname].body if isinstance(n, ast.FunctionDef) and n.name == "backward")
        saved_names = [n.id for n in backward.body[0].targets[0].elts]
        ctx = SimpleNamespace(saved_tensors=tuple(tensors[n] for n in saved_names))
        cotangents = {}
        if kind == "edge":
            cotangents = {n: tensors[n] if flags[f] else None for n, f in flag_map[kind].items()}
        else:
            aliases = {
                "grad_grad_flat_angle_ebd": "gg_flat_angle", "grad_grad_node_ebd": "gg_flat_node",
                "grad_grad_flat_edge_ebd": "gg_flat_edge", "grad_grad_sub_angle": "gg_sub_angle",
                "grad_grad_sub_node": "gg_sub_node", "grad_grad_sub_edge_ik": "gg_sub_edge_ik",
                "grad_grad_sub_edge_ij": "gg_sub_edge_ij", "grad_grad_bias": "gg_bias",
            }
            cotangents = {dst: tensors[src] if flags[flag_map[kind][src]] else None
                          for dst, src in aliases.items()}
        factories = {n: namespace[n] for n in names}
        if args.device == "cpu":
            def interpreted_factory(name, factory):
                def make(**kw):
                    kernel = factory(**kw)
                    if name.endswith("_inputs"):
                        grid = ((rows+31)//32, (max(k, da, dn, de)+31)//32)
                    else:
                        grid = ((total_d+31)//32, (k+15)//16)
                        if name.endswith("_v2"):
                            grid += (kw["SPLIT_M"],)
                    def launch(*values):
                        for interpreter.grid in itertools.product(*(range(v) for v in grid)):
                            kernel(*values)
                    return launch
                return make
            for name, factory in factories.items():
                namespace[name] = interpreted_factory(name, factory)
        try:
            with torch.no_grad():
                result = namespace[clsname].backward(
                    ctx, *[cotangents.get(p.arg) for p in backward.args.args[1:]])
        finally:
            namespace.update(factories)
        expected_by_input = dict(zip([gname, *targets], reference))
        for input_name, actual in zip(saved_names, result):
            if input_name in expected_by_input:
                torch.testing.assert_close(actual, expected_by_input[input_name],
                                           atol=2e-4 if args.device == "cuda" else 1e-10,
                                           rtol=2e-3 if args.device == "cuda" else 1e-10)
        print(f"PASS {kind} {shape} mask={mask}", flush=True)


if __name__ == "__main__":
    main()
