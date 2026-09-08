"""Standalone double-backward kernel check.

python test_tilelang_double_backward.py --device cuda
python test_tilelang_double_backward.py --device cpu

CPU mode interprets the actual kernel loops, testing formulas/indexing only.
CUDA mode compiles and executes the actual TileLang kernels in float32.
"""
import argparse
import ast
import contextlib
import importlib.util
import itertools
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
    atomic_add = staticmethod(lambda a, b: a.add_(b))
    gemm = staticmethod(lambda a, b, c, transpose_A=False:
                        c.add_((a.T if transpose_A else a) @ b))

    @contextlib.contextmanager
    def Kernel(self, *dims, **kwargs):
        yield self.grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dump-dir", type=Path,
                        help="Save generated kernel source and failing tensors for diagnosis")
    args = parser.parse_args()
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA-enabled PyTorch and a GPU are required")
    source = Path(__file__).with_name("utils_tilelang.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {f"fused_{kind}_update_double_backward_{side}"
             for kind in ("edge", "angle") for side in ("inputs", "weights")}
    definitions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}
    interpreter = LoopInterpreter()
    if args.device == "cpu":
        for node in definitions.values():
            node.decorator_list = []
        namespace = {"T": interpreter}
        exec(compile(ast.Module(body=list(definitions.values()), type_ignores=[]), str(source), "exec"), namespace)
    else:
        spec = importlib.util.spec_from_file_location("double_backward_kernels", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace = vars(module)
        torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.float64 if args.device == "cpu" else torch.float32

    for kind, shape, only_input_cotangents in itertools.product(
            ("edge", "angle"), ((7, 9, 3, 5, 4), (35, 65, 17, 33, 19), (132, 64, 32, 32, 32)), (False, True)):
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
        first = torch.autograd.grad(y, [*targets.values(), bias], tensors[gname], create_graph=True)
        phi = sum((v * u).sum() for v, u in zip(first, [*vectors.values(), tensors[uname]]))
        reference = torch.autograd.grad(phi, [tensors[gname], *targets.values()])
        for side in ("inputs", "weights"):
            name = f"fused_{kind}_update_double_backward_{side}"
            node = definitions[name]
            inner = next(n for n in node.body if isinstance(n, ast.FunctionDef))
            call_args = [tensors[p.arg] for p in inner.args.args]
            kernel = namespace[name](**dims)
            if args.device == "cuda":
                if args.dump_dir:
                    get_source = getattr(kernel, "get_kernel_source", None)
                    if get_source is not None:
                        tag = f"{kind}_{side}_{rows}_{k}_{da}_{dn}_{de}"
                        (args.dump_dir / f"{tag}.cu").write_text(get_source(), encoding="utf-8")
                kernel(*call_args)
            else:
                defaults = {p.arg: ast.literal_eval(v) for p, v in zip(node.args.args[-len(node.args.defaults):], node.args.defaults)}
                if side == "inputs":
                    grid = (interpreter.ceildiv(rows, defaults["BLOCK_M"]), interpreter.ceildiv(max(k, da, dn, de), defaults["BLOCK_N"]))
                else:
                    grid = (interpreter.ceildiv(max(da, dn, de), defaults["BLOCK_D"]), interpreter.ceildiv(k, defaults["BLOCK_K"]))
                with torch.no_grad():
                    for interpreter.grid in itertools.product(*(range(v) for v in grid)):
                        kernel(*call_args)
        for result_name, expected in zip([oname, *outputs.values()], reference):
            try:
                torch.testing.assert_close(tensors[result_name], expected,
                                           atol=2e-4 if args.device == "cuda" else 1e-10,
                                           rtol=2e-3 if args.device == "cuda" else 1e-10)
            except AssertionError:
                actual_cpu = tensors[result_name].detach().cpu()
                expected_cpu = expected.detach().cpu()
                print(f"FAIL {kind} {shape} tensor={result_name}", flush=True)
                if args.dump_dir:
                    torch.save({"tensors": {n: t.detach().cpu() for n, t in tensors.items()},
                                "failed_output": result_name, "expected": expected_cpu, "dims": dims},
                               args.dump_dir / f"failure_{kind}_{rows}_{k}.pt")
                if result_name == oname and dtype == torch.float32:
                    # Diagnostic only: model TF32 operand rounding, not GPU
                    # scheduling or exact tensor-core accumulation order.
                    def rounded(value, mode):
                        bits = value.detach().cpu().contiguous().view(torch.int32)
                        if mode == "rn":
                            bits = bits + 4095 + ((bits >> 13) & 1)
                        return (bits & -8192).view(torch.float32).double()
                    for mode in ("rn", "rz"):
                        approx = tensors[uname].detach().cpu().double().expand(rows, k).clone()
                        for x, ux, w, uw, gx, gw, ix, xs in groups:
                            xx, uu = rounded(tensors[x], mode), rounded(tensors[ux], mode)
                            if ix is not None:
                                ii = tensors[ix].cpu()
                                xx, uu = xx[ii], uu[ii]
                            approx += uu @ rounded(tensors[w], mode) + xx @ rounded(tensors[uw], mode)
                        error = (actual_cpu.double() - approx).abs()
                        print(f"  TF32-{mode} model: max_abs={error.max().item():.6g}, "
                              f"relative_l2={(error.norm()/approx.norm().clamp_min(1e-30)).item():.6g}", flush=True)
                raise
        print(f"PASS {kind} {shape} input_cotangents_only={only_input_cotangents}", flush=True)
    print("CPU interpretation only; CUDA scheduling is not verified." if args.device == "cpu" else "CUDA kernel checks passed.")


if __name__ == "__main__":
    main()
