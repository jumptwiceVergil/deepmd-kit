"""Run directly: python pt/model/descriptor/test_symmetrization_owner.py -v.

Metadata tests run on CPU without TileLang. Numerical kernel tests require CUDA
and TileLang and compare forward, VJP, and double VJP with native PyTorch.
"""
import ast
import importlib.util
from itertools import product
from pathlib import Path
import unittest

import torch


SOURCE = Path(__file__).with_name("utils_tilelang.py")


def metadata_namespace():
    # Load the actual metadata implementation without importing the CUDA DSL.
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {"_sym_owner_cache", "_SYM_OWNER_CACHE_SIZE"}
    nodes = [node for node in tree.body if
             (isinstance(node, (ast.Import, ast.ImportFrom))
              and not any(alias.name.startswith("tilelang") for alias in node.names)
              and not getattr(node, "module", "") == "tilelang")
             or (isinstance(node, ast.FunctionDef) and node.name == "_sym_owner_metadata")
             or (isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.ns = metadata_namespace()
        self.metadata = self.ns["_sym_owner_metadata"]

    def test_layouts(self):
        for values in ([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2],
                       [2, 0, 2, 1, 2, 1], [0, 0, 2], []):
            owner = torch.tensor(values, dtype=torch.int64)
            uniform, offsets, order = self.metadata(owner, 3)
            self.assertEqual(uniform, values == [0, 0, 1, 1, 2, 2])
            if not uniform:
                self.assertEqual(offsets[-1].item(), len(values))
                for o in range(3):
                    rows = order[offsets[o]:offsets[o + 1]]
                    self.assertEqual(sorted(rows.tolist()), [i for i, v in enumerate(values) if v == o])

    def test_cache_views_and_mutation(self):
        graph = torch.tensor([[0, 0, 1, 1], [1, 2, 0, 1]])
        first = self.metadata(graph[0], 2)
        self.assertIs(first, self.metadata(graph[0], 2))
        graph[0, 1] = 1
        second = self.metadata(graph[0], 2)
        self.assertIsNot(first, second)
        self.assertEqual(second[1].tolist(), [0, 1, 4])

    def test_cache_bounded(self):
        graphs = [torch.tensor([0, 1, 1]) for _ in range(20)]
        for graph in graphs:
            self.metadata(graph, 2)
        self.assertLessEqual(len(self.ns["_sym_owner_cache"]), 8)

    def test_invalid(self):
        for owner in (torch.tensor([-1, 0]), torch.tensor([0, 3])):
            with self.assertRaises(ValueError):
                self.metadata(owner, 3)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for TileLang numerical tests")
class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("tilelang") is None:
            raise unittest.SkipTest("TileLang is not installed")
        spec = importlib.util.spec_from_file_location("sym_owner_kernels", SOURCE)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        torch.backends.cuda.matmul.allow_tf32 = False

    def test_forward_first_and_second_gradients(self):
        layouts = ([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2],
                   [0, 0, 1, 2, 2], [2, 0, 2, 1, 2, 1], [0, 0, 2], [])
        for values, (e, a) in product(layouts, ((5, 3), (64, 4), (128, 4))):
            with self.subTest(owner=values, E=e, A=a):
                torch.manual_seed(31)
                owner = torch.tensor(values, device="cuda", dtype=torch.int64)
                m, o, scale = len(values), 3, .37
                inputs = [torch.randn(shape, device="cuda", requires_grad=True)
                          for shape in ((m, e), (m, 3), (m,))]
                x, h, sw = inputs
                H = torch.zeros(o, 3, e, device="cuda").index_add(
                    0, owner, h[:, :, None] * x[:, None, :] * sw[:, None, None]) * scale
                ref = (H[:, :, :a].transpose(-1, -2) @ H / 3).reshape(1, o, a * e)
                actual = self.module.FusedSymmetrizationOpDynamic.apply(
                    x, h, sw, owner, o, 1, o, scale, a)
                torch.testing.assert_close(actual, ref, atol=2e-5, rtol=2e-4)
                upstream = torch.randn_like(ref, requires_grad=True)
                first_ref = torch.autograd.grad(ref, inputs, upstream, create_graph=True, retain_graph=True)
                first_actual = torch.autograd.grad(actual, inputs, upstream, create_graph=True, retain_graph=True)
                for expected, result in zip(first_ref, first_actual):
                    torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-4)
                vectors = [torch.randn_like(t) for t in first_ref]
                # Test both combined cotangents and a single used first gradient.
                for selected in ((0, 1, 2), (0,)):
                    sr = sum((first_ref[i] * vectors[i]).sum() for i in selected)
                    sa = sum((first_actual[i] * vectors[i]).sum() for i in selected)
                    targets = [upstream, *inputs]
                    second_ref = torch.autograd.grad(sr, targets, retain_graph=True)
                    second_actual = torch.autograd.grad(sa, targets, retain_graph=True)
                    for expected, result in zip(second_ref, second_actual):
                        torch.testing.assert_close(result, expected, atol=5e-5, rtol=5e-4)
                torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
