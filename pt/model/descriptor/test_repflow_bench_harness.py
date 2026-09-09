"""CPU-only tests for reporting, autograd attribution and file isolation."""
import ast
import contextlib
import itertools
import json
import tempfile
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark_repflow_training import compare, prepare_config
from repflow_bench_adapters import Recorder, intercept_force_grad, MaterializingContext, select_version
from repflow_bench_report import attributed_kernels, render_reports


class HarnessTests(unittest.TestCase):
    def test_comparison_none_zero_and_failure(self):
        self.assertTrue(compare({"g": None}, {"g": torch.zeros(3)}, 1e-8, 1e-8)["pass"])
        self.assertFalse(compare({"g": torch.ones(3)}, {"g": torch.zeros(3)}, 1e-8, 1e-8)["pass"])

    def test_requested_absolute_tolerance(self):
        reference = {"g": torch.zeros(1)}
        actual = {"g": torch.tensor([5e-4])}
        self.assertFalse(compare(reference, actual, 2e-4, 2e-3)["pass"])
        result = compare(reference, actual, 1e-3, 2e-3)
        self.assertTrue(result["pass"])
        self.assertGreater(result["max_abs"], 0)
        self.assertFalse(compare(reference, {"g": torch.tensor([2e-3])}, 1e-3, 2e-3)["pass"])

    def test_file_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stat = root / "stats.hdf5"
            stat.write_bytes(b"test-stat")
            config = {"model": {}, "training": {"stat_file": str(stat),
                "training_data": {"systems": [str(root / "data")]}}}
            original = root / "input.json"
            original.write_text(json.dumps(config))
            before = original.read_bytes()
            output = root / "out"
            output.mkdir()
            prepared = prepare_config(SimpleNamespace(input=original), output)
            result = json.loads(prepared.read_text())
            self.assertEqual(original.read_bytes(), before)
            self.assertEqual(stat.read_bytes(), b"test-stat")
            self.assertNotEqual(result["training"]["stat_file"], str(stat))

    def test_materialization_proxy(self):
        class Ctx:
            def set_materialize_grads(self, flag):
                self.flag = flag
        ctx = Ctx()
        proxy = MaterializingContext(ctx)
        proxy.set_materialize_grads(False)
        proxy.foo = 9
        self.assertTrue(ctx.flag)
        self.assertEqual(ctx.foo, 9)

    def test_version_routes_both_method_names_and_restores(self):
        current = types.ModuleType("fake_current")
        legacy = types.ModuleType("fake_legacy")
        for module, factor in ((current, 2), (legacy, 3)):
            for clsname in ("FusedEdgeUpdateFunction", "FusedAngleUpdateFunction", "FusedSymmetrizationOpDynamic"):
                setattr(module, clsname, type(clsname, (), {"apply": staticmethod(lambda x, f=factor: x*f)}))
                setattr(module, clsname + "Backward", type(clsname + "Backward", (), {
                    "forward": staticmethod(lambda ctx, *a: a),
                    "backward": staticmethod(lambda ctx, *a: a),
                }))
            module._sym_owner_metadata = lambda *a: (True, None, None)
            module._sym_owner_cache = {}
        ns = {k: getattr(current, k) for k in
              ("FusedEdgeUpdateFunction", "FusedAngleUpdateFunction", "FusedSymmetrizationOpDynamic")}
        exec(
            "class Layer:\n"
            " def optim_edge_update_dynamic(self, x): return x\n"
            " def optim_angle_update_dynamic(self, x): return x\n"
            " def symmetrization_op_dynamic(self, x): return x\n"
            " def fused_optim_edge_update_dynamic(self, x): return FusedEdgeUpdateFunction.apply(x)\n"
            " def fused_optim_angle_update_dynamic(self, x): return FusedAngleUpdateFunction.apply(x)\n"
            " def fused_symmetrization_op_dynamic(self, x): return FusedSymmetrizationOpDynamic.apply(x)\n",
            ns,
        )
        layer_module = types.ModuleType("fake_layer")
        layer_module.RepFlowLayer = ns["Layer"]
        package = types.ModuleType("deepmd.pt.model.descriptor")
        package.repflow_layer, package.utils_tilelang = layer_module, current
        instance = ns["Layer"]()
        old = ns["Layer"].optim_edge_update_dynamic
        rec = Recorder()
        with patch.dict(sys.modules, {"deepmd.pt.model.descriptor": package, "repflow_bench_legacy": legacy}):
            for version, factor in (("original", 1), ("legacy_rebuilt", 3),
                                    ("v2_materialized", 2), ("current", 2)):
                with select_version(version, rec):
                    for name in ("optim_edge_update_dynamic", "fused_optim_edge_update_dynamic",
                                 "optim_angle_update_dynamic", "fused_optim_angle_update_dynamic",
                                 "symmetrization_op_dynamic", "fused_symmetrization_op_dynamic"):
                        self.assertEqual(getattr(instance, name)(torch.tensor(4)).item(), 4*factor)
                self.assertIs(ns["Layer"].optim_edge_update_dynamic, old)

    def test_real_autograd_sequence_attribution_cpu(self):
        rec = Recorder()
        rec.enabled = True
        rec.capture = True
        namespace = {"torch": torch}
        exec(compile(
            "def task_deriv_one(energy, extended_coord):\n"
            "    return -torch.autograd.grad([energy], [extended_coord], "
            "grad_outputs=[torch.ones_like(energy)], create_graph=True, retain_graph=True)[0]\n",
            "/example/transform_output.py", "exec"), namespace)
        x = torch.randn(5, 3, requires_grad=True)
        w = torch.randn(3, 4, requires_grad=True)
        with intercept_force_grad(rec):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                with rec.scope("forward", "edge", "layer0"):
                    y = x @ w
                rec.track_nodes(y, [x, w], "layer0", "edge")
                energy = y.square().sum()
                force = namespace["task_deriv_one"](energy, x)
                rec.phase = "loss"
                force.square().sum().backward()
        events = list(prof.events())
        # Emulate CUDA correlation only; autograd event sequences are real.
        for event in events:
            if event.name == "aten::mm":
                event.kernels.append(SimpleNamespace(name="fake_gemm", duration=10, device=0))
        rows, total = attributed_kernels(events)
        self.assertGreater(total, 0)
        stages = {r["stage"] for r in rows if r["op"] == "edge"}
        self.assertIn("forward", stages)
        self.assertIn("backward", stages)
        self.assertIn("double_backward", stages)
        self.assertEqual(rec.derivative_requests[0]["inputs"][0]["shape"], [5, 3])
        rec.clear_graph_hooks()

    def test_reports_three_tables_and_invalid_speedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = []
            for version, valid, ms in (("original", True, 4), ("legacy_rebuilt", True, 2),
                                       ("current", False, 1)):
                runs.append({"version": version, "frame": 0, "timings": {"loss_backward": [ms]},
                             "correctness": {"pass": valid}, "memory": {"loss_backward": [100, 200]}})
            render_reports(tmp, runs)
            for name in ("01_stage_times.csv", "02_attribution.csv", "03_correctness_resources.csv"):
                self.assertTrue((Path(tmp) / name).is_file())
            import csv
            with (Path(tmp) / "01_stage_times.csv").open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[1]["speedup_vs_original"], "2.0")
            self.assertEqual(rows[2]["speedup_vs_original"], "")

    def test_legacy_is_vendored_and_bias_fixed(self):
        source = Path(__file__).with_name("repflow_bench_legacy.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "fused_edge_update_input_backward_v1")
        loops = [n for n in ast.walk(factory) if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Name) and n.target.id == "ko"]
        self.assertIn("T.clear(bias_tile)", ast.unparse(loops[0].body[0]))

    def test_legacy_edge_bias_k65_math(self):
        # Interpret the vendored source, not a separately retyped formula.
        class T:
            grid = ()
            Tensor = staticmethod(lambda *a: None)
            prim_func = staticmethod(lambda f: f)
            ceildiv = staticmethod(lambda a, b: (a+b-1)//b)
            serial = staticmethod(range)
            Parallel = staticmethod(lambda *s: range(s[0]) if len(s) == 1
                                    else itertools.product(*(range(v) for v in s)))
            alloc_shared = staticmethod(lambda s, d: torch.zeros(s, dtype=torch.float64))
            alloc_fragment = alloc_shared
            clear = staticmethod(lambda a: a.zero_())
            sync_threads = staticmethod(lambda: None)
            atomic_add = staticmethod(lambda a, b: a.add_(b))
            cast = staticmethod(lambda a, d: a)
            gemm = staticmethod(lambda a, b, c: c.add_(a @ b))

            @contextlib.contextmanager
            def Kernel(self, *args, **kwargs):
                yield self.grid
        source = Path(__file__).with_name("repflow_bench_legacy.py").read_text(encoding="utf-8")
        factory = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                       and n.name == "fused_edge_update_input_backward_v1")
        factory.decorator_list = []
        interpreter = T()
        ns = {"T": interpreter}
        exec(compile(ast.Module(body=[factory], type_ignores=[]), "<baseline>", "exec"), ns)
        torch.manual_seed(7)
        e, k, de, dn, dx, nn, nx = 7, 65, 3, 5, 4, 4, 6
        g = torch.randn(e, k, dtype=torch.float64)
        we, wn, wx = [torch.randn(d, k, dtype=torch.float64) for d in (de, dn, dx)]
        ni, xi = torch.tensor([0, 0, 1, 2, 1, 0, 2]), torch.tensor([1, 0, 1, 0, 3, 2, 3])
        ge, gn, gx, gb = [torch.zeros(s, dtype=torch.float64)
                          for s in ((e, de), (nn, dn), (nx, dx), (k,))]
        kernel = ns[factory.name](e, k, de, dn, dx, nn, nx)
        for interpreter.grid in itertools.product(range((e+31)//32), range((max(de,dn,dx)+15)//16)):
            kernel(g, we, wn, wx, ni, xi, ge, gn, gx, gb)
        torch.testing.assert_close(gb, g.sum(0))
        torch.testing.assert_close(ge, g @ we.T)
        torch.testing.assert_close(gn, torch.zeros_like(gn).index_add_(0, ni, g @ wn.T))
        torch.testing.assert_close(gx, torch.zeros_like(gx).index_add_(0, xi, g @ wx.T))


if __name__ == "__main__":
    unittest.main()
