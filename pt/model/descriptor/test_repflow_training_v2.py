"""CPU regression tests for graph-preserving timing and trace reporting."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import benchmark_repflow_training_v2 as bench
from repflow_trace_v2 import analyze


class Tests(unittest.TestCase):
    def test_real_grad_is_called_once_with_original_inputs(self):
        x = torch.tensor(2., requires_grad=True)
        owner = torch.arange(8).reshape(2, 4)[0]
        before_base = owner._base
        y = x ** 3
        rec = SimpleNamespace(phase="forward", derivative_requests=[])
        ns = {"torch": torch}
        exec(compile("def task_deriv_one(y, x):\n return torch.autograd.grad([y], [x], create_graph=True, retain_graph=True)[0]\n",
                     "/model/transform_output.py", "exec"), ns)
        original = torch.autograd.grad
        with patch.object(torch.autograd, "grad", wraps=original) as spy:
            with bench.force_calls(rec, False, []):
                force = ns["task_deriv_one"](y, x)
            self.assertEqual(spy.call_count, 1)
            self.assertIs(spy.call_args.args[1][0], x)
        force.backward()
        self.assertEqual(x.grad.item(), 12.)
        self.assertIs(owner._base, before_base)
        self.assertEqual(rec.phase, "forward")
        self.assertEqual(len(rec.derivative_requests), 1)

    def test_boundary_mode_only_times_outer_grad(self):
        rec = SimpleNamespace(phase="forward", derivative_requests=[])
        ns = {"torch": torch}
        exec(compile("def task_deriv_one(y, x):\n return torch.autograd.grad([y], [x], create_graph=True, retain_graph=True)[0]\n",
                     "/model/transform_output.py", "exec"), ns)
        x = torch.tensor(2., requires_grad=True)
        with patch.object(bench, "timed", side_effect=lambda fn, rows, stage: fn()) as timer:
            with bench.force_calls(rec, True, []):
                ns["task_deriv_one"](x.square(), x)
            timer.assert_called_once()
            self.assertEqual(timer.call_args.args[2], "force_autograd")

    def test_custom_double_correlation_and_factory(self):
        result = analyze({"traceEvents": [
            dict(cat="cpu_op", name="FusedAngleUpdateFunctionBackwardBackward", pid=1, tid=2, ts=0, dur=40, args={"External id": 1}),
            dict(cat="user_annotation", name="rfbench_kernel|partials", pid=1, tid=2, ts=5, dur=20, args={"External id": 99}),
            dict(cat="cuda_driver", name="cuLaunchKernel", pid=1, tid=2, ts=10, dur=1, args={"correlation": 4, "External id": 1}),
            dict(cat="kernel", name="kernel", pid=0, tid=7, ts=100, dur=8, args={"correlation": 4})]})
        row = result["kernels"][0]
        self.assertEqual((row["op"], row["stage"], row["factory"]), ("angle", "double_backward", "partials"))
        self.assertEqual(row["kernel_ms"], .008)

    def test_native_sequence_double_attribution(self):
        result = analyze({"traceEvents": [
            dict(cat="user_annotation", name="rfbench|backward|edge|layer0", pid=1, tid=2, ts=0, dur=20),
            dict(cat="cpu_op", name="aten::mm", pid=1, tid=2, ts=1, dur=10,
                 args={"Sequence number": 50, "Fwd thread id": 0}),
            dict(cat="cpu_op", name="autograd::engine::evaluate_function: MmBackward0", pid=1, tid=3, ts=30, dur=20,
                 args={"Sequence number": 50, "Fwd thread id": 9}),
            dict(cat="cuda_runtime", name="cudaLaunchKernel", pid=1, tid=3, ts=35, dur=1, args={"correlation": 8}),
            dict(cat="kernel", name="mm", pid=0, tid=7, ts=100, dur=4, args={"correlation": 8})]})
        self.assertEqual(result["kernels"][0]["stage"], "double_backward")

    def test_report_labels_and_no_boundary_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            bench.write_report(path, [dict(version="original", frame=0, samples=[[
                dict(stage="full_step", wall_ms=2., cuda_ms=1.)]], profiles=[])])
            data = json.loads((path / "results.json").read_text())
            self.assertEqual(data["summary"][0]["correctness"], "SKIPPED")
            text = (path / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("缺失", text)
            self.assertIn("不同阶段来自不同新图采样", text)
        tree = ast.parse(Path(bench.__file__).read_text(encoding="utf-8"))
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertNotIn("repflow_full_timing", imports)

    def test_zeroing_resources_and_no_double_counting(self):
        events = [dict(cat="user_annotation", name="rfbench|backward|angle|layer0", pid=1, tid=2, ts=0, dur=100),
                  dict(cat="user_annotation", name="rfbench_kernel|fused_angle_update_backward_weights_v2", pid=1, tid=2, ts=1, dur=90),
                  dict(cat="cpu_op", name="aten::zeros", pid=1, tid=2, ts=2, dur=10),
                  dict(cat="cuda_runtime", name="cudaLaunchKernel", pid=1, tid=2, ts=4, dur=1, args={"correlation": 1}),
                  dict(cat="cuda_runtime", name="cudaLaunchKernel", pid=1, tid=2, ts=20, dur=1, args={"correlation": 2}),
                  dict(cat="kernel", name="FillFunctor", pid=0, tid=7, ts=200, dur=3, args={"correlation": 1}),
                  dict(cat="kernel", name="partials", pid=0, tid=7, ts=204, dur=7,
                       args={"correlation": 2, "registers per thread": 32, "shared memory": 0,
                             "grid": [8, 1, 1], "block": [128, 1, 1]})]
        result = analyze({"traceEvents": events})
        self.assertEqual({r["component"] for r in result["attribution"]}, {"zeroing", "weights_partial"})
        self.assertAlmostEqual(sum(r["gpu_ms"] for r in result["attribution"]), .01)
        self.assertAlmostEqual(sum(r["kernel_ms"] for r in result["kernels"]), .01)
        zero, partial = result["kernel_resources"]
        self.assertIsNone(zero["shared_bytes_per_block"])
        self.assertEqual(partial["shared_bytes_per_block"], 0)
        self.assertEqual(partial["registers_per_thread"], 32)
        self.assertEqual(partial["kernel_launches"], 1)

    def test_unknown_fill_and_memset_are_not_assumed_zero(self):
        from repflow_trace_v2 import component
        self.assertEqual(component("torch_or_runtime", dict(cat="kernel", name="FillFunctor"), []),
                         "fill_value_unknown")
        self.assertEqual(component("torch_or_runtime", dict(cat="gpu_memset", name="Memset"), []),
                         "memset_value_unknown")

    def test_metadata_hit_and_miss_cpu_diagnostics(self):
        from repflow_resource_report_v2 import tables
        result = analyze({"traceEvents": [
            dict(cat="user_annotation", name="rfbench|forward|sym|layer0", pid=1, tid=2, ts=0, dur=100),
            dict(cat="user_annotation", name="rfbench_metadata|hit", pid=1, tid=2, ts=1, dur=20),
            dict(cat="user_annotation", name="rfbench_kernel|owner_metadata", pid=1, tid=2, ts=2, dur=10)]})
        with tempfile.TemporaryDirectory() as tmp:
            _, data = tables(Path(tmp), [dict(version="current", frame=0, profiles=[result])])
        meta = data["owner_metadata"][0]
        self.assertEqual(meta["hit_calls"], 1)
        self.assertEqual(meta["miss_calls"], 0)
        self.assertEqual(meta["hit_cpu_ms"], .01)

    def test_memory_sample_reset_and_baseline(self):
        rows, actions = [], []
        fake = SimpleNamespace(synchronize=lambda: actions.append("sync"),
                               memory_allocated=lambda: 100, memory_reserved=lambda: 200,
                               reset_peak_memory_stats=lambda: actions.append("reset"),
                               max_memory_allocated=lambda: 180, max_memory_reserved=lambda: 256)
        with patch.object(torch, "cuda", fake):
            result = bench.measure_memory(lambda: actions.append("compute") or 9, rows, "loss_backward")
        self.assertEqual(result, 9)
        self.assertEqual(actions, ["sync", "reset", "compute", "sync"])
        self.assertEqual(rows[0]["increment_allocated_bytes"], 80)
        self.assertEqual(rows[0]["baseline_reserved_bytes"], 200)

    def test_resource_table_skipped_memory_and_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench.write_report(Path(tmp), [dict(version="current", frame=0, samples=[], profiles=[], memory=[
                dict(stage="full_step", baseline_allocated_bytes=1048576, peak_allocated_bytes=2097152,
                     increment_allocated_bytes=1048576, baseline_reserved_bytes=3145728,
                     peak_reserved_bytes=3145728)])])
            text = (Path(tmp) / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("优化归因表", text)
            self.assertIn("正确性与资源表", text)
            data = json.loads((Path(tmp) / "attribution_resources.json").read_text())
            self.assertIsNone(data["correctness"][0]["max_abs"])
            self.assertEqual(data["memory"][0]["peak_allocated_bytes"], 2097152)


if __name__ == "__main__":
    unittest.main()
