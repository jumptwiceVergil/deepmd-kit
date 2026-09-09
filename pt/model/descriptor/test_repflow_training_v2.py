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


if __name__ == "__main__":
    unittest.main()
