"""CPU tests of measurement plumbing, not validation of the TileLang operators."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from repflow_full_timing import Timer, enclose, force_path, kernel_sums
from benchmark_repflow_full_calls import report


class FullTimingTests(unittest.TestCase):
    def test_real_force_loss_chain(self):
        def run(wrapped):
            torch.manual_seed(31)
            coord = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
            weight = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
            bias = torch.randn(4, dtype=torch.double, requires_grad=True)
            timer = Timer(cuda=False)
            timer.enabled = True
            def apply(op, fn, xs):
                return enclose(timer, op, op, fn, xs) if wrapped else fn(xs)
            x = coord.sin()
            y = apply("angle", lambda xs: xs[0] @ xs[1] + xs[2], (x, weight, bias))
            z = apply("sym", lambda xs: xs[0].square(), (y,))
            energy = z.sin().sum()
            timer.phase, timer.targets = "force", (coord,)
            force = torch.autograd.grad(energy, coord, create_graph=True, retain_graph=True)[0]
            timer.phase = "loss"
            loss = force.square().sum() + energy.square() * .1
            loss.backward()
            return force.detach(), weight.grad, bias.grad, coord.grad, timer.rows
        plain, wrapped = run(False), run(True)
        for a, b in zip(plain[:4], wrapped[:4]):
            torch.testing.assert_close(a, b)
        for op in ("sym", "angle"):
            self.assertEqual({r["stage"] for r in wrapped[4] if r["op"] == op},
                             {"forward", "backward", "double_backward", "ordinary_backward"})

    def test_unused_and_repeated_inputs(self):
        x = torch.tensor([2.], requires_grad=True)
        unused = torch.tensor([3.], requires_grad=True)
        timer = Timer(cuda=False)
        y = enclose(timer, "edge", "test", lambda xs: xs[0] * xs[1], (x, x, unused))
        timer.phase, timer.targets = "force", (x,)
        g = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)[0]
        timer.phase = "loss"
        g.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([2.]))
        self.assertIsNone(unused.grad)

    def test_force_intercept_and_restore(self):
        timer = Timer(cuda=False)
        before = torch.autograd.grad
        namespace = {"torch": torch}
        exec(compile("def task_deriv_one(y, x):\n return torch.autograd.grad([y], [x], create_graph=True, retain_graph=True)[0]\n",
                     "/test/transform_output.py", "exec"), namespace)
        x = torch.tensor(2., requires_grad=True)
        with force_path(timer):
            namespace["task_deriv_one"](x.square(), x)
        self.assertIs(torch.autograd.grad, before)
        self.assertEqual(len(timer.requests), 1)

    def test_trace_correlation_not_external_id(self):
        rows = kernel_sums({"traceEvents": [
            dict(cat="user_annotation", name="rf_full|angle|double_backward|layer0", pid=1, tid=2, ts=10, dur=10),
            dict(cat="cuda_driver", name="cuLaunchKernel", pid=1, tid=2, ts=12, dur=1,
                 args={"correlation": 30, "External id": 999}),
            dict(cat="kernel", name="gemm", pid=0, tid=7, ts=100, dur=5,
                 args={"correlation": 30, "External id": 999})]})
        self.assertEqual(rows[0]["kernel_ms"], .005)
        self.assertEqual(rows[0]["stage"], "double_backward")

    def test_report_skipped_and_three_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = [dict(version="original", frame=0, samples=[[
                dict(op="sym", stage="forward", wall_ms=2., cuda_ms=1.),
                dict(op="sym", stage="forward", wall_ms=4., cuda_ms=2.)]], profiles=[])]
            report(Path(directory), runs)
            data = json.loads((Path(directory) / "results.json").read_text())
            self.assertEqual(data["summary"][0]["wall_ms_median"], 6.)
            self.assertEqual(data["summary"][0]["correctness"], "SKIPPED")
            self.assertEqual((Path(directory) / "REPORT.md").read_text(encoding="utf-8").count("## "), 3)


if __name__ == "__main__":
    unittest.main()
