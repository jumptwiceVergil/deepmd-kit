"""Synchronized operator boundaries; derivative formulas always come from autograd.

Benchmark-only graph encapsulation, NOT a replacement for production autograd.
It creates local leaf views and reconnects their VJPs to the real training graph.
"""
import contextlib
import inspect
import time
from collections import defaultdict

import torch


class Timer:
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.enabled = False
        self.profile = False
        self.phase = "forward"
        self.targets = ()
        self.rows = []
        self.requests = []
        self.reachable = {}

    def depends_on_target(self, tensor):
        if any(tensor is t for t in self.targets):
            return True
        targets = {t.grad_fn for t in self.targets if t.grad_fn is not None}
        root = tensor.grad_fn
        pending = [(root, False)]
        while pending:
            node, visited = pending.pop()
            if node is None or node in self.reachable:
                continue
            if node in targets or any(getattr(node, "variable", None) is t for t in self.targets):
                self.reachable[node] = True
            elif visited:
                self.reachable[node] = any(self.reachable.get(n, False) for n, _ in node.next_functions)
            else:
                pending.append((node, True))
                pending.extend((n, False) for n, _ in node.next_functions)
        return self.reachable.get(root, False)

    def call(self, op, stage, label, fn):
        if not self.enabled and not self.profile:
            return fn()
        # Profiler is a separate pass: its overhead never enters timing samples.
        if self.profile:
            with torch.profiler.record_function(f"rf_full|{op}|{stage}|{label}"):
                return fn()
        if self.cuda:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            # Initialize lazy event resources outside the timed interval.
            start.record()
            end.record()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if self.cuda:
            start.record()
        result = fn()
        if self.cuda:
            end.record()
            torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000
        self.rows.append(dict(op=op, stage=stage, call=label, wall_ms=wall,
                              cuda_ms=start.elapsed_time(end) if self.cuda else None))
        return result


class _VJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, go, *outer):
        timer, op, label, local, output, selected = state
        ctx.set_materialize_grads(False)
        g = go.detach().requires_grad_(True)
        with torch.enable_grad():
            grads = torch.autograd.grad(output, [local[i] for i in selected], g,
                                        create_graph=True, retain_graph=True,
                                        allow_unused=True)
        ctx.state = state
        ctx.g = g
        ctx.grads = grads
        # None outputs are supported by Function.apply; do not replace with zeros.
        return tuple(g.detach() if g is not None else None for g in grads)

    @staticmethod
    def backward(ctx, *cotangents):
        timer, op, label, local, output, selected = ctx.state
        def compute():
            active = [(g, u) for g, u in zip(ctx.grads, cotangents)
                      if g is not None and g.requires_grad and u is not None]
            indices = [i for i, x in enumerate(local) if x.requires_grad]
            result = [None] * len(local)
            gg = None
            if active:
                # Differentiate the actual first-backward graph, including the
                # installed TileLang custom backward when using a fused variant.
                values = torch.autograd.grad(
                    [g for g, _ in active], [ctx.g] + [local[i] for i in indices],
                    [u for _, u in active], allow_unused=True, retain_graph=True)
                gg = values[0]
                for i, value in zip(indices, values[1:]):
                    result[i] = value
            return (None, gg, *result)
        return timer.call(op, "double_backward", label, compute)


class _Boundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, timer, op, label, fn, *outer):
        ctx.set_materialize_grads(False)
        local = tuple(x.detach().requires_grad_(x.requires_grad) for x in outer)
        def compute():
            with torch.enable_grad():
                return fn(local)
        output = timer.call(op, "forward", label, compute)
        if not isinstance(output, torch.Tensor):
            raise TypeError("RepFlow benchmark expects one Tensor output")
        ctx.state = timer, op, label, local, output
        ctx.save_for_backward(*outer)
        return output.detach()

    @staticmethod
    def backward(ctx, go):
        timer, op, label, local, output = ctx.state
        outer = ctx.saved_tensors
        if go is None:
            return (None,) * (4 + len(outer))
        force = timer.phase == "force"
        selected = [i for i, x in enumerate(outer) if x.requires_grad
                    and (not force or timer.depends_on_target(x))]
        def compute():
            result = [None] * len(outer)
            if selected:
                if torch.is_grad_enabled():
                    state = timer, op, label, local, output, selected
                    values = _VJP.apply(state, go, *outer)
                else:
                    values = torch.autograd.grad(
                        output, [local[i] for i in selected], go,
                        retain_graph=True, allow_unused=True)
                for i, value in zip(selected, values):
                    result[i] = value
            return (None, None, None, None, *result)
        stage = "backward" if force else "ordinary_backward"
        return timer.call(op, stage, label, compute)


def enclose(timer, op, label, fn, tensors):
    return _Boundary.apply(timer, op, label, fn, *tensors)


@contextlib.contextmanager
def force_path(timer):
    """Intercept only the real task_deriv_one coordinate derivative request."""
    original = torch.autograd.grad
    def grad(*args, **kwargs):
        frame = inspect.currentframe().f_back
        matched = (frame.f_code.co_name == "task_deriv_one" and
                   frame.f_code.co_filename.replace("\\", "/").endswith("/transform_output.py"))
        del frame
        if not matched:
            return original(*args, **kwargs)
        if not kwargs.get("create_graph") or not kwargs.get("retain_graph"):
            raise RuntimeError("Expected create_graph=True, retain_graph=True")
        timer.targets = tuple(kwargs.get("inputs", args[1] if len(args) > 1 else ()))
        timer.reachable.clear()
        timer.requests.append([dict(shape=list(x.shape), dtype=str(x.dtype)) for x in timer.targets])
        old = timer.phase
        timer.phase = "force"
        try:
            return original(*args, **kwargs)
        finally:
            timer.phase = old
            timer.reachable.clear()
            timer.targets = ()
    torch.autograd.grad = grad
    try:
        yield
    finally:
        torch.autograd.grad = original


class _Proxy:
    def __init__(self, base, overrides):
        self.base, self.overrides = base, overrides

    def __getattr__(self, name):
        return self.overrides[name] if name in self.overrides else getattr(self.base, name)


@contextlib.contextmanager
def boundaries(timer, layer_class, names):
    """Wrap already-selected methods, with matrix/bias explicit graph inputs."""
    from repflow_bench_adapters import OPS
    saved = []
    counts = defaultdict(int)
    try:
        for op, (plain, fused, _) in OPS.items():
            fn = getattr(layer_class, plain)
            # The version adapter already routes both names to this same method.
            def method(self, *args, _op=op, _fn=fn, **kwargs):
                # Signatures come from the original methods stored by the runner.
                bound = names["signatures"][_op].bind(self, *args, **kwargs)
                bound.apply_defaults()
                arguments = dict(bound.arguments)
                arguments.pop("self")
                tensor_keys = [k for k, v in arguments.items() if isinstance(v, torch.Tensor)]
                tensors = [arguments[k] for k in tensor_keys]
                feat = arguments.get("feat", "")
                linear_name = None
                if _op == "edge":
                    linear_name = {"node": "node_edge_linear", "edge": "edge_self_linear"}[feat]
                elif _op == "angle":
                    linear_name = {"edge": "edge_angle_linear1", "angle": "angle_self_linear"}[feat]
                if linear_name:
                    linear = getattr(self, linear_name)
                    if linear.bias is None:
                        raise RuntimeError("This benchmark expects the supplied model's linear bias")
                    tensors.extend([linear.matrix, linear.bias])
                key = f'{names["layers"][id(self)]}/{_op}/{feat}'
                index = counts[key]
                counts[key] += 1
                label = f"{key}/call{index}"
                def invoke(values):
                    kw = dict(arguments)
                    kw.update(zip(tensor_keys, values))
                    target = self
                    if linear_name:
                        target = _Proxy(self, {linear_name: _Proxy(linear, {
                            "matrix": values[-2], "bias": values[-1]})})
                    return _fn(target, **kw)
                return enclose(timer, _op, label, invoke, tensors)
            for name in (plain, fused):
                saved.append((name, getattr(layer_class, name)))
                setattr(layer_class, name, method)
        yield counts
    finally:
        for name, fn in reversed(saved):
            setattr(layer_class, name, fn)


def kernel_sums(trace):
    """Raw trace: GPU correlation -> actual launch -> innermost full-call range."""
    events = trace["traceEvents"]
    launches = {e["args"]["correlation"]: e for e in events
                if e.get("cat") in ("cuda_runtime", "cuda_driver")
                and "correlation" in e.get("args", {})}
    ranges = defaultdict(list)
    for e in events:
        if e.get("cat") == "user_annotation" and e["name"].startswith("rf_full|"):
            ranges[e["pid"], e["tid"]].append(e)
    totals = defaultdict(lambda: dict(kernel_ms=0., activities=0))
    for e in events:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        launch = launches.get(e.get("args", {}).get("correlation"))
        if launch is None:
            continue
        enclosing = [r for r in ranges[launch["pid"], launch["tid"]]
                     if r["ts"] <= launch["ts"] < r["ts"] + r["dur"]]
        if not enclosing:
            continue
        _, op, stage, label = min(enclosing, key=lambda r: r["dur"])["name"].split("|", 3)
        row = totals[op, stage]
        row["kernel_ms"] += e.get("dur", 0) / 1000
        row["activities"] += 1
    return [dict(op=op, stage=stage, **value) for (op, stage), value in totals.items()]
