"""Process-local version selection and attribution. Never edits package files."""
import contextlib
import functools
import inspect
import types

import torch


OPS = {
    "sym": ("symmetrization_op_dynamic", "fused_symmetrization_op_dynamic", "FusedSymmetrizationOpDynamic"),
    "edge": ("optim_edge_update_dynamic", "fused_optim_edge_update_dynamic", "FusedEdgeUpdateFunction"),
    "angle": ("optim_angle_update_dynamic", "fused_optim_angle_update_dynamic", "FusedAngleUpdateFunction"),
}
VERSIONS = ("original", "legacy_rebuilt", "v2_materialized", "current")


class MaterializingContext:
    def __init__(self, ctx):
        object.__setattr__(self, "_ctx", ctx)

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def __setattr__(self, name, value):
        setattr(self._ctx, name, value)

    def set_materialize_grads(self, flag):
        self._ctx.set_materialize_grads(True)


class Recorder:
    def __init__(self):
        self.enabled = False
        self.phase = "forward"
        self.handles = []
        self.claimed = set()
        self.call_counts = {}
        self.shapes = {}
        self.index_samples = {}
        self.kernel_specs = {}
        self.metadata_stats = {}
        self.missing = {}
        self.layer_names = {}
        self.derivative_requests = []
        self.force_events = []
        self.force_memory = []
        self.measure = False
        self.memory_target = None
        self.capture = False
        self.attribution = True

    def clear_graph_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.claimed.clear()
        self.call_counts.clear()
        self.force_events.clear()
        self.force_memory.clear()

    def label(self, obj, op, bound):
        layer = self.layer_names.get(id(obj), "unknown_layer")
        feature = bound.arguments.get("feat", "")
        tensors = [v for v in bound.arguments.values() if isinstance(v, torch.Tensor)]
        signature = ",".join("x".join(map(str, v.shape)) for v in tensors[:3])
        key = f"{layer}/{op}/{feature}/{signature}"
        call = self.call_counts.get(key, 0)
        self.call_counts[key] = call + 1
        return f"{key}/call{call}"

    @contextlib.contextmanager
    def scope(self, stage, op, label):
        if self.enabled:
            with torch.profiler.record_function(f"rfbench|{stage}|{op}|{label}"):
                yield
        else:
            yield

    def track_nodes(self, result, boundary, label, op):
        if not self.enabled or not self.attribution:
            return
        stops = {v.grad_fn for v in boundary if isinstance(v, torch.Tensor) and v.grad_fn is not None}
        pending = [result.grad_fn]
        while pending:
            node = pending.pop()
            if node is None or node in stops or node in self.claimed:
                continue
            if node.name().endswith("AccumulateGrad"):
                continue
            self.claimed.add(node)
            stack = []

            def before(grads, stack=stack, label=label, op=op):
                stage = "backward" if self.phase == "force" else "ordinary_backward"
                cm = self.scope(stage, op, label)
                cm.__enter__()
                stack.append(cm)

            def after(inputs, outputs, stack=stack):
                stack.pop().__exit__(None, None, None)

            self.handles.extend((node.register_prehook(before), node.register_hook(after)))
            pending.extend(n for n, _ in node.next_functions)


@contextlib.contextmanager
def select_version(version, recorder):
    """Route both original/fused method names, including mixed current call sites."""
    from deepmd.pt.model.descriptor import repflow_layer as layer_module
    from deepmd.pt.model.descriptor import utils_tilelang as current
    import repflow_bench_legacy as legacy

    cls = layer_module.RepFlowLayer
    selected = legacy if version == "legacy_rebuilt" else current
    restore = []
    dict_restore = []
    try:
        if version != "original":
            old_metadata = selected._sym_owner_metadata

            def owner_metadata(owner, num_owner):
                if not recorder.enabled and not recorder.capture:
                    return old_metadata(owner, num_owner)
                base = owner
                while base._base is not None:
                    base = base._base
                key = (id(base), owner.data_ptr(), owner.numel(), owner.stride(),
                       owner._version, num_owner, owner.device)
                cached = selected._sym_owner_cache.get(key)
                hit = cached is not None and cached[0]() is base
                with torch.profiler.record_function("rfbench_kernel|owner_metadata") if recorder.enabled else contextlib.nullcontext():
                    result = old_metadata(owner, num_owner)
                if recorder.capture:
                    label = ("hit" if hit else "miss") + ("/uniform" if result[0] else "/segmented")
                    recorder.metadata_stats[label] = recorder.metadata_stats.get(label, 0) + 1
                return result

            dict_restore.append((selected.__dict__, "_sym_owner_metadata", old_metadata))
            selected._sym_owner_metadata = owner_metadata
        if version == "v2_materialized":
            for name in ("FusedEdgeUpdateFunctionBackward", "FusedAngleUpdateFunctionBackward"):
                target = getattr(current, name)
                original = target.forward

                def forward(ctx, *args, _fn=original):
                    return _fn(MaterializingContext(ctx), *args)

                restore.append((target, "forward", original))
                target.forward = staticmethod(forward)

        # Record the actual None pattern at the custom double-backward boundary.
        if version != "original":
            for name in ("FusedEdgeUpdateFunctionBackward", "FusedAngleUpdateFunctionBackward",
                         "FusedSymmetrizationOpDynamicBackward"):
                target = getattr(selected, name)
                original = target.backward

                def backward(ctx, *args, _fn=original, _name=name):
                    if recorder.capture:
                        key = _name + ":" + "".join("0" if x is None else "1" for x in args)
                        recorder.missing[key] = recorder.missing.get(key, 0) + 1
                    return _fn(ctx, *args)

                restore.append((target, "backward", original))
                target.backward = staticmethod(backward)

        # Wrap factory launches, not just partial GEMMs: reductions remain visible.
        for name, factory in list(vars(selected).items()):
            if not name.startswith("fused_") or not callable(factory):
                continue

            def make(*args, _fn=factory, _name=name, **kw):
                kernel = _fn(*args, **kw)
                if not recorder.enabled and not recorder.capture:
                    return kernel
                if recorder.capture:
                    spec = repr((args, sorted(kw.items())))
                    recorder.kernel_specs[_name + ":" + spec] = {
                        "factory": _name, "args": repr(args), "kwargs": kw,
                    }

                @functools.wraps(kernel)
                def launch(*values):
                    if recorder.enabled:
                        with torch.profiler.record_function("rfbench_kernel|" + _name):
                            return kernel(*values)
                    return kernel(*values)
                return launch

            dict_restore.append((selected.__dict__, name, factory))
            selected.__dict__[name] = make

        for op, (plain, fused, class_name) in OPS.items():
            original_plain = getattr(cls, plain)
            original_fused = getattr(cls, fused)
            if version == "original":
                implementation = original_plain
            else:
                namespace = dict(original_fused.__globals__)
                namespace[class_name] = getattr(selected, class_name)
                implementation = types.FunctionType(
                    original_fused.__code__, namespace, original_fused.__name__,
                    original_fused.__defaults__, original_fused.__closure__)
            signature = inspect.signature(original_plain)

            def method(self, *args, _impl=implementation, _sig=signature, _op=op, **kw):
                if not recorder.enabled and not recorder.capture:
                    return _impl(self, *args, **kw)
                bound = _sig.bind(self, *args, **kw)
                bound.apply_defaults()
                label = recorder.label(self, _op, bound)
                boundary = [v for v in bound.arguments.values() if isinstance(v, torch.Tensor)]
                if recorder.capture:
                    recorder.shapes[label] = {
                        k: {"shape": list(v.shape), "dtype": str(v.dtype), "requires_grad": v.requires_grad}
                        for k, v in bound.arguments.items() if isinstance(v, torch.Tensor)
                    }
                    for k, v in bound.arguments.items():
                        if isinstance(v, torch.Tensor) and (k == "owner" or "index" in k):
                            recorder.index_samples[label + "/" + k] = v.detach().cpu().tolist()
                with recorder.scope("forward", _op, label):
                    result = _impl(self, *args, **kw)
                recorder.track_nodes(result, boundary, label, _op)
                return result

            for name, old in ((plain, original_plain), (fused, original_fused)):
                restore.append((cls, name, old))
                setattr(cls, name, method)
        yield
    finally:
        recorder.clear_graph_hooks()
        for namespace, key, old in reversed(dict_restore):
            namespace[key] = old
        for obj, name, old in reversed(restore):
            setattr(obj, name, staticmethod(old) if name in ("forward", "backward") else old)


@contextlib.contextmanager
def intercept_force_grad(recorder):
    """Keep task_deriv_one intact; intercept only its exact energy/coordinate grad call."""
    import inspect
    original = torch.autograd.grad

    def grad(*args, **kw):
        frame = inspect.currentframe().f_back
        is_force = (frame.f_code.co_name == "task_deriv_one"
                    and frame.f_code.co_filename.replace("\\", "/").endswith("/transform_output.py"))
        del frame
        if not is_force:
            return original(*args, **kw)
        if not kw.get("create_graph") or not kw.get("retain_graph"):
            raise RuntimeError("Training force path must use create_graph=True, retain_graph=True")
        previous = recorder.phase
        recorder.phase = "force"
        if recorder.capture:
            xs = kw.get("inputs", args[1] if len(args) > 1 else ())
            recorder.derivative_requests.append({
                "inputs": [{"shape": list(x.shape), "dtype": str(x.dtype)} for x in xs],
                "create_graph": kw.get("create_graph"), "retain_graph": kw.get("retain_graph"),
            })
        if recorder.measure:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        if recorder.memory_target == "force":
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        try:
            with torch.profiler.record_function("rfbench_force_grad") if recorder.enabled else contextlib.nullcontext():
                result = original(*args, **kw)
        finally:
            if recorder.measure:
                end.record()
                recorder.force_events.append((start, end))
            if recorder.memory_target == "force":
                torch.cuda.synchronize()
                recorder.force_memory.append((baseline, torch.cuda.max_memory_allocated()))
            recorder.phase = previous
        return result
    torch.autograd.grad = grad
    try:
        yield
    finally:
        torch.autograd.grad = original
