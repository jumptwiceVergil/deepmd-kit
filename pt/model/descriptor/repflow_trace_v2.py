"""Read-only raw trace attribution. Never infer full-call latency from GPU sums."""
from bisect import bisect_right
from collections import defaultdict
import json


def component(factory, event, ancestors):
    """Disjoint GPU buckets. A fill kernel alone does not prove a zero fill."""
    names = {p["name"] for p in ancestors}
    zero_ops = {"aten::zero_", "aten::zeros", "aten::zeros_like", "aten::new_zeros"}
    if names & zero_ops:
        return "zeroing"
    if event.get("cat") == "gpu_memcpy":
        return "memcpy"
    if event.get("cat") == "gpu_memset":
        return "memset_value_unknown"
    if "owner_metadata" in factory:
        return "owner_metadata"
    if "reduce" in factory:
        return "reduce"
    if "weights" in factory and "v2" in factory:
        return "weights_partial"
    if "weights" in factory:
        return "weights_unsplit"
    if "inputs" in factory:
        return "inputs"
    if factory != "torch_or_runtime":
        return "other_fused"
    if "FillFunctor" in event.get("name", ""):
        return "fill_value_unknown"
    return "other_torch_runtime"


def analyze(trace):
    events = trace["traceEvents"]
    cpu = [e for e in events if e.get("cat") in ("cpu_op", "user_annotation")
           and "dur" in e]
    cpu.sort(key=lambda e: (e["ts"], -e["dur"]))
    stacks = defaultdict(list)
    intervals = defaultdict(list)
    sources = defaultdict(set)
    external = {}
    labels = {}
    parents = {}
    unresolved_sequences = set()
    double_names = {"FusedSymmetrizationOpDynamicBackwardBackward": "sym",
                    "FusedEdgeUpdateFunctionBackwardBackward": "edge",
                    "FusedAngleUpdateFunctionBackwardBackward": "angle"}
    cpu_ranges = []
    factory_ranges = []
    for e in cpu:
        stack = stacks[e["pid"], e["tid"]]
        while stack and e["ts"] >= stack[-1]["ts"] + stack[-1]["dur"]:
            stack.pop()
        inherited = labels.get(id(stack[-1])) if stack else None
        parents[id(e)] = stack[-1] if stack else None
        label = inherited
        name, args = e["name"], e.get("args", {})
        if name.startswith("rfbench|"):
            _, stage, op, call = name.split("|", 3)
            label = stage, op, call
            # These are CPU annotation durations, not GPU-complete call times.
            cpu_ranges.append(dict(stage=stage, op=op, call=call, cpu_range_ms=e["dur"] / 1000))
        elif name in double_names:
            label = ("double_backward", double_names[name], "custom_function")
        elif name.startswith("autograd::engine::evaluate_function"):
            seq = args.get("Sequence number", -1)
            candidates = sources[e["pid"], seq]
            if len(candidates) == 1:
                stage, op, call = next(iter(candidates))
                label = ("double_backward" if stage == "backward" else "ordinary_backward", op, call)
            elif len(candidates) > 1:
                unresolved_sequences.add((e["pid"], seq))
                label = None  # Never guess when thread-local sequences collide.
        labels[id(e)] = label
        if name.startswith("rfbench_kernel|"):
            stage, op, call = label or ("unattributed", "other", "unknown")
            ancestor = parents[id(e)]
            cache_status = "unknown"
            while ancestor is not None:
                if ancestor["name"].startswith("rfbench_metadata|"):
                    cache_status = ancestor["name"].split("|", 1)[1]
                    break
                ancestor = parents[id(ancestor)]
            factory_ranges.append(dict(stage=stage, op=op, call=call,
                                       factory=name.split("|", 1)[1],
                                       cpu_range_ms=e["dur"] / 1000,
                                       cache_status=cache_status,
                                       inclusive=True))
        seq = args.get("Sequence number", -1)
        if (label and seq >= 0 and args.get("Fwd thread id", 0) == 0
                and not name.startswith("autograd::") and e.get("cat") == "cpu_op"
                and label[0] in ("forward", "backward")):
            sources[e["pid"], seq].add(label)
        if "External id" in args:
            external[e["pid"], args["External id"]] = e
        stack.append(e)
        intervals[e["pid"], e["tid"]].append(e)
    starts = {key: [e["ts"] for e in es] for key, es in intervals.items()}
    launches = {e["args"]["correlation"]: e for e in events
                if e.get("cat") in ("cuda_runtime", "cuda_driver")
                and "correlation" in e.get("args", {})}
    totals = defaultdict(lambda: dict(kernel_ms=0., activities=0, first_gpu_us=None, last_gpu_us=None))
    attribution = defaultdict(lambda: dict(gpu_ms=0., kernel_launches=0, memcpy_activities=0,
                                          memset_activities=0, activities=0))
    resources = {}
    missing = 0
    for e in events:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        launch = launches.get(e.get("args", {}).get("correlation"))
        label = None
        factory = "torch_or_runtime"
        containing = []
        if launch:
            key = launch["pid"], launch["tid"]
            es = intervals[key]
            j = bisect_right(starts.get(key, []), launch["ts"]) - 1
            # The shortest containing CPU range wins. Do not rely only on
            # External id: it can point to the enclosing custom autograd op.
            containing = []
            p = es[j] if j >= 0 else None
            while p is not None:
                if p["ts"] <= launch["ts"] < p["ts"] + p["dur"]:
                    containing.append(p)
                p = parents[id(p)]
            containing.sort(key=lambda p: p["dur"])
            label = next((labels[id(p)] for p in containing if labels.get(id(p))), None)
            factory = next((p["name"].split("|", 1)[1] for p in containing
                            if p["name"].startswith("rfbench_kernel|")), factory)
            if label is None:
                p = external.get((launch["pid"], launch.get("args", {}).get("External id")))
                label = labels.get(id(p))
        if label is None:
            label = "unattributed", "other", "unknown"
            missing += 1
        stage, op, call = label
        bucket = component(factory, e, containing)
        detail = attribution[stage, op, bucket, factory]
        detail["gpu_ms"] += e.get("dur", 0) / 1000
        detail["activities"] += 1
        counter = {"kernel": "kernel_launches", "gpu_memcpy": "memcpy_activities",
                   "gpu_memset": "memset_activities"}[e["cat"]]
        detail[counter] += 1
        if e["cat"] == "kernel":
            args = e.get("args", {})
            configuration = dict(kernel=e["name"], device=args.get("device"),
                                 registers_per_thread=args.get("registers per thread"),
                                 shared_bytes_per_block=args.get("shared memory"),
                                 grid=args.get("grid"), block=args.get("block"))
            key = stage, op, factory, bucket, json.dumps(configuration, sort_keys=True)
            if key not in resources:
                resources[key] = dict(stage=stage, op=op, factory=factory, component=bucket,
                                      **configuration, kernel_launches=0, gpu_ms=0.)
            resources[key]["kernel_launches"] += 1
            resources[key]["gpu_ms"] += e.get("dur", 0) / 1000
        row = totals[stage, op, factory]
        row["kernel_ms"] += e.get("dur", 0) / 1000
        row["activities"] += 1
        begin, end = e["ts"], e["ts"] + e.get("dur", 0)
        row["first_gpu_us"] = begin if row["first_gpu_us"] is None else min(begin, row["first_gpu_us"])
        row["last_gpu_us"] = end if row["last_gpu_us"] is None else max(end, row["last_gpu_us"])
    return dict(kernels=[dict(stage=s, op=o, factory=f, **v) for (s, o, f), v in totals.items()],
                attribution=[dict(stage=s, op=o, component=c, factory=f, **v)
                             for (s, o, c, f), v in attribution.items()],
                kernel_resources=list(resources.values()), factory_ranges=factory_ranges,
                cpu_ranges=cpu_ranges, unattributed_activities=missing,
                ambiguous_sequences=len(unresolved_sequences))
