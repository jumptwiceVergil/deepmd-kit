"""Read-only raw trace attribution. Never infer full-call latency from GPU sums."""
from bisect import bisect_right
from collections import defaultdict


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
    missing = 0
    for e in events:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        launch = launches.get(e.get("args", {}).get("correlation"))
        label = None
        factory = "torch_or_runtime"
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
        row = totals[stage, op, factory]
        row["kernel_ms"] += e.get("dur", 0) / 1000
        row["activities"] += 1
        begin, end = e["ts"], e["ts"] + e.get("dur", 0)
        row["first_gpu_us"] = begin if row["first_gpu_us"] is None else min(begin, row["first_gpu_us"])
        row["last_gpu_us"] = end if row["last_gpu_us"] is None else max(end, row["last_gpu_us"])
    return dict(kernels=[dict(stage=s, op=o, factory=f, **v) for (s, o, f), v in totals.items()],
                cpu_ranges=cpu_ranges, unattributed_activities=missing,
                ambiguous_sequences=len(unresolved_sequences))
