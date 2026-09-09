"""Attribution and resource tables, independent of wall-clock timing samples."""
from collections import defaultdict
import json


def tables(directory, runs):
    attribution, metadata, resources, memory, correctness = [], [], [], [], []
    for run in runs:
        identity = dict(version=run["version"], frame=run["frame"])
        correctness.append(dict(**identity, status="SKIPPED", max_abs=None,
                                relative_l2=None, runtime_error=run.get("error")))
        memory.extend(dict(**identity, **r) for r in run.get("memory", []))
        for sample, profile in enumerate(run.get("profiles", [])):
            tag = dict(**identity, profile_sample=sample)
            attribution.extend(dict(**tag, **r) for r in profile.get("attribution", []))
            resources.extend(dict(**tag, **r) for r in profile.get("kernel_resources", []))
            grouped = defaultdict(lambda: dict(cpu_ms=0., calls=0, hit_calls=0, miss_calls=0,
                                               unknown_calls=0, hit_cpu_ms=0., miss_cpu_ms=0.))
            for r in profile.get("factory_ranges", []):
                if r["factory"] != "owner_metadata":
                    continue
                row = grouped[r["op"], r["stage"]]
                row["cpu_ms"] += r["cpu_range_ms"]
                row["calls"] += 1
                status = r.get("cache_status", "unknown")
                row[status + "_calls"] += 1
                if status in ("hit", "miss"):
                    row[status + "_cpu_ms"] += r["cpu_range_ms"]
            for (op, stage), r in grouped.items():
                # Inclusive metadata GPU work is a cross-cutting view. Its
                # zeroing/memcpy activities remain in disjoint component rows.
                gpu = sum(a["gpu_ms"] for a in profile.get("attribution", [])
                          if a["factory"] == "owner_metadata" and a["op"] == op and a["stage"] == stage)
                metadata.append(dict(**tag, op=op, stage=stage, **r,
                                     inclusive_gpu_ms=gpu, inclusive=True))
    payload = dict(correctness=correctness, attribution=attribution,
                   owner_metadata=metadata, kernel_resources=resources, memory=memory)
    (directory / "attribution_resources.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def cell(x):
        if x is None:
            return "N/A"
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x).replace("|", "\\|").replace("\n", " ")

    def render(title, columns, rows):
        text = [f"### {title}", "", "| " + " | ".join(title for _, title in columns) + " |",
                "|" + "|".join("---" for _ in columns) + "|"]
        for r in rows:
            text.append("| " + " | ".join(cell(r.get(key)) for key, _ in columns) + " |")
        if not rows:
            text.append("| " + " | ".join(["未采集"] + ["—"] * (len(columns) - 1)) + " |")
        text.append("")
        return text

    common = [("version", "版本"), ("frame", "frame"), ("profile_sample", "profile"),
              ("op", "算子"), ("stage", "阶段")]
    lines = ["## 4. 优化归因表", "",
             "逐 profile 样本列出；不同样本不是同一次执行，不叠加。每个 GPU 活动仅计入一个 component。",
             "zeroing 需要 CPU zeros/zero_ 调用证据；无法确认填充值时标为 fill_value_unknown 或 memset_value_unknown。",
             "inputs / weights_partial / reduce 为拆分融合 kernel；weights_unsplit 为未拆分权重 kernel。",
             "原始 Torch 不强行拆成 inputs/weights，无法可靠分类的计算保留在 other_torch_runtime。", ""]
    lines += render("GPU 互斥分项", common + [("component", "分项"), ("factory", "factory"),
                    ("gpu_ms", "GPU ms"), ("kernel_launches", "kernel launch 数"),
                    ("memcpy_activities", "拷贝活动数"), ("memset_activities", "memset 活动数")],
                    [r for r in attribution if r["op"] != "other"])
    lines += render("owner 元数据完整范围（交叉统计，不重复相加）", common + [
                    ("calls", "调用数"), ("cpu_ms", "CPU 范围 ms"),
                    ("hit_calls", "缓存命中数"), ("hit_cpu_ms", "命中 CPU ms"),
                    ("miss_calls", "构建/未命中数"), ("miss_cpu_ms", "构建/未命中 CPU ms"),
                    ("unknown_calls", "缓存状态未知数"),
                    ("inclusive_gpu_ms", "范围内 GPU ms")], metadata)
    lines.extend(["元数据 CPU 范围包括缓存查询、未命中时的构建和同步；不是纯构建时间。",
                  "GPU 交叉统计已包含在上表对应 factory 的分项中，不要再次加到总耗时。", "",
                  "## 5. 正确性与资源表", ""])
    lines += render("正确性状态", [("version", "版本"), ("frame", "frame"), ("status", "正确性"),
                    ("max_abs", "最大绝对误差"), ("relative_l2", "相对 L2"),
                    ("runtime_error", "运行错误（N/A 为未记录）")], correctness)
    lines.extend(["误差未计算，N/A 不代表误差为零。", ""])
    mem_rows = []
    for r in memory:
        converted = dict(r)
        for key in ("baseline_allocated_bytes", "peak_allocated_bytes", "increment_allocated_bytes",
                    "baseline_reserved_bytes", "peak_reserved_bytes"):
            converted[key] = r[key] / 2**20
        mem_rows.append(converted)
    lines += render("独立显存采样（MiB，整个阶段）", [("version", "版本"), ("frame", "frame"),
                    ("stage", "完整阶段"), ("baseline_allocated_bytes", "起始 allocated"),
                    ("peak_allocated_bytes", "峰值 allocated"), ("increment_allocated_bytes", "峰值增量"),
                    ("baseline_reserved_bytes", "起始 reserved"), ("peak_reserved_bytes", "峰值 reserved")], mem_rows)
    lines.extend(["使用 PyTorch CUDA allocator 统计；不包含所有 CUDA context/第三方直接分配。",
                  "reserved 受缓存历史影响；阶段峰值不是单算子峰值，阶段间不可相加。", ""])
    lines += render("各 kernel 的资源与 launch 数（不同配置分行）", common + [
                    ("factory", "factory"), ("kernel", "kernel"), ("component", "分项"),
                    ("kernel_launches", "launch 数"), ("gpu_ms", "GPU ms"),
                    ("shared_bytes_per_block", "shared bytes/block"),
                    ("registers_per_thread", "registers/thread"), ("grid", "grid"), ("block", "block")],
                    [r for r in resources if r["op"] != "other"])
    lines.extend(["资源字段从 GPU trace 原样提取，缺失为 N/A；0 是已记录的零。",
                  "不跨 kernel 相加 shared memory 或寄存器；未将估算 occupancy 当作硬件实测值。",
                  "launch 数按 GPU kernel 活动计数，拷贝/memset 分开，不能等同所有 CUDA API 调用次数。",
                  "完整未归属分项也保存在 attribution_resources.json。", ""])
    return lines, payload
