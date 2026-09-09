# RepFlow 真实训练图基准

本目录新增文件组成独立基准。不修改现有 Python 源码、原配置、checkpoint 或数据。
运行期间仅在当前进程替换方法，退出时恢复。不开启正式训练，不调用 optimizer.step，
不保存训练 checkpoint，不执行梯度裁剪。临时配置和统计文件副本位于新建的运行目录。

## 远程运行

将以下文件放在同一目录，使用安装了当前 DeePMD/TileLang 的 GH100 Python 环境：

- benchmark_repflow_training.py
- repflow_bench_adapters.py
- repflow_bench_legacy.py
- repflow_bench_report.py

不要求服务器上有 Git；旧实现已随文件一起保存。

```bash
python /path/to/benchmark_repflow_training.py \
  --input input_finetune_1.json \
  --finetune DPA-3.2-5M.pt \
  --model-branch OMat24 \
  --out repflow_benchmark \
  --frames 1 --warmup 3 --repeat 20 --profile-repeat 2
```

确认流程、编译与正确性后，可增加到 --frames 3 --warmup 10 --repeat 100。
首次 JIT 可能耗时很长。--warmup 是基准预热，与 training.warmup_steps 无关。
请保持单进程单 GPU，不使用 torchrun/DDP；必须是 eager DeePMD（非 TorchScript）。
建议先确保没有其他进程占用同一 GPU。

--input 是你现有的训练 JSON 输入文件，不是输出报告。所有相对数据/统计路径与原
dp 启动命令一样，按照启动时的工作目录解析。checkpoint 必须是可信的训练文件。
基准目录每次新建 run_日期_时间 子目录，不覆盖之前的报告。
若原统计文件存在，会复制到运行目录，再交给训练初始化使用；不存在时只在运行目录创建。

默认执行所有版本。快速检查可使用 --versions original current。
原始版本始终运行，用于建立正确性基线。
--continue-on-error 允许某个融合版本编译/执行失败后尝试其余版本；
原始版本失败则停止。正确性失败仍记录时间，但不填有效加速比，最终返回非零退出码。

## 实际求导路径

初始化沿用 deepmd.pt.entrypoints.main.train，包含 finetune 分支选择、配置标准化、
邻居统计以及 Trainer 的模型/损失初始化。仅将 Trainer.run 替换为基准流程。
每个版本使用同一组真实采样 batch、同一份加载后的模型状态和随机种子。
默认 --step 0，按该训练步的学习率计算损失权重，不进行参数更新。
当前只支持本次提供的单任务、warmup_steps=0 配置。

模型内部 task_deriv_one 不改写，仍实际执行：

```python
torch.autograd.grad(
    [energy], [extended_coord],
    grad_outputs=[torch.ones_like(energy)],
    create_graph=True, retain_graph=True,
)[0]
```

返回梯度取负得到力，之后真实训练 wrapper 构造包含能量/力/virial 的损失，
由 loss.backward() 进行反向。绝不以手写 PyTorch 二阶公式替代原始基线。
每次重新构建图，清理 parameter.grad，图构建不计入 loss.backward 时间。

原始基线直接调用 RepFlowLayer 的：

- symmetrization_op_dynamic
- optim_edge_update_dynamic
- optim_angle_update_dynamic

保留其原有 matmul、index_select 顺序。无论目前源码调用的是普通方法还是
fused_ 方法，基准均会将两种入口路由到同一被测版本，避免漏测混合调用路径。
不硬编码边数、angle 数或层数；记录实际加载后的模型和每次调用的真实形状/index。

## 被测版本

| 名称 | 一阶 backward | double backward | 缺失上游梯度 |
|---|---|---|---|
| original | 原始函数产生的 autograd 图 | 上述图经过 loss.backward | PyTorch 原生处理 |
| legacy_rebuilt | 历史 tiled 融合、weights 非 split 版本 | 历史 tiled inputs + 非 split weights | 补零 |
| v2_materialized | 当前 inputs/weights v2 | 当前 inputs/weights v2 | 强制物化零梯度，触发完整分支 |
| current | 当前代码 | 当前代码 | 使用现有 None/HAS_* 优化 |

legacy_rebuilt 来源：6ff0569062f8c920507137524dd1ba42e5ecc555。
从该提交提取可达的函数/类，去掉不执行的注释/字符串块；为正确比较，修正
fused_edge_update_input_backward_v1 的 bias_tile 在每个 K tile 内清零。
因此它是“重建并修正正确性的旧算法基线”，不是该提交的逐字/逐周期历史快照。
该版本已包含 Sym 的通用 owner 支持；不将不适用的 uniform-only 实现强行用于无序 owner。
Sym 在旧基线与当前版之间可能没有明显差别，这是合理结果。

所有版本 forward 只比较实际存在的实现，不凭空伪造不存在的 forward v2。
v2_materialized 和 current 的差别用于衡量 None/HAS_* 优化。
legacy_rebuilt 到 v2_materialized 的差别包含 inputs/weights 等中间改进，
不能单凭这一差值宣称全部收益都来自 weights split。

## 三张结果表

REPORT.md 包含三张汇总表；CSV 包含完整逐层、逐调用的数据：

1. **01_stage_times.csv**
   - 真实模型/损失调用（包含内部求力）的 CUDA-event 时间；
   - task_deriv_one 内 autograd.grad 的 CUDA-event 时间；
   - loss.backward 的 CUDA-event 时间；
   - 对应主机耗时；
   - Sym/Edge/Angle 的 forward、backward、double_backward、ordinary_backward
     的 profiler 归属 GPU activity 时间；
   - 相对 original、legacy_rebuilt 的加速比。
2. **02_attribution.csv**
   - 每层/调用、阶段、kernel factory、实际 GPU kernel/拷贝/清零 activity；
   - 单独展示 weights partial 与 reduce；不遗漏归约和清零；
   - activity 次数及 GPU 时间。
3. **03_correctness_resources.csv**
   - 整个真实图的 prediction（包括 force/virial）、loss 和所有参数梯度误差；
   - 每阶段 baseline allocated、peak allocated、增量显存；
   - 分析采样的 kernel/memset/memcpy 次数；
   - 预热耗时、失败信息；无法自动取得的硬件资源指标标为 N/A。

results.json 保存未汇总数据、逐张量误差、每次调用的形状和 index、
缺失梯度模式、owner 元数据缓存命中/未命中以及 JIT 参数。
effective_input.json 是 finetune/标准化/邻居统计后的配置。
metadata.json 记录实际模型结构、GPU/PyTorch/CUDA、源码 SHA256 和损失调度位置。
sampled_batches.pt 保存本次采样 batch，trace_*.json 可用 Perfetto/Chrome trace 查看。
注意报告中的 batch/index 数据是你的真实训练样本元数据，请按数据权限管理这些文件。

## 时间和归属的严格含义

**不要混淆两个 metric。**

- cuda_event_interval：未开启逐算子 profiler/hooks 的实际训练阶段流区间，
  包含阶段内分配/清零/workspace/归约的 GPU 操作和提交间隙。
- profile_attributed_kernel_sum：独立 profiling pass 中归属于算子的 GPU activity
  时间之和，不包含所有 Python 开销，不等价于整个 autograd 调用延迟。
  profiler 可能扰动执行，只与同样 profiling 的其他版本比较。
- host_wall_including_phase_sync：包含等待该阶段 GPU 完成的主机时间。

总模型/损失调用中已经包含 force_autograd，不能将两者相加。
loss_backward 包含能量的一阶路径和力/virial 的二阶路径，不能整个标作纯二阶时间。
算法内部 double backward 的归属使用 profiler 的 sequence number + forward thread：
第一次反向中创建的操作，其第二次反向归属 double_backward；
原始前向图的再次反向归属 ordinary_backward。
未归属的活动保留为 other/unattributed，不强行分摊给某个算子。
如果没有归属到任何 double_backward，会在 results.json 发出 attribution_warning；
应检查 trace，而不是把缺失项当作零耗时。

每层总计是实际调用的求和，不是“单层中位数 × 24”。多个 frame 分开报告，
避免把不同邻居/angle 数的样本混在一个加速比中。
资源统计使用独立 pass；force、wrapper、loss 的 peak 不互相混用。
预热耗时可能包含 JIT，但不能视为严格的冷编译耗时，因为编译缓存可能已存在。

## 数值精度

按用户要求，默认绝对容差调整为 atol=1e-3，相对容差保持 rtol=2e-3。
实际判据为 abs(actual-reference) <= atol + rtol * abs(reference)。
可用 --atol/--rtol 覆盖；运行采用的容差记录在 metadata.json 中。
放宽容差不改变计算结果，也不证明误差来源或训练精度可接受。
TF32-RZ 是此前已知风险，当前 full-model 回归仍可能失败。
--disable-tf32 只控制 PyTorch/cuBLAS 设置，不会把 TileLang T.gemm 自动变成全精度 FP32。
需要同时查看 max_abs、relative_l2 和 per_tensor，区分数值问题与索引/求导错误。
正确性失败的版本保留原始耗时，但不生成有效 speedup。

## 更多硬件指标

寄存器/线程、shared memory/block、achieved occupancy、SM 利用率和实际 DRAM 流量
需要在独立 Nsight Compute 采样中获取。本脚本不猜测这些指标，不把 block 数当成
achieved occupancy；默认标记 N/A。
可从第二张表挑选代表性的 kernel，在预热后的独立运行中做 Nsight Compute 分析，
并将资源数据按 kernel 和真实形状对齐。不要用重型 profiler 的整段耗时替代正常计时。

## 本机验证

```bash
python test_repflow_bench_harness.py
```

CPU 测试覆盖真实 autograd 的 sequence 归属、None/zero 比较、三表输出、
配置/统计副本隔离，以及重建 edge v1 在 K=65 时的 bias 和 scatter 数学。
本机不具备 GH100、TileLang、真实数据和 checkpoint；完整 finetune、GPU 编译、
GPU profiler 归属与性能必须在远程验证，不能将 CPU 测试通过等同于 GPU 验证通过。
