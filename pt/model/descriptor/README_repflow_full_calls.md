# RepFlow 完整调用计时

## 运行

在原来的 GH100 训练环境，把以下新增文件放在已有 benchmark 脚本同目录：

- `benchmark_repflow_full_calls.py`
- `repflow_full_timing.py`

仍依赖已有 `benchmark_repflow_training.py`、`repflow_bench_adapters.py`、
`repflow_bench_legacy.py` 和 `repflow_bench_report.py`。不修改生产算子。

从原先运行训练命令的目录执行（将脚本路径替换为服务器实际路径）：

```bash
python /usr/local/lib/python3.10/dist-packages/deepmd/pt/model/descriptor/benchmark_repflow_full_calls.py \
  --input input_finetune_1.json \
  --finetune DPA-3.2-5M.pt \
  --model-branch OMat24 \
  --warmup 3 --repeat 20 --profile-repeat 1 \
  --out repflow_full_calls
```

先做运行检查可以用 `--warmup 1 --repeat 2 --profile-repeat 0`。
正确性验证始终跳过，报告明确记录 `SKIPPED`，不是 `PASS`。
编译、运行和缺失计时范围仍会报错；异常前的结果也会写入报告。

## 三种时间不能混淆

1. `wall_ms`：每个完整算子调用前同步 GPU，再开始计时，调用返回后等待
   GPU 完成才结束。包括范围内 Python/C++ 调度、分配、补零、kernel launch、
   执行及等待；包含计时事件和结束同步的额外开销。开始前同步不计入。
2. `cuda_ms`：完整范围两端 CUDA Event 的区间，不等于 kernel duration 求和。
   若算子使用其他 stream，此值不保证覆盖那些 stream 的全部工作；墙钟使用
   device synchronize 等待所有 stream。
3. `kernel_ms`：单独 profiler pass 的实际 GPU 活动时长之和，按 correlation
   找到真实 CPU launch，再匹配完整调用范围，包含范围内辅助 kernel、拷贝和清零。
   profiler 开销不混入前两项计时。

`REPORT.md` 有 forward/backward/double backward 三张表。每行统计一个 frame
内某类算子的实际全部调用之和，再对重复采样取中位数，不使用单层时间乘 24。
`results.json` 保留每次调用、每次采样、调用数和单独的 ordinary backward。
正确性未验证时，耗时比只是探索性指标。

## 原始求导路径与测量边界

外层继续调用真实训练 wrapper。`task_deriv_one` 的 inputs 仍是实际
`extended_coord`，仍使用 `create_graph=True, retain_graph=True`；二阶阶段由
真实训练 loss 的 `loss.backward()` 触发。没有替换成手写一阶/二阶公式。

为了把原始 Torch 分散的 autograd 节点作为完整调用计时，脚本在内存中加入
benchmark-only graph boundary：使用局部 leaf view 运行所选原始/融合函数，
由局部 `autograd.grad` 求 VJP，并将导数连接回外层训练图。其二阶计算仍然
由 autograd 对该一阶导图求导。训练坐标求导时，仅请求能到达实际坐标的输入
梯度，不把参数一阶梯度强行纳入 force 计算。参数仍保留在二阶导图中。

因此这些是**带独立 autograd 边界的、同步隔离的算子完整调用延迟**，不是
完全无插桩的训练延迟。额外局部 autograd 调度和同步会改变性能，尤其是很小的
算子；不能将三张表简单相加推断生产训练吞吐量。forward 的局部 leaf view 和
代理参数准备位于计时边界外，原函数内的 split/reshape/gather/分配均在边界内。
该封装只支持本任务所需的一阶及二阶求导，不用于生产训练或三阶导测试。

每次重复重新建立训练图；版本之间恢复相同权重并使用同一批数据，不执行
optimizer.step，不写训练 checkpoint，不改输入 JSON。配置中的动态邻居规模
来自实际数据和预训练模型，而不是把 e_sel/a_sel 当作真实边/角数量。

`legacy_rebuilt` 沿用已有重建旧版，不能当作任意历史提交的精确性能。
`v2_materialized` 使用当前 v2 算子，但把缺失二阶上游梯度物化为零张量；
`current` 保留 None，并使用现有编译期标志跳过对应计算。

## 本机验证

```bash
python -m unittest test_repflow_full_timing -v
```

CPU 测试只验证计时框架的导数连接、未使用输入和报告行为，不验证 TileLang
算子，也不能替代 GH100 的运行测试。
