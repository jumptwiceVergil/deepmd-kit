# 真实训练图计时 v2

替代 `benchmark_repflow_full_calls.py` 的图封装计时。旧版文件保留，但其墙钟
结果不适合作为真实训练加速结论。

## 运行

将 `benchmark_repflow_training_v2.py`、`repflow_trace_v2.py` 复制到服务器已有
benchmark 脚本同目录。依赖已有 `benchmark_repflow_training.py`、
`repflow_bench_adapters.py`、`repflow_bench_legacy.py`、`repflow_bench_report.py`。
不依赖 `repflow_full_timing.py`，不修改生产代码。

从原训练工作目录运行：

```bash
python /usr/local/lib/python3.10/dist-packages/deepmd/pt/model/descriptor/benchmark_repflow_training_v2.py \
  --input input_finetune_1.json \
  --finetune DPA-3.2-5M.pt --model-branch OMat24 \
  --warmup 3 --repeat 20 --profile-repeat 1 \
  --out repflow_training_v2
```

先检查运行可使用 `--warmup 1 --repeat 2 --profile-repeat 1`。
沿用旧入口的 CLI，`--atol/--rtol` 在此版本不参与计算；正确性始终 SKIPPED。
目前遇到运行错误会保存部分结果并退出（不使用旧入口的 continue-on-error 行为）。

## 与上一版的区别

- 没有 `_Boundary`、`_VJP` 或局部嵌套 autograd.grad。
- 坐标 grad 直接调用真实 task_deriv_one 请求，保持 inputs、create_graph 和
  retain_graph 原样；loss.backward 直接运行原图。
- 不分离算子输入，不改变 owner/index 身份和缓存语义。
- 计时 pass 禁用 profiler、逐算子 hooks、索引 CPU 拷贝。只在最外层阶段
  起止同步，不对每一个 sym/edge/angle 调用同步。
- 四个指标使用四次独立的新训练图，避免在 full_step 中插入内部同步。

完整阶段指标：

1. `full_step`：wrapper + loss.backward，不含数据准备、zero_grad、optimizer.step。
2. `train_forward_including_force`：整个 wrapper，包含坐标求导及 loss 构建，
   不能称为单纯能量 forward。
3. `force_autograd`：完整真实坐标求导，包含所有相关算子，不是某个单独算子。
4. `loss_backward`：完整训练 loss.backward，包含一阶和二阶路径。

阶段墙钟包含调用、CPU/GPU 调度、分配、等待及结束同步的开销；开始前的
同步不计入。CUDA Event 值在当前 stream 上测量，若有多 stream，优先解释
等待整个设备完成的墙钟值。不同指标不可相减来估算纯 forward，也不可相加。

## 独立 profiler

在额外的新图上启用原 benchmark 的 forward 标记和 autograd 节点 hooks。
这些 hooks 只用于归属，不封装或重建求导图，不参与完整阶段计时。
输出 `trace_v2_*.json`，使用实际 launch correlation 与 CPU 嵌套范围归属，
不只依赖 External id。原生二阶路径通过 sequence 关联，跨线程同序列出现
多重归属时标为歧义，不猜测。

`REPORT.md`：完整阶段表、算子 GPU 求和表、诊断表。
`results.json`：完整采样、factory 分解、CPU 标记时长、GPU 首末时间戳。
原生二阶 CPU 完整调用不存在单一函数范围，不能将节点时长或首末跨度冒充它。
GPU 首末跨度可夹杂其他算子；未归属活动还包含模型其他部分。缺失不是零。

所有版本共用加载的模型权重与批次，每次采样重新建图，无 optimizer 更新。
旧版仍是已有 `legacy_rebuilt`，并非任意历史版本；v2_materialized 仍为
当前 v2 但物化缺失梯度的对照。所有耗时比均未经正确性验证。
