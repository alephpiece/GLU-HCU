# 性能测试

`glu_bench` 测量重复数值分解，与 `lu_cmd` 共用 GPU 执行实现，支持 CUDA/HIP 和所选构建精度。它每轮恢复模板默认初值，不启用扰动。构建见[构建与验证](../BUILDING.md)，文件格式见[矩阵输入](solver.md#矩阵输入)。

参数概要可用 `glu_bench --help` 查看。命令示例：

```bash
# 单任务
/path/to/build/glu_bench -i /path/to/matrix.mtx --output single.json

# 同一模板的 12 个独立任务，每批最多 4 个副本
/path/to/build/glu_bench -i /path/to/matrix.mtx \
  --tasks 12 --batch-size 4 --updates mapped --graph --output repeated.json

# 三模板循环分配：3、3、2 个任务
/path/to/build/glu_bench \
  -i /path/to/matrix-a.mtx -i /path/to/matrix-b.mtx -i /path/to/matrix-c.mtx \
  --tasks 8 --output multiple.json
```

## 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `-i PATH` | 与 `--workload` 二选一，可重复 | 按给定顺序组成模板列表，任务循环引用它们 |
| `--workload FILE` | 与 `-i` 二选一 | 从显式清单读取模板与有序任务，不能同时指定 `--tasks` |
| `--tasks N` | `-i` 模板数 | 仅用于 `-i` 模式，独立任务总数必须至少覆盖全部模板 |
| `--warmup N` | 5 | 预热的完整 LOOP 数，允许 0 |
| `--loops N` | 30 | 计时的完整 LOOP 数，必须大于 0 |
| `--batch-size N` | 1 | 同模板独立副本的每批上限，每批一条 stream，尾批保留实际数量 |
| `--graph` | 关闭 | 准备阶段捕获每批数值与输出图，随后重放 |
| `--snapshot-policy auto\|per-copy\|shared\|device` | `auto` | 自动选择初值恢复方式，或强制逐副本上传、共享上传、设备初值恢复 |
| `--updates scan\|mapped\|panel\|tiled\|fused` | `scan` | 选择扫描、位置映射、依赖窗口目标归约、局部密集块或块内分解 |
| `--submit-threads N` | 1 | 并行提交数值与输出的主机线程数上限，实际不超过批数。输入由主线程提前排队 |
| `--tmp-slots N` | 32 | scan 每个副本的 scratch 槽数上限，其它策略不使用 scratch |
| `--check full\|basic` | `full` | 完整因子残差检查，或轻量有限性检查 |
| `--diagnostics` | 关闭 | 输出提交与等待的墙钟、线程 CPU 时间、kernel 次数和传输量 |
| `--output PATH` | 无 | 成功后写 JSON，程序同时在标准输出打印日志和结果 |

任务数、批量上限、线程数和槽数均须为正整数。一个逻辑任务对应一份矩阵，同模板任务可以合为一批，共用 kernel 执行链，各自保留独立数值区和因子输出。每个模板只预处理和准备一次，同模板副本每轮完整恢复相同默认初值。

## 显式任务清单

需要不均匀的模板频率或指定逻辑任务顺序时，使用 `--workload FILE` 替代 `-i` 和 `--tasks`：

```text
glu-workload 1
matrix light "inputs/light.mtx"
matrix heavy "inputs/heavy.mtx"
task light
task heavy
task light
```

```bash
/path/to/build/glu_bench --workload /path/to/case.workload \
  --batch-size 4 --updates mapped --graph --output case.json
```

首条有效记录必须是版本头，随后定义全部 `matrix ID PATH`，再列出 `task ID`。ID 必须唯一、先定义后引用，全部模板都必须被使用。每个 ID 只准备一次模板，不按文件路径或内容自动合并不同 ID。

路径相对于清单目录解析，也可使用绝对路径。包含空格时用双引号，双引号和反斜杠可用反斜杠转义。允许空行和以 `#` 开头的整行注释。未知记录、额外字段、无效 ID 及与 `-i` / `--tasks` 混用均报错。

清单顺序表示逻辑任务，不表示依赖或到达时间。`--batch-size 1` 按该顺序组织任务，更大批量按模板定义顺序交错分批。并行提交时，实际开始提交的次序可以变化。

## 执行组织与资源

每批使用独立 stream，按 `输入恢复与布局准备 → 数值分解 → 完整因子回传` 执行，全部批次完成后开始下一轮。主线程先将所有批次的输入排队，各流保持自身依赖。`--graph` 只重放每批的数值与输出操作，不启用时直接提交。Graph 每批一次 D2H，直接提交则逐副本 D2H。

`--submit-threads` 大于 1 时，工作线程并行提交独立批次，只有一批时仍使用主线程。工作线程每轮创建和汇合，相关成本包含在 LOOP 中。主机线程数和逻辑 stream 数都不等于实际 GPU 并发度。

`--updates` 显式选择数值更新策略。`scan` 使用独立 scratch，其余策略准备模板共享索引或计划。`fused` 每矩阵使用一个线程块完成全部依赖层，仍按本手册的同模板批次提交。不同结构的集合执行使用 [GpuTaskSet](numeric-api.md#异构任务集合)。各策略的工作方式与准备成本见[数值更新策略](numeric-api.md#数值更新策略)。

`--snapshot-policy` 选择默认初值恢复方式。默认 `auto` 对 panel 多副本使用 `shared`，其余使用 `per-copy`，尾批按实际副本数解析。`device` 需显式指定，并额外保留一份模板设备初值。完整规则见[默认初值恢复](numeric-api.md#默认初值恢复)。

按实际复用轮数比较含准备的完成时间。批内副本数影响并行规模，跨轮复用影响准备成本的摊销。自定义新值输入和流水线使用[C++ 数值任务接口](numeric-api.md#数值输入)。

程序对显存容量与执行网格做预检，超限时报错，任务、批量和槽数由调用者调整。预检统计显式设备缓冲，运行时资源使用额外容量，固定页主机缓冲另行分配。

## 计时与结果字段

一次 LOOP 包含全部任务的输入恢复、数值提交与执行、D2H、线程创建与汇合及轮末等待。预处理、资源和 Graph 准备、数值检查、solve、输出文件和释放资源不计入 LOOP。

| JSON 字段 | 含义 |
| --- | --- |
| `setup_ms` / `graph_setup_ms` | 准备总耗时及其中的 Graph 捕获与实例化耗时，准备时不执行分解 |
| `update_map_setup_ms` | mapped 路径的映射及辅助元数据准备耗时，包含在 `setup_ms` 中 |
| `update_plan_bytes` / `update_plan_setup_ms` | panel/tiled/fused 的设备计划载荷与构造上传时间，包含在共享载荷和 `setup_ms` 中 |
| `native_mmac` | 至少一个模板具有使用原生矩阵指令的密集 tile。标量回退或无 tile 时为 false |
| `warmup_ms` | 全部预热耗时，包含可能的首次重放开销 |
| `loop_ms` / `ms_per_loop` | 主机单调时钟测量的逐轮完整耗时及算术平均值 |
| `factor_output_ready_ms` | 从预处理前至最后一轮全部因子可读的实际墙钟，包含准备、预热与计时轮，检查、solve、报告与释放在外 |
| `validation_ms` | 计时结束后检查全部任务的耗时 |
| `task_device_bytes` / `shared_device_bytes` | 任务缓冲及模板共享设备缓冲载荷，后者包含位置映射、更新计划及 Device 策略的只读初值 |
| `snapshot_policy` / `snapshot_setup_ms` | 请求的初值恢复策略及设备初值准备耗时，后者已计入 `setup_ms` |
| `batch_snapshot_policies` | 按批顺序记录实际恢复策略，Auto 的单副本尾批可与前面批次不同 |
| `update_map_bytes` | 共享设备缓冲中的位置映射载荷 |
| `pinned_bytes` | 模板共享初值与各独立副本因子输出的 pinned host 载荷 |
| `templates[].effective_tmp_slots` | 实际 scratch 槽数，其它策略为 0 |
| `task_templates` | 按逻辑任务顺序记录的零起始模板索引 |
| `batch_templates` / `batch_sizes` / `ordering` | 各批的模板、实际副本数与提交策略。并行时不表示实际提交时序 |

比较 N 个完整轮次的含准备时间时，使用 `--warmup 0 --loops N`，读取 `factor_output_ready_ms`，每轮仍完成全部逻辑任务。该字段直接测量总时间，也计入各轮之间的主机开销。

容量字段统计显式缓冲区载荷，allocator、stream、Graph 和运行时内部分配另计。报告还记录后端、精度、设备、检查模式和实际任务覆盖。

## 诊断字段

启用 `--diagnostics` 后另报：

| JSON 字段 | 含义 |
| --- | --- |
| `submit_ms` / `drain_ms` | 提交全部任务及提交返回后的等待时间，按主机墙钟分段。提交可含运行时等待，设备执行可与提交重叠 |
| `submit_thread_cpu_ms` / `drain_thread_cpu_ms` | 主线程在提交及等待阶段的 CPU 时间，分别统计，不含其他线程的 CPU 时间 |
| `submit_workers_cpu_ms` | 全部提交工作线程的 CPU 时间之和，可大于提交墙钟。单线程时为 0 |
| `submit_worker_max_wall_ms` | 最慢提交工作线程的墙钟耗时。单线程时为 0 |
| `launches_per_loop` | 每轮数值与输出布局转换 kernel 次数，Graph 重放时也统计底层 kernel，区别于主机 launch API 次数 |
| `input_restore_launches_per_loop` | 每轮快照广播、设备恢复及输入布局转换 kernel 次数，所有输入布局准备均在数值 Graph 之外 |
| `wide_blocks_per_loop` | scan/mapped 每轮宽层数值 kernel 的线程块总数，包含适用路径的归一化与目标更新。panel/tiled/fused 返回 0 表示不适用 |
| `template_work` | 每模板的层宽及 L 非空源列的结构目标数量，各批实际启动次数另计。panel/tiled/fused 不适用的层内启动字段为 null |
| `h2d_bytes_per_loop` / `d2h_bytes_per_loop` | 每轮所选快照策略的实际传输载荷，完整因子 D2H 按全部逻辑副本统计 |
| `h2d_copy_ops_per_loop` / `d2h_copy_ops_per_loop` | 每轮安排的 H2D/D2H 操作次数 |

每轮 `submit_ms + drain_ms = loop_ms`。主线程与工作线程的 CPU 时间分别统计，需结合提交墙钟评价并行效果。诊断统计覆盖循环执行，准备阶段与统计输出在计时区间外。

## 结果检查

默认 `--check full` 在计时后检查最后一轮的全部任务：因子有限性、非零主元、全 1 RHS 的解有限性，以及预处理矩阵的完整因子残差 `||B−LU||∞ / ||B||∞`。FP64 上限为 `1e-10`，FP32 为 `1e-3`。因子残差评价预处理矩阵的分解质量，原系统解误差单独评价。检查耗时单列为 `validation_ms`。

`--check basic` 保留全部任务的因子有限性、非零主元和解有限性检查，跳过因子残差，残差及容差字段输出为 `null`。`status=ok` 表示所选检查通过，检查模式记录于 `check_mode` 和 `validation` 字段。两种模式都只检查最后一轮，计时中检查提交与等待的 GPU 错误。

GPU、检查或输出失败均以非零状态退出。为每次运行指定独立输出路径，避免误读先前的成功文件。
