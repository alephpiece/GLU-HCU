# 构建与运行

GLU 使用同一份源码构建 CUDA 或原生 HIP 版本，生成求解程序 `lu_cmd` 和性能测试程序 `glu_bench`。默认计算精度为 FP64。

## 环境要求

- Linux x86_64，支持 C++17 的 C/C++ 编译工具链。
- CMake 3.21 或更高版本，以及 make 或 Ninja。
- CUDA 后端需要 CUDA Toolkit，`nvcc` 应在 `PATH` 中。
- HIP 后端需要原生 HIP 工具链。下文以安装在 `/opt/dtk` 的 DTK 26.04 为例。

NICSLU 和预处理代码随项目一起编译，无需单独安装。CMake 构建不包含 Python 扩展。

## CMake 构建

将示例中的源码和构建路径替换为实际路径。构建目录放在源码目录之外，不同后端、架构或精度使用独立目录。

### CUDA

以下配置适用于 H100，目标架构为 `sm_90`：

```bash
cmake -S /path/to/GLU_public -B /path/to/build-cuda \
  -DGLU_BACKEND=CUDA \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build /path/to/build-cuda -j64
```

使用其他 GPU 时调整 `CMAKE_CUDA_ARCHITECTURES`。FP64 路径使用原生双精度原子加法，CUDA 目标架构需为 `sm_60` 或更高。

### HIP

以下配置适用于 BW150，目标架构为 `gfx936`：

```bash
cmake -S /path/to/GLU_public -B /path/to/build-hip \
  -DGLU_BACKEND=HIP \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/dtk \
  -DCMAKE_HIP_ARCHITECTURES=gfx936
cmake --build /path/to/build-hip -j64
```

此入口使用原生 HIP 编译器和运行时，无需加载 DTK 的 CUDA 兼容环境。DTK 安装路径变化时，同时调整编译器和工具链根目录。构建配置已包含 DTK 所需的 1024 线程 block 编译选项。

### 配置选项

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `GLU_BACKEND` | `CUDA` | 选择 `CUDA` 或 `HIP` 后端 |
| `GLU_USE_FP32` | `OFF` | 设为 `ON` 时使用 FP32，否则使用 FP64 |
| `GLU_BUILD_TESTS` | `OFF` | 设为 `ON` 时构建 GPU kernel 与主机调度回归测试 |

例如，在新的构建目录配置时加入 `-DGLU_USE_FP32=ON`，即可构建 FP32 版本。

## 运行

程序每次读取一个矩阵，求解右端向量全为 1 的线性系统，并将解写入当前目录的 `x.dat`。在独立输出目录运行，以免覆盖已有结果：

```bash
mkdir -p /path/to/output
cd /path/to/output
/path/to/build-hip/lu_cmd -i /path/to/matrix.mtx
```

CUDA 版本使用相应构建目录中的 `lu_cmd`。程序选择第一个可见 GPU，可通过 `CUDA_VISIBLE_DEVICES` 或 `HIP_VISIBLE_DEVICES` 指定设备。

输入采用实数方阵的坐标格式：尺寸行包含行数、列数和非零元素数，随后每行包含从 1 开始的行索引、列索引和值。元素需先按列、再按行排序，并显式提供全部非零元素。读取器不自动展开对称存储，`.mtx` 扩展名本身也不保证满足这些要求。

加入 `-p` 可启用静态主元扰动：

```bash
/path/to/build-hip/lu_cmd -i /path/to/matrix.mtx -p
```

扰动会改变分解，比较结果时应保持该选项一致。程序先检查解是否有限，再写入 `x.dat`，保留所选精度往返转换所需的有效数字。GPU 操作失败、因子出现非有限值或零对角、解出现非有限值、文件打开或写入失败时，程序以非零状态退出。退出码为零表示执行完成，数值正确性仍需通过残差或参考解检查。

## GPU 数值回归测试

在已配置的构建目录启用测试，然后重新构建并运行：

```bash
cmake -S /path/to/GLU_public -B /path/to/build-hip \
  -DGLU_BUILD_TESTS=ON
cmake --build /path/to/build-hip -j64
HIP_VISIBLE_DEVICES=0 ctest --test-dir /path/to/build-hip --output-on-failure
```

CUDA 后端使用对应构建目录和 `CUDA_VISIBLE_DEVICES`。测试需要可用 GPU，包含 kernel、主机调度、批量与图重放回归。检查覆盖稀疏尾部、主元广播、并发列更新、扰动、临时槽不足时的复用、多模板多任务四轮执行、模板寿命、独立副本偏移、批量尾部、图准备与重复重放，以及零主元和非有限因子的拒绝。CTest 还检查 CLI 正常输出、文件打开及写入失败，以及无可见 GPU 时的错误返回。

## CUDA Makefile 入口

`src/Makefile` 提供 CUDA 构建入口，默认使用 FP64，目标架构为 `sm_60`。以下命令以 H100 为例：

```bash
cd /path/to/GLU_public/src
make clean
make -C nicslu/lib -j64
make -C nicslu/util -j64
make -j64 CUDA_ARCH=sm_90
```

生成的程序为 `src/lu_cmd`。修改目标架构或编译参数后，应先清理旧构建产物再重新编译。此入口在源码目录内生成对象文件，可在源码副本中使用。

## 多矩阵性能测试

`glu_bench` 与 `lu_cmd` 共用 GPU 执行实现，支持 CUDA/HIP 和所选构建精度。
性能基线使用默认 FP64，benchmark 不启用扰动。命令示例：

```bash
# 单任务
/path/to/build/glu_bench -i /path/to/Ud00023.mtx --output single.json

# 单模板 2000 个独立任务
/path/to/build/glu_bench -i /path/to/Ud00023.mtx --tasks 2000 --output repeated.json

# 三模板循环分配：667、667、666 个任务
/path/to/build/glu_bench \
  -i /path/to/Ud00002.mtx -i /path/to/Ud00023.mtx -i /path/to/Ud00800.mtx \
  --tasks 2000 --warmup 5 --loops 30 --tmp-slots 32 --output mixed.json
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `-i PATH` | 与 `--workload` 二选一，可重复 | 按给定顺序组成模板列表，任务循环引用它们 |
| `--workload FILE` | 与 `-i` 二选一 | 从显式清单读取模板与有序任务，不能同时指定 `--tasks` |
| `--tasks N` | `-i` 模板数 | 仅用于 `-i` 模式；独立任务总数必须至少覆盖全部模板 |
| `--warmup N` | 5 | 预热的全局 LOOP 数，允许 0 |
| `--loops N` | 30 | 计时的全局 LOOP 数，必须大于 0 |
| `--batch-size N` | 1 | 同模板独立副本的每批上限，每批一条 stream；必须大于 0 |
| `--graph` | 关闭 | 在准备阶段捕获每批数值与输出图，随后重放 |
| `--updates scan\|mapped` | `scan` | 选择目标列扫描或预计算更新位置。`mapped` 增加每模板共享的准备和显存成本 |
| `--submit-threads N` | 1 | 主机提交线程数上限，必须大于 0。不同批次可由不同线程提交 |
| `--tmp-slots N` | 32 | scan 每任务 scratch 槽数上限，必须大于 0。mapped 不使用 scratch |
| `--check full\|basic` | `full` | 完整残差验收或轻量有限性检查，见下文 |
| `--diagnostics` | 关闭 | 输出提交/尾部等待的墙钟与主线程CPU时间、静态 launch 数及传输字节 |
| `--output PATH` | 无 | 成功后写 JSON；程序同时在标准输出打印日志和结果 |

`mixed-2000` 是上述第三条命令的实验名称，程序没有硬编码场景。三个模板的重复
按相同初值重复执行，不模拟逐轮变化的数值。不同输入可通过重复 `-i` 传入。
加入 `--batch-size 512` 可将同模板副本合并为批，尾批保留实际数量；它不改变逻辑
任务数。批大小与是否使用 `--graph` 应结合负载选择。

### 执行与计时

每个模板只做一次预处理、符号分析和只读索引上传。任务共享模板的只读 pinned
输入和设备结构，每个副本拥有独立数值区和 pinned 输出，scan 另有独立 scratch。
每批使用一条 nonblocking stream。模板实现由任务保持寿命；副本之间不共享输出。批量减少
stream 与分配次数。Mapped 多副本布局还分配一份交错 values，物理行按实际 copy tile 补齐，
预检按各批和尾批准确累计显存。单副本不做布局转换。

每轮先由主线程在各批自己的 stream 上排队全部输入，再按模板交错顺序提交数值与输出。
上传与数值之间没有全局同步，每批的 stream 维护 `H2D 初值 → 数值分解 → D2H 因子` 依赖。
全部批提交后逐流等待完成，然后才能开始下一轮。batch-size=1 按逻辑任务顺序组织批次。
所有 kernel 显式使用所属批的 stream。
`--submit-threads N` 大于 1 时，最多使用 N 个工作线程并行提交独立批次的后续 enqueue。
输入提前排队仍由主线程完成，包含在主线程提交计时中。
实际线程数不超过批数，只有一批时仍由主线程提交。工作线程在每轮创建和汇合，
这些成本包含在 LOOP 中。全部提交线程完成后，主线程再等待各 stream。
此时批次开始提交的次序可以变化，逻辑任务和每批内部依赖保持一致。
Scan 宽层使用 batched RL，按 scratch 槽数分段，小层按列执行 factor/update/clear。
Mapped 不分配 scratch，宽层保留整层多源并行，小层按 source-target 对展开更新。
多副本使用交错 values，使相邻 lanes 访问不同副本的同一符号位置，计时内包含双向转置。
CUDA FP64 的多副本小层先归一化 L，其它 mapped 路径在输出前归一化小层 L，保留除法和 atomic 更新。
Scan 的批内副本数至少 16 时，RL block 最多128线程，单列 kernel 使用256线程。

较小批量沿用原线程规则，尾批按实际副本数选择。
`LUonDevice` 是同一执行接口的单任务封装。性能计时使用 `glu_bench` 的 LOOP 指标；
求解 CLI 的准备与分解耗时不使用同一计时边界。

scan 的 scratch 在准备阶段于任务流清零并等待完成。正常执行中 kernel 清理使用过的槽位，
跨层及跨轮复用。实际槽数是请求上限与模板 batched 层最大宽度的较小值；只有小层时
用 1 槽。32 是可重复的起步配置，不是已调优值。槽数影响分批与提交次数，跨平台比较
必须固定它。mapped 不使用槽数，也不受此参数分段。映射分配前检查其载荷能否放入剩余显存，
随后检查全部任务的数值区和 scan 的 scratch 是否能放入剩余容量。这些检查不能保证驱动资源或 pinned host 分配一定成功。
分配失败明确退出，不自动调整任务或槽数。

`--updates mapped` 在准备阶段保存每个结构更新的目标位置，所有同模板副本共享，
各轮重复使用。它保留归一化、零贡献判断和原子更新，避免在数值阶段反复扫描目标列。
位置数组可能远大于因子本身，因此需要结合模板复用次数和可用显存选择。
`update_map_bytes` 报告位置映射的设备载荷，`update_map_setup_ms` 报告其构建与上传时间，
均包含在相应总量中。`shared_device_bytes` 为模板共享的结构与映射缓冲区，
与 `task_device_bytes` 分开报告，两者不包括驱动和 Graph 内部资源。
`templates[].effective_tmp_slots` 在 mapped 下为 0，表示不使用 scratch。

`setup_ms` 包含预处理、资源分配、初始化和可选的图准备；`graph_setup_ms` 单列其中的
捕获与实例化耗时。图捕获包含数值和批量 D2H，H2D 在每轮的 Graph 外排队；准备时不执行分解。
`warmup_ms` 单列全部预热时间，包括可能的首次重放开销。`loop_ms` 使用主机单调时钟，包含 H2D、
计算提交、GPU 执行、D2H 和轮末等待；`ms_per_loop` 是这些完整轮次的算术平均。
输出文件、检查、释放不在该区间。逻辑 stream 数不等于实际同时执行的任务数。

同一配置使用独立进程重复，结合逐轮时间及进程间均值差异判断稳定性。预热、计时轮数
和重复次数按单轮成本及待判断的效应大小选择；比较配置保持匹配。正式计时不挂 profiler。

`--diagnostics` 额外输出 `submit_ms`（提交全部任务）和 `drain_ms`（随后逐流等待）；
两者之和为 `loop_ms`。提交可能包含运行时背压，不能直接解释为纯 CPU 时间；尾部等待
也不是全部 GPU 时间。静态 launch 数表示底层 kernel 执行次数，按实际批数统计；图重放时不等于主机
launch API 调用次数。传输字节仍按逻辑副本数统计。统计不包含准备阶段，且在计时区间外。

诊断还输出逐轮 `submit_thread_cpu_ms` 和 `drain_thread_cpu_ms`，使用主提交线程的CPU时钟，
分别对应提交与尾部等待阶段。CPU时间包含运行时执行或忙等，不覆盖其他线程；
它与墙钟的差值也不能单独区分睡眠、抢占或辅助线程代办。读取CPU时钟失败会报错。
并行提交时另报 `submit_workers_cpu_ms`，表示全部提交工作线程的 CPU 时间之和，
以及 `submit_worker_max_wall_ms`，表示该轮最慢工作线程的提交墙钟时间。
工作线程采样覆盖设备选择、任务领取和 enqueue，线程创建与汇合的全部成本由
LOOP 和提交墙钟覆盖。主线程 CPU 字段仍只覆盖主线程，工作线程 CPU 之和可以大于
提交墙钟。实际只有一个提交线程时走主线程，两个工作线程字段均为 0。

### 正确性与结果

默认 `--check full` 在计时后检查**最后一轮的全部任务**：非有限因子、零主元、全 1 RHS 的解是否有限，
以及预处理矩阵 `B` 的完整 `||B−LU||∞ / ||B||∞`。残差在主机按稀疏列乘积用
`long double` 累积，包含乘积落在 B 稀疏模式之外的项，不以抽样或校验和代替。
FP64 验收上限为 `1e-10`，FP32 为 `1e-3`；这检验分解质量，不保证病态原问题的
前向精度，也不是区间算术认证。检查全部副本可能耗时较长，单列为 `validation_ms`。
计时中只检查提交与等待的 GPU 错误；最后一轮检查不等于逐轮数值检查，重复复位由
多轮已知因子回归另行覆盖。

`--check basic` 仍检查最后一轮全部任务的有限因子、非零主元和有限解，但跳过完整
分解残差。适合已完成完整验收后的日常性能诊断；改变数值执行路径或出现可疑结果时
应补做 full。basic 的残差及容差输出为 `null`，`status=ok` 仅表示所选检查通过。

JSON 记录检查模式、后端、精度、设备、调度方式、模板与任务数、请求及实际槽数、任务设备
缓冲区字节、全部 pinned 输入输出字节、逐轮耗时和检查结果。`task_device_bytes`
不包含共享结构与运行时内部资源。任何 GPU、检查或输出失败均非零退出，失败运行
不能作为有效性能结果。为每次运行指定独立输出路径，避免误读先前的成功文件。

输入文件摘要、源码提交及未提交补丁、构建参数、驱动、环境变量等实验身份由外层
实验记录保存，不在执行器中加入 Git、SSH 或设备管理工具依赖。

### 代码边界

`numeric.h` / `numeric.cu` 提供 `GpuMatrix`、`GpuTask` 和单任务兼容入口。
`GpuTask(matrix, batch_size)` 表示同模板的一批独立副本；`factors(copy)` 读取对应输出。
`prepare_graph()` 显式捕获并实例化图，可重复调用，且要求没有已排队输入或未完成的 enqueue。
`enqueue()` 自动恢复输入，也可先调用 `upload_input()` 提前排队。一次输入之后只调用一次 enqueue，
上传会使前一轮因子失效，需完成数值与输出后才能再次读取。
`GpuTask::enqueue()` 不分配、不全局等待、不计时；`wait()` 完成该任务并传播异步错误。
只有 wait 成功后才能读取因子，前一次提交未完成时禁止重用 task。析构会等待尚未
完成的任务，再释放资源，包括异常后的部分提交。`Symbolic_Matrix::solve` 的因子
视图重载支持检查任务结果而不覆盖模板初值。

`lu_bench.cpp` 负责参数、模板和任务组织、全局 LOOP、计时及检查。
`GpuMatrix` 的第四个参数选择 `UpdateMode::Scan` 或 `UpdateMode::Mapped`，默认扫描。
从新主机线程调用任务前，使用 `GpuMatrix::select_device()` 选择可见 GPU0。
不同任务可由不同线程提交，同一任务的方法调用需要串行。

### 显式任务清单

需要不均匀的模板频率或固定参考顺序时，使用 `--workload FILE` 替代 `-i` 和
`--tasks`。其他计时、检查及诊断参数保持不变：

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
  --warmup 2 --loops 10 --check full --diagnostics --output case.json
```

首条有效记录必须是版本头。随后定义全部 `matrix ID PATH`，再按参考顺序列出
`task ID`；每条 task 是一个独立任务。同一 ID 只准备一次模板，任务保留独立数值区、
输出，以及 scan 所需的 scratch。ID 必须唯一、先定义后引用，全部模板都必须被使用。
不会按文件路径或文件内容自动合并不同 ID。路径相对于清单目录解析，也可使用绝对路径；包含空格时
用双引号，双引号和反斜杠可用反斜杠转义。允许空行和以 `#` 开头的整行注释。
未知记录、额外字段、未定义 ID、重复 ID、未使用模板以及与 `-i` / `--tasks` 混用均报错。

单线程时，`--batch-size 1` 按清单顺序提交。更大的 batch 按模板定义顺序交错分批，尾批保留
实际副本数。每轮仍等待全部任务完成。该顺序是独立任务的基线参考
顺序，不表示任务依赖或到达时间。未来执行优化可以改变分组和提交顺序，但比较时
应保留相同逻辑任务集合，并另行记录执行策略。

JSON 的 `templates` 按定义顺序列出各模板及实际任务数；`task_templates` 按逻辑任务
顺序记录零起始的模板索引。原 `-i` / `--tasks` 入口也输出这个数组，仍按原循环分配
规则执行。诊断 launch 数、传输量及资源预检查按实际任务频率计算，检查覆盖全部任务。
样本选择、频率、随机顺序和输入摘要由外部负载准备工具负责，执行器不生成负载。

JSON 还输出 `batch_templates`、`batch_sizes` 与 `ordering`，记录批次
的模板、独立副本数和提交策略；`task_templates` 始终表示原逻辑任务顺序。

并行提交时，`batch_templates` 和 `batch_sizes` 描述批次的组织顺序，
不表示该轮观测到的开始提交时序。`ordering` 此时为 `parallel-batches`。
