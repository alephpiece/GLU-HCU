# 构建与运行

GLU 使用同一份源码构建 CUDA 或原生 HIP 版本，生成求解程序 `lu_cmd` 和性能测试程序 `glu_bench`。计算精度由构建选项决定，默认 FP64。

## 环境要求

- Linux x86_64，支持 C++17 的 C/C++ 编译工具链。
- CMake 3.21 或更高版本，以及 make 或 Ninja。
- CUDA 后端需要 CUDA Toolkit，`nvcc` 应在 `PATH` 中。
- HIP 后端需要原生 HIP 工具链。下文以安装在 `/opt/dtk` 的 DTK 26.04 为例。

NICSLU 和预处理代码随项目一起编译，无需单独安装。CMake 构建不包含 Python 扩展，Python 绑定的安装方法见 [README](README.md#python-bindings-pyglu)。

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
| `GLU_BUILD_TESTS` | `OFF` | 设为 `ON` 时构建回归测试 |

例如，在新的构建目录配置时加入 `-DGLU_USE_FP32=ON`，即可构建 FP32 版本。

## CUDA Makefile 入口

`src/Makefile` 也提供 CUDA 求解程序的构建入口，默认使用 FP64，目标架构为 `sm_60`。以下命令以 H100 为例：

```bash
cd /path/to/GLU_public/src
make clean
make -C nicslu/lib -j64
make -C nicslu/util -j64
make -j64 CUDA_ARCH=sm_90
```

生成的程序为 `src/lu_cmd`。修改目标架构或编译参数后，应先清理构建产物再重新编译。此入口在源码目录内生成对象文件，可在独立 worktree 中使用。

## 运行

`lu_cmd` 每次读取一个矩阵，求解右端向量全为 1 的线性系统，并将解写入当前目录的 `x.dat`。在独立输出目录运行，以免覆盖已有结果：

```bash
mkdir -p /path/to/output
cd /path/to/output
/path/to/build-hip/lu_cmd -i /path/to/matrix.mtx
```

CUDA 版本使用相应构建目录中的 `lu_cmd`。两个程序均选择第一个可见 GPU，可通过 `CUDA_VISIBLE_DEVICES` 或 `HIP_VISIBLE_DEVICES` 指定设备。

输入采用实数方阵的坐标格式：尺寸行包含行数、列数和非零元素数，随后每行包含从 1 开始的行索引、列索引和值。元素需先按列、再按行排序，并显式提供全部非零元素。读取器不自动展开对称存储，`.mtx` 扩展名本身也不保证满足这些要求。

加入 `-p` 可启用静态主元扰动：

```bash
/path/to/build-hip/lu_cmd -i /path/to/matrix.mtx -p
```

扰动会改变分解。程序先检查解是否有限，再写入 `x.dat`，保留所选精度往返转换所需的有效数字。GPU 操作、因子检查、解检查或文件读写失败时，程序以非零状态退出。退出码为零表示执行完成，数值正确性仍需通过残差或参考解检查。

## GPU 数值回归测试

在已配置的构建目录启用测试，然后重新构建并运行：

```bash
cmake -S /path/to/GLU_public -B /path/to/build-hip \
  -DGLU_BUILD_TESTS=ON
cmake --build /path/to/build-hip -j64
HIP_VISIBLE_DEVICES=0 ctest --test-dir /path/to/build-hip --output-on-failure
```

CUDA 后端使用对应构建目录和 `CUDA_VISIBLE_DEVICES`。测试需要可用 GPU，覆盖数值 kernel、批量任务与 Graph、任务生命周期，以及 CLI 输出和错误处理。具体回归用例维护在 [tests/](tests/)。

## 多矩阵性能测试

`glu_bench` 与 `lu_cmd` 共用 GPU 执行实现，使用相同输入格式，支持 CUDA/HIP 和所选构建精度。benchmark 不启用扰动，参数概要可用 `glu_bench --help` 查看。命令示例：

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

### 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `-i PATH` | 与 `--workload` 二选一，可重复 | 按给定顺序组成模板列表，任务循环引用它们 |
| `--workload FILE` | 与 `-i` 二选一 | 从显式清单读取模板与有序任务，不能同时指定 `--tasks` |
| `--tasks N` | `-i` 模板数 | 仅用于 `-i` 模式，独立任务总数必须至少覆盖全部模板 |
| `--warmup N` | 5 | 预热的完整 LOOP 数，允许 0 |
| `--loops N` | 30 | 计时的完整 LOOP 数，必须大于 0 |
| `--batch-size N` | 1 | 同模板独立副本的每批上限，每批一条 stream，尾批保留实际数量 |
| `--graph` | 关闭 | 准备阶段捕获每批数值与输出图，随后重放 |
| `--updates scan\|mapped` | `scan` | 扫描目标列，或预计算更新位置。mapped 增加模板共享的准备和显存成本 |
| `--submit-threads N` | 1 | 并行提交数值与输出的主机线程数上限，实际不超过批数。输入由主线程提前排队 |
| `--tmp-slots N` | 32 | scan 每个副本的 scratch 槽数上限，mapped 不使用 scratch |
| `--check full\|basic` | `full` | 完整因子残差检查，或轻量有限性检查 |
| `--diagnostics` | 关闭 | 输出提交与等待的墙钟、线程 CPU 时间、kernel 次数和传输量 |
| `--output PATH` | 无 | 成功后写 JSON，程序同时在标准输出打印日志和结果 |

任务数、批量上限、线程数和槽数均须为正整数。每个模板只预处理和准备一次，各任务拥有独立数值区和因子输出。同模板副本使用相同的不可变初值，每轮完整恢复，不模拟逐轮变化的输入。合批改变执行组织，不改变逻辑任务数。

### 显式任务清单

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

### 执行与资源

每批使用一条 nonblocking stream，维护 `H2D 初值 → 数值分解 → D2H 因子` 的依赖。每轮先由主线程排队全部输入，再提交数值与输出，上传后没有全局同步。全部任务及输出完成后才能开始下一轮。Graph 捕获数值与每批一次 D2H，H2D 在每轮的 Graph 外排队。不启用 Graph 时，直接提交各项操作并逐副本 D2H。

`--submit-threads` 大于 1 时，工作线程并行提交独立批次，只有一批时仍使用主线程。工作线程每轮创建和汇合，相关成本包含在 LOOP 中。主机线程数和逻辑 stream 数都不等于实际 GPU 并发度。

scan 使用每副本独立的 scratch，`--tmp-slots` 控制其槽数上限，实际槽数可能小于请求值。mapped 预计算并共享更新位置，不分配 scratch。mapped 多副本使用额外的交错 values buffer 和行补齐，计时内包含双向布局转换，单副本不做布局转换。

程序预检映射和任务缓冲能否放入当前空闲显存，失败时明确退出，不自动调整任务、批量或槽数。预检不包含运行时内部开销，也不保证后续设备或 pinned host 分配成功。

### 计时与结果

一次 LOOP 包含全部任务的 H2D、数值提交与执行、D2H、线程创建与汇合及轮末等待。预处理、资源和 Graph 准备、数值检查、solve、输出文件和释放资源不计入 LOOP。

| JSON 字段 | 含义 |
| --- | --- |
| `setup_ms` / `graph_setup_ms` | 准备总耗时及其中的 Graph 捕获与实例化耗时，准备时不执行分解 |
| `update_map_setup_ms` | mapped 路径的映射及辅助元数据准备耗时，包含在 `setup_ms` 中 |
| `warmup_ms` | 全部预热耗时，包含可能的首次重放开销 |
| `loop_ms` / `ms_per_loop` | 主机单调时钟测量的逐轮完整耗时及算术平均值 |
| `validation_ms` | 计时结束后检查全部任务的耗时 |
| `task_device_bytes` / `shared_device_bytes` | 任务缓冲及模板共享设备缓冲的载荷，分别统计，后者包含位置映射 |
| `update_map_bytes` | 共享设备缓冲中的位置映射载荷 |
| `pinned_bytes` | 模板共享初值与各独立副本因子输出的 pinned host 载荷 |
| `templates[].effective_tmp_slots` | 实际 scratch 槽数，mapped 为 0 |
| `task_templates` | 按逻辑任务顺序记录的零起始模板索引 |
| `batch_templates` / `batch_sizes` / `ordering` | 各批的模板、实际副本数与提交策略。并行时不表示实际提交时序 |

容量字段统计显式缓冲区载荷，不包含 allocator、stream、Graph 和运行时内部资源，也不代表进程峰值显存。报告还记录后端、精度、设备、检查模式和实际任务覆盖。

启用 `--diagnostics` 后另报：

| JSON 字段 | 含义 |
| --- | --- |
| `submit_ms` / `drain_ms` | 提交全部任务及随后等待的墙钟分段。submit 可含运行时等待，drain 不等于全部 GPU 时间 |
| `submit_thread_cpu_ms` / `drain_thread_cpu_ms` | 主线程在提交及等待阶段的 CPU 时间，分别统计，不含其他线程的 CPU 时间 |
| `submit_workers_cpu_ms` | 全部提交工作线程的 CPU 时间之和，可大于提交墙钟。单线程时为 0 |
| `submit_worker_max_wall_ms` | 最慢提交工作线程的墙钟耗时。单线程时为 0 |
| `launches_per_loop` | 每轮底层 GPU kernel 次数，Graph 重放时不等于主机 launch API 次数 |
| `wide_blocks_per_loop` | 每轮宽层数值 kernel 的线程块总数，包含适用路径的归一化与目标更新 |
| `template_work` | 每模板的层宽及结构目标数量，各批的实际启动次数另计 |
| `h2d_bytes_per_loop` / `d2h_bytes_per_loop` | 每轮按逻辑副本数统计的传输字节 |
| `h2d_copy_ops_per_loop` / `d2h_copy_ops_per_loop` | 每轮 H2D/D2H 操作次数，字节相同不代表操作次数相同 |

每轮 `submit_ms + drain_ms = loop_ms`。CPU 时间与墙钟分别解释，主线程 CPU 时间下降不代表全部线程工作减少。诊断统计不包括准备阶段，统计输出在计时区间外。

### 正确性与检查

默认 `--check full` 在计时后检查最后一轮的全部任务：有限因子、非零主元、全 1 RHS 的有限解，以及预处理矩阵的完整因子残差 `||B−LU||∞ / ||B||∞`。FP64 上限为 `1e-10`，FP32 为 `1e-3`。这检验分解质量，不保证病态原问题的前向精度。完整残差计算可能耗时较长，单列为 `validation_ms`。

`--check basic` 保留全部任务的有限因子、非零主元和有限解检查，跳过因子残差，残差及容差字段输出为 `null`。`status=ok` 仅表示所选检查通过，basic 通过不等于 full 验收。两种模式都只检查最后一轮，计时中检查提交与等待的 GPU 错误。

GPU、检查或输出失败均以非零状态退出。为每次运行指定独立输出路径，避免误读先前的成功文件。

C++ 执行接口的所有权、调用顺序和线程约束见 [numeric.h](include/numeric.h)，因子残差接口见 [symbolic.h](include/symbolic.h)。
