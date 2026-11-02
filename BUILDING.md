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

CUDA 后端使用对应构建目录和 `CUDA_VISIBLE_DEVICES`。测试需要可用 GPU，包含 10 个 kernel 用例和 4 个主机调度、重复任务及失败返回用例。检查覆盖稀疏尾部、主元广播、并发列更新、扰动、临时槽不足时的复用、多模板多任务四轮执行、模板寿命，以及零主元和非有限因子的拒绝。CTest 还检查 CLI 正常输出、文件打开及写入失败，以及无可见 GPU 时的错误返回。

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
| `-i PATH` | 必填，可重复 | 按给定顺序组成模板列表，任务循环引用它们 |
| `--tasks N` | 模板数 | 独立任务总数，必须至少覆盖全部模板；每任务一条 stream |
| `--warmup N` | 5 | 预热的全局 LOOP 数，允许 0 |
| `--loops N` | 30 | 计时的全局 LOOP 数，必须大于 0 |
| `--tmp-slots N` | 32 | 每任务 scratch 槽数上限，必须大于 0 |
| `--output PATH` | 无 | 成功后写 JSON；程序同时在标准输出打印日志和结果 |

`mixed-2000` 是上述第三条命令的实验名称，程序没有硬编码场景。三个模板的重复
按相同初值重复执行，不模拟逐轮变化的数值。不同输入可通过重复 `-i` 传入。

### 执行与计时

每个模板只做一次预处理、符号分析和只读索引上传。任务共享模板的只读 pinned
输入和设备结构，分别拥有数值区、scratch、pinned 输出和 nonblocking stream。
模板实现由任务保持寿命；输出不共享。

每轮由一个主机线程按任务编号提交完整的 `H2D 初值 → 数值分解 → D2H 因子` 链，
全部任务提交后逐流等待完成，然后才能开始下一轮。所有 kernel 显式使用任务流。
宽层沿用 batched RL，小层的 factor/update/clear 在任务流内按列顺序执行。
`LUonDevice` 是同一执行接口的单任务封装。性能计时使用 `glu_bench` 的 LOOP 指标；
求解 CLI 的准备与分解耗时不使用同一计时边界。

scratch 在准备阶段于任务流清零并等待完成；正常执行中 kernel 清理使用过的槽位，
跨层及跨轮复用。实际槽数是请求上限与模板 batched 层最大宽度的较小值；只有小层时
用 1 槽。32 是可重复的起步配置，不是已调优值。槽数影响分批与提交次数，跨平台比较
必须固定它。显存预检查只检验全部任务的数值区和 scratch 能否放入剩余容量，不能
保证驱动资源或 pinned host 分配一定成功；分配失败明确退出，不自动调整任务或槽数。

`setup_ms` 包含预处理、资源分配和初始化。`loop_ms` 使用主机单调时钟，包含 H2D、
计算提交、GPU 执行、D2H 和轮末等待；`ms_per_loop` 是这些完整轮次的算术平均。
输出文件、检查、释放不在该区间。逻辑 stream 数不等于实际同时执行的任务数。

同一配置运行三次独立进程，汇总 `ms_per_loop` 的中位数和最小—最大值；若轮次过短
或波动大，两平台同时增加预热或计时轮数。正式计时不挂 profiler。

### 正确性与结果

计时后检查**最后一轮的全部任务**：非有限因子、零主元、全 1 RHS 的解是否有限，
以及预处理矩阵 `B` 的完整 `||B−LU||∞ / ||B||∞`。残差在主机按稀疏列乘积用
`long double` 累积，包含乘积落在 B 稀疏模式之外的项，不以抽样或校验和代替。
FP64 验收上限为 `1e-10`，FP32 为 `1e-3`；这检验分解质量，不保证病态原问题的
前向精度，也不是区间算术认证。检查全部副本可能耗时较长，单列为 `validation_ms`。
计时中只检查提交与等待的 GPU 错误；最后一轮检查不等于逐轮数值检查，重复复位由
多轮已知因子回归另行覆盖。

JSON 记录后端、精度、设备、调度方式、模板与任务数、请求及实际槽数、任务设备
缓冲区字节、全部 pinned 输入输出字节、逐轮耗时和检查结果。`task_device_bytes`
不包含共享结构与运行时内部资源。任何 GPU、检查或输出失败均非零退出，失败运行
不能作为有效性能结果。为每次运行指定独立输出路径，避免误读先前的成功文件。

输入文件摘要、源码提交及未提交补丁、构建参数、驱动、环境变量等实验身份由外层
实验记录保存，不在执行器中加入 Git、SSH 或设备管理工具依赖。

### 代码边界

`numeric.h` / `numeric.cu` 提供 `GpuMatrix`、`GpuTask` 和单任务兼容入口。
`GpuTask::enqueue()` 不分配、不全局等待、不计时；`wait()` 完成该任务并传播异步错误。
只有 wait 成功后才能读取因子，前一次提交未完成时禁止重用 task。析构会等待尚未
完成的任务，再释放资源，包括异常后的部分提交。`Symbolic_Matrix::solve` 的因子
视图重载支持检查任务结果而不覆盖模板初值。

`lu_bench.cpp` 负责参数、模板和任务组织、全局 LOOP、计时及检查。
