# C++ 数值任务接口

[numeric.h](../include/numeric.h) 提供单设备上的 GPU 数值分解接口，实现位于 [numeric.cu](../src/numeric.cu)。CMake 入口构建可执行程序，嵌入应用时需采用同一后端、计算精度和 NICSLU 依赖，配置见 [CMakeLists.txt](../CMakeLists.txt)。

## 矩阵模板与任务

`Symbolic_Matrix` 保存填充后的 CSC 结构、初始数值和依赖层。创建 `GpuMatrix` 前，先完成输入预处理及 `fill_in()`、`csr()`、`predictLU()`、`leveling()`。完整准备流程见 [lu_cmd.cpp](../src/lu_cmd.cpp)，数据结构见 [symbolic.h](../include/symbolic.h)。

`GpuMatrix` 是准备后的矩阵模板，保存共享执行结构、不可变默认初值及所选策略的索引或计划。执行单位持有可变数值和输出，并通过共享所有权保留模板；销毁 `GpuMatrix` 包装对象后，已有执行单位仍可继续使用。

| 执行单位 | 任务组成 | 提交与资源 |
| --- | --- | --- |
| `GpuTask` | 同一模板的一批独立数值副本，支持所有更新策略 | 每批一个 stream、独立数值与输出缓冲，可准备 Graph |
| `GpuTaskSet` | 有序的独立数值任务，各任务可引用同一模板或不同模板，所有模板统一使用 `Mapped` 或 `Fused` | 整个集合共用一个 stream，按逻辑任务顺序拼接数值和输出缓冲 |

逐任务 Mapped 为每个逻辑任务配置一个单副本 `GpuTask`，各自在自己的 stream 上提交。同一 Mapped 模板有多个数值副本时，也可用多副本 `GpuTask` 合批，利用交错的 SoA 布局并行处理不同副本。集合 Mapped 用一个 `GpuTaskSet` 按各任务的消元依赖层合并更新，集合 Fused 则在同一网格中为每个任务分配一个线程块。逐任务 Mapped 和集合 Mapped 都可直接执行，也可调用 `prepare_graph()`，分别为每个任务或整个集合捕获一份 Graph。

下面的函数对已准备的结构重复分解，每轮恢复相同默认初值：

```cpp
#include "numeric.h"

void factor_repeated(const Symbolic_Matrix& symbolic, unsigned copies, unsigned rounds)
{
    GpuMatrix::select_device();
    GpuMatrix prepared(symbolic, 32, false, UpdateMode::Mapped);
    GpuTask task(prepared, copies);
    task.prepare_graph(); // Optional; prepare once before submitting input.

    for (unsigned round = 0; round < rounds; ++round) {
        task.enqueue(); // Restore the default input, factorize and copy back.
        task.wait();
        task.check();
        const REAL* factors = task.factors(0);
        // Consume this copy's factors before reusing the task.
    }
}
```

`prepare_graph()` 在没有待处理输入或未完成提交时调用，重复调用会复用已有图。所有输入方式都先在图外恢复数值执行布局，图只包含分解与输出。不调用 `prepare_graph()` 时，任务直接提交这些操作。

## 数值更新策略

| `UpdateMode` | 数值更新与准备成本 |
| --- | --- |
| `Scan`，默认 | 数值阶段扫描目标列，每副本使用独立 scratch，槽数上限由构造参数指定 |
| `Mapped` | 预计算稀疏更新位置。单副本 `GpuTask` 可在稠密子矩阵中做分块消元 |
| `Panel` | 按依赖层窗口聚合同目标贡献，准备目标分组与 L/U 位置计划 |
| `PanelTiled` | 从窗口更新中提取局部矩形，在 gfx936 使用原生矩阵指令，其它目标采用标量回退 |
| `Fused` | 一个线程块完成一个矩阵的全部依赖层，复用 Panel 分组计划，采用 AoS 数值布局 |

消元第 `k` 步将源列的下三角条目除以对角主元 `U(k,k)` 得到 `L(i,k)`，再执行 `A(i,j) -= L(i,k) × U(k,j)`，其中 `i > k`、`j > k`。目标列 `j` 的 CSC 条目按行排序，上三角条目 `U(k,j)` 的槽位在更新目标之前。当所有因子列长度至多为 65,536 时，`Mapped` 用 16 位无符号数保存两个槽位之差 `slot(i,j) - slot(k,j)`。只要有更长的因子列，整个模板就使用 32 位绝对槽位。同一模板在 `GpuTask` 和 `GpuTaskSet` 中共用这份映射。

**稠密子矩阵的分块消元。** 单副本 Mapped 任务从符号结构中识别可以连续消去的一串主元。第一个主元的 L 列和 U 行确定子矩阵的行号和列号，两组索引各自无重复。消去一个主元后，如果余下的行、列恰好与下一个主元的 L 列、U 行吻合，就可以在同一子矩阵内继续消元。这样的主元链至少有 32 个主元，首个子矩阵至多有 2²⁰ 个条目。任务将子矩阵读入稠密工作区，按链中顺序分块消元，再将结果写回原位置。各条链复用同一工作区，所需容量计入 `GpuMatrix::task_device_bytes(1)`。

各后端都在连续的单源层中识别主元链。HIP 和 CUDA FP32 还可以从每层含 2 至 16 个源列的连续层中选出主元链，在起始层就处理整条链。为保证消元次序，链中各主元依赖的链外更新必须在起始层之前完成。为分摊工作区读写和内核启动的成本，HIP 要求每条链至少覆盖 500 万项更新乘积，CUDA FP32 要求至少 100 万项。其余源列仍在各自的依赖层按稀疏映射更新。

**提前归一化 L。** 单副本 Mapped 任务按目标列分配线程块时，多个线程块会用到同一源列的 L 值。CUDA FP64 将每个源列的严格下三角条目数乘以右侧目标列数，求和得到整层的潜在更新乘积数。达到 10 万项时，先将各源列的下三角条目除以各自主元并写回，随后各目标线程块直接读取算好的 L 值。低于门槛的目标并行层以及采用分块消元的子矩阵，在分解结束时统一归一化下三角条目。

CUDA FP64 的多副本 Mapped 任务使用 SoA 布局，逐层归一化 L。`GpuTask::normalizes_each_level()` 在这种情况下返回 `true`，对仅在部分层提前归一化的单副本任务返回 `false`。

`Fused` 使用 Panel 的标量依赖窗口，每个更新阶段为目标分配唯一写者，保留各目标贡献的累加顺序。归一化按下三角元素平铺，即时更新区分单项与多项贡献，延迟更新把相同贡献长度的目标排成 logical32 条带，交错保存操作数索引。它不提取密集 tile。`GpuTask` 和 `GpuTaskSet` 共用块内分解函数，各矩阵保持独立 AoS 数值。

Fused 线程块大小根据模板的目标并行度选择 256 或 1024。集合 Fused 使用参与任务所需的最大线程块，并按因子条目数降序分配任务描述符。输入、输出和 API 下标保持逻辑任务顺序。矩阵之间提供块间并行，单个矩阵的并行度限于一个线程块。同结构高复用批次可与分层 SoA 执行比较。

映射和计划由同模板任务共享。调用者按结构、准备时间、设备容量和实际复用次数选择策略。`PanelTiled` 保持所选输入精度，各策略仍可能改变浮点累加次序，需要相应的数值检查。任务内部完成 AoS/SoA 布局准备，调用者始终按下面的输入顺序提供数值。

## 默认初值恢复

Snapshot 是模板的不可变默认初值。`SnapshotPolicy` 选择 `GpuTask` 的默认初值恢复方式，与完整或紧凑新值输入分开。

| `SnapshotPolicy` | 恢复方式与容量 |
| --- | --- |
| `Auto`，默认 | 创建任务时解析一次，Panel 多副本采用 Shared，其余采用 PerCopy |
| `PerCopy` | 为每个副本分别上传主机初值 |
| `Shared` | 每批只上传一份初值，再向各数值副本广播 |
| `Device` | 准备阶段额外保留每模板一份只读设备初值，每轮从设备恢复，直到模板释放 |

`GpuMatrix::snapshot_policy()` 返回请求策略，`GpuTask::snapshot_policy()` 返回该批已确定的实际策略。Device 仅通过显式指定启用。

Device 适用于跨轮初值不变的场景，需比较每轮耗时与额外容量。跨轮复用次数决定准备成本的摊销，批内副本数决定并行规模，两者分别评价。修改默认初值或执行结构时重建模板。

`GpuTaskSet` 使用自己的连续主机初值缓存，首次默认恢复时打包，此后复用。模板的 `SnapshotPolicy` 不改变集合的恢复方式；显式以 Device 策略准备的模板仍占有额外设备初值，该缓冲不参与集合恢复。

## 数值输入

两种执行单位均提供以下三种输入调用。`GpuTask` 的各输入路径可复用已准备的 Graph，`Mapped` 集合也可复用 Graph。上传新值保留模板默认初值。

| 调用 | 输入与元素数 |
| --- | --- |
| `upload_input()` | 恢复模板默认初值，省略时由 `enqueue()` 自动恢复 |
| `upload_input(values, elements)` | `GpuTask` 为连续 `[copy][factor slot]`，共 `factor_nnz × copies` 个元素；集合按逻辑任务拼接完整因子区间 |
| `upload_compact_input(values, elements)` | `GpuTask` 为连续 `[copy][input position]`，共 `compact_input_elements() × copies` 个元素；集合按逻辑任务拼接紧凑区间，所用模板均须准备位置表 |

因子条目数 `factor_nnz` 对应 `symbolic.nnz`，是预处理后 L/U 数值数组的元素数，包括填充项和已存储的零值项。完整输入按 `symbolic.val` 顺序提供初值，包括应为零的填充位置。每个副本可以有不同数值。

紧凑位置表是非空索引序列，各项互异，范围为 `[0, symbolic.nnz)`，按给定顺序对应紧凑数组中的值。它须覆盖所有可能非零的位置，包括固定非零项和当前为零、以后可能非零的项。每次紧凑上传时，省略位置置零。

接口接收预处理后的数值。调用者负责排列、缩放、重复项处理，以及输入数值与固定执行结构的一致性。原始稀疏模式相同，预处理仍可能改变执行结构。执行结构或紧凑位置集合、顺序改变时需重新准备模板与任务。

以下示例假定 `symbolic` 已准备，`input_positions` 和主机数组 `host_values` 满足上述顺序与大小要求：

```cpp
GpuMatrix prepared(symbolic, input_positions, 32, false, UpdateMode::Mapped);
GpuTask task(prepared, copies);
task.prepare_graph(); // Optional; call before queuing input.
task.upload_compact_input(host_values, prepared.compact_input_elements() * copies);
task.enqueue();
task.wait();
const REAL* factors = task.factors(0);
```

显式完整新值和紧凑新值均不能与静态主元扰动同用，扰动阈值属于模板默认初值。

## 异步调用与资源生存期

每个 `GpuTask` 或 `GpuTaskSet` 一次只能排队一份输入、保留一个未完成提交。显式上传后必须调用 `enqueue()`，再由 `wait()` 等待输入、分解和输出全部完成。源数组须保持可读且不变，直到对应 `wait()` 返回，pinned host 内存支持异步上传。

`GpuTask::factors(copy)` 或 `GpuTaskSet::factors(task)` 返回执行单位持有的主机视图，含对应模板的 `symbolic.nnz` 个因子值，沿用填充 CSC 顺序。结果在成功 `wait()` 后读取，在下一次上传、提交或执行单位销毁前消费。GPU 提交或执行失败后需重建该执行单位，提交失败会尝试排空已排队工作并保留原异常，析构也会同步后释放资源。

同一执行单位的调用由调用者串行化，不同执行单位可由不同主机线程提交。每个提交线程先调用 `GpuMatrix::select_device()`，选择第一个可见 GPU。

长期重复分解可保留模板、执行单位及可用的 Graph，每轮只恢复默认初值或上传新值，再提交、等待并消费结果。调用者可直接生产到最终输入缓冲。多个独立执行单位可共享模板，让生产与执行形成流水线。缓冲按对应 `wait()` 和结果消费完成的时机复用，流水线由调用者组织。

## 异构任务集合

创建 `GpuTaskSet` 时，调用者给出同为 `Fused` 或同为 `Mapped` 的模板列表，以及按逻辑任务顺序排列的模板下标。即使两个任务引用同一模板，它们也各有自己的数值和因子输出。只有一个任务的集合仍按所选的集合执行方式提交。

集合 Mapped 把各任务在同一依赖层的更新合并到同一个网格。每组 32 个线程处理一个源列 `k` 和目标列 `j` 的组合，沿 `L(:,k)` 的严格下三角部分分担各行的更新。同一任务中，不同源列对同一因子槽位的贡献使用原子加。CUDA FP64 在每层更新前写回该层归一化的 L 值。HIP 和 CUDA FP32 在更新时用 `U(k,k)` 求出所需的 `L(i,k)`，待各层更新完成后统一写回 L。空闲时可调用 `prepare_graph()` 捕获数值阶段和因子回传。三种输入都在图外恢复数值布局，可以复用同一份 Graph。

```cpp
GpuMatrix a(symbolic_a, positions_a, 32, false, UpdateMode::Fused);
GpuMatrix b(symbolic_b, positions_b, 32, false, UpdateMode::Fused);
GpuTaskSet tasks({&a, &b}, {1, 0, 1});
// Produce values in logical task order into a caller-owned pinned array.
const size_t elements = tasks.compact_input_offset(tasks.task_count());
tasks.upload_compact_input(host_values, elements);
tasks.enqueue();
tasks.wait();
const REAL* first = tasks.factors(0); // Task 0's factors, in template b's slot order.
```

集合分别按完整因子数和紧凑位置数计算前缀偏移，`factor_offset(t)` 与 `compact_input_offset(t)` 给出第 `t` 个任务的起点，终点下标 `task_count()` 返回总元素数。完整输入按 `[task][factor slot]` 拼接，紧凑输入按 `[task][input position]` 拼接。紧凑上传要求全部任务的模板都已准备位置表，完整输入和默认恢复不要求位置表。

两种集合每轮各做一次连续 H2D 上传和 D2H 输出，完整因子使用普通 AoS 布局。Fused 把清零、紧凑输入 scatter 和分解合入一次 grid 启动。Mapped 先清零并按固定位置 scatter 紧凑输入，再提交分层数值工作。

`upload_input()` 或没有显式输入的 `enqueue()` 恢复各模板的默认初值。新值上传不分配或修改默认缓存，含静态主元扰动的集合只接受默认初值。

`wait()` 一次完成整个集合。`default_input_setup_ms()` 单列首次默认缓存的分配与打包时间，调用者可在计时前显式恢复并执行一轮，区分这项准备成本与后续重复执行。

## 容量统计

`GpuMatrix::shared_device_bytes()` 统计模板结构、映射或计划、紧凑位置表及可选设备初值。同一模板被多个执行单位引用时只计一次。按实际副本数调用 `task_device_bytes(copies)`，得到一个 `GpuTask` 的设备缓冲字节数，包含布局补齐与紧凑输入缓冲；集合使用 `GpuTaskSet::device_bytes()`，统计集合数值、紧凑输入和任务描述，模板资源另计。

`task_host_bytes()` 是模板的一份初值或因子向量的字节数。每模板保留一份固定页初值，每个逻辑任务保留一份固定页因子输出。`GpuTaskSet::pinned_bytes()` 统计集合的全部输出及已分配的默认缓存，不含模板初值。调用者的新值输入缓冲另计。上述容量统计只包括显式数组，allocator、stream 和 Graph 等运行时资源另计。

## 数值检查

`GpuTask::check()` 与 `GpuTaskSet::check()` 检查各自全部因子的有限性与非零主元。完整因子残差由 `Symbolic_Matrix::factor_residual()` 计算，以该对象的 `val` 作为输入矩阵。每个逻辑任务的检查需使用保存其本轮初值的符号矩阵对象，原系统的解误差还需单独评价。

完整方法与错误约定见 [numeric.h](../include/numeric.h)，因子残差和 CPU 三角求解接口见 [symbolic.h](../include/symbolic.h)。通用数值与失败路径回归见 [gpu-kernels.cu](../tests/gpu-kernels.cu)，块内执行和集合契约分别见 [fused-execution.cpp](../tests/fused-execution.cpp) 与 [task-set.cpp](../tests/task-set.cpp)。
