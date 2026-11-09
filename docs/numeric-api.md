# C++ 数值任务接口

[numeric.h](../include/numeric.h) 提供单设备上的 GPU 数值分解接口，实现位于 [numeric.cu](../src/numeric.cu)。CMake 入口构建可执行程序，嵌入应用时需采用同一后端、计算精度和 NICSLU 依赖，配置见 [CMakeLists.txt](../CMakeLists.txt)。

## 矩阵模板与任务

`Symbolic_Matrix` 保存填充后的 CSC 结构、初始数值和依赖层。创建 `GpuMatrix` 前，先完成输入预处理及 `fill_in()`、`csr()`、`predictLU()`、`leveling()`。完整准备流程见 [lu_cmd.cpp](../src/lu_cmd.cpp)，数据结构见 [symbolic.h](../include/symbolic.h)。

`GpuMatrix` 是准备后的矩阵模板，保存共享执行结构、不可变默认初值及所选策略的索引或计划。`GpuTask` 执行同一结构的一批独立数值副本，每批拥有自己的 stream、数值缓冲、Graph 和主机因子输出。任务通过共享所有权保留模板，销毁 `GpuMatrix` 对象后，已有任务仍可继续使用。

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
| `Mapped` | 预计算更新位置，直接读取 L 值，不使用 scratch |
| `Panel` | 按依赖层窗口聚合同目标贡献，准备目标分组与 L/U 位置计划 |
| `PanelTiled` | 从窗口更新中提取局部矩形，在 gfx936 使用原生矩阵指令，其它目标采用标量回退 |
| `Fused` | 一个线程块完成一个矩阵的全部依赖层，复用 Panel 分组计划，采用 AoS 数值布局 |

`Fused` 由独立矩阵提供块间并行，适合大量小矩阵；较大矩阵仍需实测块内串行推进的成本。

映射和计划由同模板任务共享。调用者按结构、准备时间、设备容量和实际复用次数选择策略。`PanelTiled` 保持所选输入精度，各策略仍可能改变浮点累加次序，需要相应的数值检查。任务内部完成 AoS/SoA 布局准备，调用者始终按下面的输入顺序提供数值。

## 默认初值恢复

Snapshot 是模板的不可变默认初值。`SnapshotPolicy` 只选择默认初值的恢复方式，与完整或紧凑新值输入分开。

| `SnapshotPolicy` | 恢复方式与容量 |
| --- | --- |
| `Auto`，默认 | 创建任务时解析一次，Panel 多副本采用 Shared，其余采用 PerCopy |
| `PerCopy` | 为每个副本分别上传主机初值 |
| `Shared` | 每批只上传一份初值，再向各数值副本广播 |
| `Device` | 准备阶段额外保留每模板一份只读设备初值，每轮从设备恢复，直到模板释放 |

`GpuMatrix::snapshot_policy()` 返回请求策略，`GpuTask::snapshot_policy()` 返回该批已确定的实际策略。Device 仅通过显式指定启用。

Device 适用于跨轮初值不变的场景，需比较每轮耗时与额外容量。跨轮复用次数决定准备成本的摊销，批内副本数决定并行规模，两者分别评价。修改默认初值或执行结构时重建模板。

## 数值输入

以下三种调用均可复用已准备的 Graph。上传新值绕过快照恢复策略，并保留模板默认初值。

| 调用 | 输入与元素数 |
| --- | --- |
| `upload_input()` | 恢复模板默认初值，省略时由 `enqueue()` 自动恢复 |
| `upload_input(values, elements)` | 连续 `[copy][factor slot]`，元素数为 `factor_nnz × copies` |
| `upload_compact_input(values, elements)` | 连续 `[copy][input position]`，元素数为 `compact_input_elements() × copies`，须先准备位置表 |

`factor_nnz` 对应 `symbolic.nnz`，包含填充后的全部 L/U 存储位置。完整输入按 `symbolic.val` 顺序提供初值，包括应为零的填充槽。每个副本可以有不同数值。

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

每个任务一次只能排队一份输入、保留一个未完成提交。显式上传后必须调用 `enqueue()`，再由 `wait()` 等待输入、分解和输出全部完成。源数组须保持可读且不变，直到对应 `wait()` 返回，pinned host 内存支持异步上传。

`factors(copy)` 返回任务持有的主机视图，每副本含 `symbolic.nnz` 个因子值，沿用符号矩阵的填充 CSC 顺序。结果在成功 `wait()` 后读取，在下一次上传、提交或任务销毁前消费。GPU 提交或执行失败后需重建任务，提交失败会尝试排空已排队工作并保留原异常，析构也会同步后释放资源。

同一任务的调用由调用者串行化，不同任务可由不同主机线程提交。每个提交线程先调用 `GpuMatrix::select_device()`，选择第一个可见 GPU。

长期重复分解可保留模板、任务与 Graph，每轮只恢复默认初值或上传新值，再提交、等待并消费结果。调用者可直接生产到最终输入缓冲。独立批次还可使用多个缓冲和共享模板的任务，让生产与执行形成流水线。缓冲按对应 `wait()` 和结果消费完成的时机复用，流水线由调用者组织。

## 容量统计

`GpuMatrix::shared_device_bytes()` 统计模板结构、映射或计划、紧凑位置表及可选设备初值。按实际副本数调用 `task_device_bytes(copies)`，得到包含布局补齐与紧凑输入缓冲的任务设备缓冲字节数。

`task_host_bytes()` 是一份初值或因子向量的字节数。每模板保留一份固定页初值，每个副本保留一份固定页因子输出，调用者的新值输入缓冲另计。上述容量统计只包括显式数组，allocator、stream 和 Graph 等运行时资源另计。

## 数值检查

`GpuTask::check()` 检查因子有限性与非零主元。完整因子残差由 `Symbolic_Matrix::factor_residual()` 计算，以该对象的 `val` 作为输入矩阵。异值副本的检查需使用保存本轮该副本初值的符号矩阵对象，原系统的解误差还需单独评价。

完整方法与错误约定见 [numeric.h](../include/numeric.h)，因子残差和 CPU 三角求解接口见 [symbolic.h](../include/symbolic.h)。对应回归见 [gpu-kernels.cu](../tests/gpu-kernels.cu)。
