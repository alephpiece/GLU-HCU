# GLU 文档

## 使用手册

| 文档 | 解决的问题 |
| --- | --- |
| [构建与验证](../BUILDING.md) | 配置 CUDA/HIP 和计算精度，构建程序并运行回归测试 |
| [命令行求解](solver.md) | 准备输入文件，求解单个线性系统，理解输出与主元扰动 |
| [性能测试](benchmark.md) | 定义任务、选择执行选项，解释计时、检查与 JSON 结果 |
| [C++ 数值任务接口](numeric-api.md) | 复用执行结构，上传新值，管理异步任务、结果与资源 |

首次使用从构建和命令行求解开始。测量重复分解或多矩阵吞吐时阅读性能测试手册，嵌入应用时阅读数值任务接口。Python 绑定见根目录 [README](../README.md#python-bindings-pyglu)。

## 算法资料与论文

以下资料介绍原版 GLU 及相关算法。CUDA/HIP 构建、批量执行和数值任务接口按上面的使用手册说明。

- [GLU 3.0 Manual](GLU_man.pdf)：原版使用手册。
- [GLU V3.0 算法介绍](GLU_v3.0_introduction.pdf)：hybrid right-looking 分解与 GPU 并行组织。
- [GLU3.0: Fast GPU-based Parallel Sparse LU Factorization for Circuit Simulation](Peng_DT_2020.pdf)：GLU 3.0 论文，另有 [2019 年预印本](Peng_GLU3.0_2019.pdf)。
- [GPU-Accelerated Parallel Sparse LU Factorization Method for Fast Circuit Analysis](He_TVLSI_15.pdf)：早期 GLU 方法，另有[论文草稿](tvlsi_gpu_lu14.pdf)。
- [Dynamic GPU Parallel Sparse LU Factorization for Fast Circuit Simulation](Lee_TVLSI_18.pdf)：相关工作，讨论动态 GPU 并行分解。
