# 构建与验证

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

## GPU 数值回归测试

在已配置的构建目录启用测试，然后重新构建并运行：

```bash
cmake -S /path/to/GLU_public -B /path/to/build-hip \
  -DGLU_BUILD_TESTS=ON
cmake --build /path/to/build-hip -j64
HIP_VISIBLE_DEVICES=0 ctest --test-dir /path/to/build-hip --output-on-failure
```

CUDA 后端使用对应构建目录和 `CUDA_VISIBLE_DEVICES`。测试需要可用 GPU，覆盖数值 kernel、批量任务与 Graph、任务生命周期，以及 CLI 输出和错误处理。具体回归用例维护在 [tests/](tests/)。

## 运行检查

构建后可在独立输出目录运行求解器：

```bash
mkdir -p /path/to/output
cd /path/to/output
/path/to/build-hip/lu_cmd -i /path/to/matrix.mtx
```

程序求解右端向量全为 1 的线性系统，将解写入 `x.dat`。矩阵格式与运行用法见[命令行求解](docs/solver.md)。
