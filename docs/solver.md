# 命令行求解

构建方法见[构建与验证](../BUILDING.md)。`lu_cmd` 在 CPU 上完成预处理与符号分析，在 GPU 上进行数值分解，再由 CPU 三角求解。计算精度由构建配置决定，默认 FP64。

## 运行

`lu_cmd` 每次读取一个矩阵，求解右端向量全为 1 的线性系统，并将解写入当前目录的 `x.dat`。在独立输出目录运行，以免覆盖已有结果：

```bash
mkdir -p /path/to/output
cd /path/to/output
/path/to/build-hip/lu_cmd -i /path/to/matrix.mtx
```

CUDA 版本使用相应构建目录中的 `lu_cmd`。`lu_cmd` 和 `glu_bench` 均选择第一个可见 GPU，可通过 `CUDA_VISIBLE_DEVICES` 或 `HIP_VISIBLE_DEVICES` 指定设备。

## 矩阵输入

输入采用实数方阵的坐标格式：尺寸行包含行数、列数和非零元素数，随后每行包含从 1 开始的行索引、列索引和值。元素需先按列、再按行排序，并显式提供全部非零元素。对称存储需先展开，读取器只处理显式列出的项。

例如，以下内容可保存为 `matrix.mtx`，供两个程序使用：

```text
%%MatrixMarket matrix coordinate real general
3 3 7
1 1 4
2 1 2
1 2 1
2 2 5
3 2 3
2 3 1
3 3 6
```

## 主元扰动

加入 `-p` 可启用静态主元扰动：

```bash
/path/to/build-hip/lu_cmd -i /path/to/matrix.mtx -p
```

扰动会改变分解，应根据应用的数值要求启用。

## 输出与检查

程序先检查解是否有限，再写入 `x.dat`，每行一个解分量，保留所选精度往返转换所需的有效数字。执行或文件读写失败时，程序以非零状态退出。成功退出表示程序执行与有限性检查通过，解精度应结合原系统残差或可靠参考解评价。
