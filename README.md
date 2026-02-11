# RMSD分析插件

## 插件功能说明

本插件用于计算分子动力学轨迹的均方根偏差（RMSD），支持对齐和非对齐两种计算方式。主要功能包括：

1. 读取多种格式的轨迹文件（PDB、DCD、XTC、TRR、NC、NPY、NPZ等）
2. 计算相对于参考帧的RMSD值
3. 支持Kabsch对齐算法进行结构叠合
4. 输出RMSD数据文件（.dat格式）
5. 生成RMSD可视化图片（.png格式）

## 输入数据说明

### 必需输入

- **参数配置文件（params.json）**：包含所有运行参数的JSON文件

### 可选输入

- **轨迹文件（input_file）**：分子动力学轨迹文件，支持以下格式：
  - `.pdb`：蛋白质数据库格式
  - `.dcd`：二进制轨迹格式（需要拓扑文件）
  - `.xtc`：GROMACS压缩轨迹格式（需要拓扑文件）
  - `.trr`：GROMACS全精度轨迹格式（需要拓扑文件）
  - `.nc` / `.netcdf`：NetCDF格式（需要拓扑文件）
  - `.npy`：NumPy数组格式
  - `.npz`：NumPy压缩数组格式

- **拓扑文件（topology_file）**：当输入为dcd/xtc等轨迹格式时需要提供拓扑文件（如.pdb文件）

**注意**：如果不提供输入文件，程序将自动生成示例数据进行演示。

## 输出数据说明

所有输出文件统一保存在指定的输出目录（默认为 `./output`）中：

### 数据文件

- **rmsd.dat**：RMSD数据文件，包含每帧的RMSD值（单位：nm）

### 可视化文件

- **rmsd.png**：RMSD随帧数变化的折线图

### 日志文件

- **run.log**：运行日志文件，记录分析过程的详细信息

## 参数说明

### 必需参数

- **output_dir**（字符串）：结果输出目录路径，默认为 `"./output"`

### 可选参数

- **input_file**（字符串）：输入轨迹文件路径，为空时使用示例数据
- **topology_file**（字符串）：拓扑文件路径，某些轨迹格式需要
- **reference_frame**（整数）：参考帧索引，0表示第一帧，-1表示最后一帧，默认为 `0`
- **align**（布尔值）：是否进行Kabsch对齐，默认为 `"True"`
- **data_filename**（字符串）：RMSD数据文件名（不含扩展名），默认为 `"rmsd"`
- **plot_filename**（字符串）：RMSD图片文件名，默认为 `"rmsd.png"`
- **sample_frames**（整数）：示例数据的帧数，默认为 `100`
- **sample_atoms**（整数）：示例数据的原子数，默认为 `50`

## 使用示例

### 命令行运行

```bash
python plot.py params.json
```

### params.json 示例

```json
{
  "input_file": "",
  "topology_file": "",
  "output_dir": "./output",
  "reference_frame": 0,
  "align": "True",
  "data_filename": "rmsd",
  "plot_filename": "rmsd.png"
}
```

## 依赖环境

- Python 3.8+
- numpy
- matplotlib
- mdtraj（可选，用于读取真实轨迹文件）

## 版本信息

- 版本：v1.0.0
- 更新日期：2026-02-11