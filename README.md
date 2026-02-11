---
AIGC:
    ContentProducer: Minimax Agent AI
    ContentPropagator: Minimax Agent AI
    Label: AIGC
    ProduceID: "00000000000000000000000000000000"
    PropagateID: "00000000000000000000000000000000"
    ReservedCode1: 3045022100b4ad04f130e68da6f9187b7b4d62321658b2bf55f9b5be497db6c680e8a2d88702200bf1f136d5f13466ed4d33f496a8787e6a97ab74114381e7b5c9323bdcfd644f
    ReservedCode2: 3046022100c882d05f993e2463cc5b42f71eeecc2163e37dc58865ca94d8959f8736204a64022100b4cd48bba8f60c50a83e6aa44679316ecc43ee13b4f57633f46968544096f075
---

# RMSD分析插件

基于FastMDAnalysis架构的分子动力学轨迹均方根偏差分析插件

## 概述

这是一个完整的RMSD（Root-Mean-Square Deviation）分析插件，专为分子动力学轨迹分析而设计。插件遵循FastMDAnalysis的架构规范，提供从数据加载到结果可视化的完整分析流程。

## 特性

### 核心功能
- ✅ **RMSD计算**: 支持多种对齐方式的RMSD计算
- ✅ **数据验证**: 自动验证轨迹文件和拓扑文件格式
- ✅ **统计分析**: 提供全面的统计摘要和分布分析
- ✅ **异常值检测**: 支持IQR、Z-score等多种异常值检测方法
- ✅ **收敛分析**: 自动检测RMSD收敛点
- ✅ **批量处理**: 支持批量轨迹分析

### 可视化功能
- 📊 **时间序列图**: RMSD随时间变化的动态图
- 📈 **分布分析**: 直方图、密度图、箱线图
- 🔍 **趋势分析**: 移动平均、变化率、自相关分析
- 🌡️ **热力图**: 多轨迹RMSD矩阵可视化
- 📋 **综合仪表板**: 发表质量的综合分析图表

### 数据格式支持
- **轨迹文件**: .dcd, .xtc, .trr, .netcdf, .nc
- **拓扑文件**: .pdb, .gro, .crd, .mol2
- **输出格式**: CSV, JSON, PNG, PDF, TXT

## 安装

### 系统要求
- Python 3.8+
- 依赖包详见 `requirements.txt`

### 安装步骤

1. **克隆或下载插件**
```bash
git clone <repository-url>
cd rmsd_plugin
```

2. **安装依赖包**
```bash
pip install -r requirements.txt
```

3. **验证安装**
```bash
python main.py --demo
```

## 快速开始

### 1. 基本使用

```python
from rmsd_plugin import RMSDAnalysis, quick_analysis

# 方法1: 快速分析
results = quick_analysis("trajectory.dcd", "topology.pdb")

# 方法2: 详细分析
analyzer = RMSDAnalysis("config/config.json")
analyzer.load_trajectory("trajectory.dcd", "topology.pdb")
results = analyzer.analyze(save_plots=True)
```

### 2. 自定义分析

```python
# 创建分析器
analyzer = RMSDAnalysis()

# 加载数据
analyzer.load_trajectory("traj.dcd", "top.pdb")

# 自定义RMSD计算
rmsd_values = analyzer.compute_rmsd(
    ref_frame=10,
    align=True,
    atom_selection="protein and name CA"
)

# 执行完整分析
results = analyzer.analyze(
    save_plots=True,
    save_intermediate=True,
    output_dir="my_analysis"
)
```

### 3. 批量分析

```python
from rmsd_plugin import batch_analysis

# 批量处理多个轨迹
traj_files = ["traj1.dcd", "traj2.dcd", "traj3.dcd"]
top_files = ["top1.pdb", "top2.pdb", "top3.pdb"]

results = batch_analysis(
    traj_files, top_files, 
    output_dir="batch_results",
    config={"verbose": True}
)
```

### 4. 可视化

```python
from rmsd_plugin import RMSDVisualizer, create_rmsd_dashboard

# 创建可视化器
visualizer = RMSDVisualizer({
    'figure_size': [15, 10],
    'dpi': 300,
    'style': 'seaborn-v0_8'
})

# 绘制时间序列图
fig = visualizer.plot_rmsd_timeseries(
    rmsd_values, 
    time_values,
    save_path="rmsd_timeseries.png"
)

# 创建综合仪表板
dashboard_files = create_rmsd_dashboard(
    rmsd_data, 
    output_dir="visualization"
)
```

## 配置说明

### 主配置文件 (config/config.json)

```json
{
  "rmsd_parameters": {
    "reference_frame": 0,
    "align_trajectory": true,
    "atom_selection": "protein and name CA"
  },
  "visualization": {
    "figure_size": [12, 8],
    "dpi": 300,
    "save_plots": true
  },
  "output_settings": {
    "save_raw_data": true,
    "save_statistics": true,
    "save_summary": true
  }
}
```

### 参数配置文件 (config/params.json)

包含详细的参数验证规则和约束条件。

## 原子选择语法

插件支持MDTraj的原子选择语法：

| 选择类型 | 表达式 | 描述 |
|---------|--------|------|
| 蛋白质Cα | `protein and name CA` | 蛋白质主链Cα原子 |
| 主链原子 | `protein and name C N O CA` | 蛋白质主链原子 |
| 全部蛋白质 | `protein` | 所有蛋白质原子 |
| 疏水核心 | `protein and (name CB or name CG)` | 疏水残基 |
| 表面残基 | `protein and (name OD1 or name OD2)` | 极性残基 |
| 配体 | `not protein and not water` | 非蛋白质、非水分子 |

## 输出文件说明

### 数据文件
- `rmsd_results.csv`: RMSD时间序列数据
- `rmsd_statistics.json`: 统计分析结果
- `rmsd_analysis_summary.json`: 完整分析摘要

### 可视化文件
- `rmsd_timeseries.png`: RMSD时间序列图
- `rmsd_distribution.png`: 分布分析图
- `rmsd_trend_analysis.png`: 趋势分析图
- `rmsd_publication_figure.png`: 发表质量综合图

### 日志文件
- `analysis_log.log`: 详细分析日志

## 示例演示

### 运行示例数据演示

```bash
python main.py --demo
```

这将：
1. 生成1000帧的示例RMSD数据
2. 执行完整分析流程
3. 生成所有可视化图表
4. 创建分析报告

### 使用真实数据演示

```bash
python main.py --real trajectory.dcd topology.pdb -o my_analysis
```

## API参考

### RMSDAnalysis类

```python
class RMSDAnalysis(BaseAnalysis):
    def __init__(self, config_path=None)
    def load_trajectory(self, traj_path, top_path)
    def compute_rmsd(self, ref_frame=0, align=True, atom_selection=None)
    def analyze(self, **kwargs)
```

### RMSDVisualizer类

```python
class RMSDVisualizer:
    def __init__(self, config=None)
    def plot_rmsd_timeseries(self, rmsd_values, time_values=None, ...)
    def plot_rmsd_distribution(self, rmsd_values, ...)
    def plot_rmsd_heatmap(self, rmsd_matrix, ...)
    def create_publication_figure(self, rmsd_data, ...)
```

## 高级功能

### 1. 自定义平滑

```python
from rmsd_plugin.utils import smooth_rmsd

# Savitzky-Golay平滑
smoothed = smooth_rmsd(rmsd_values, method='savgol', window_length=51)

# 高斯平滑
smoothed = smooth_rmsd(rmsd_values, method='gaussian', sigma=2.0)
```

### 2. 异常值检测

```python
from rmsd_plugin.utils import detect_outliers

# IQR方法
outliers_iqr = detect_outliers(rmsd_values, method='iqr')

# Z-score方法
outliers_zscore = detect_outliers(rmsd_values, method='zscore', factor=2.5)
```

### 3. 收敛分析

```python
from rmsd_plugin.utils import find_convergence_point

convergence_point = find_convergence_point(
    rmsd_values, 
    window_size=50, 
    threshold=0.1
)
```

## 故障排除

### 常见问题

1. **MDTraj未安装**
   - 解决方案: `pip install mdtraj`

2. **文件格式不支持**
   - 检查轨迹文件格式是否在支持列表中
   - 确保拓扑文件与轨迹文件兼容

3. **内存不足**
   - 减少轨迹帧数
   - 使用原子选择减少计算量

4. **原子选择无效**
   - 使用预设的选择模板
   - 检查语法是否正确

### 调试模式

```python
# 启用详细日志
analyzer = RMSDAnalysis()
analyzer.logger.setLevel(logging.DEBUG)
```

## 性能优化

### 大轨迹文件处理
- 使用原子选择减少计算量
- 分块处理超长轨迹
- 启用内存优化模式

### 批量处理
- 使用多进程处理
- 预先验证所有输入文件
- 统一输出目录结构

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

MIT License

## 联系方式

- 作者: MiniMax Agent
- 邮箱: support@minimax.chat
- 项目地址: [GitHub Repository]

## 更新日志

### v1.0.0 (2025-02-05)
- 初始版本发布
- 完整的RMSD分析功能
- 多种可视化选项
- 批量处理支持
- 详细文档和示例