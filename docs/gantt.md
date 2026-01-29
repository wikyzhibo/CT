# 甘特图绘制工具规范

## 概述

本规范描述用于可视化 Petri 网调度结果的甘特图绘制工具。该工具基于 matplotlib 实现，支持多阶段、多机器的调度可视化，并可选显示机械手占用情况。

## 核心数据结构

### Requirement: Op 数据类

系统 SHALL 使用 `Op` 数据类表示单个操作（工序）：

```python
@dataclass
class Op:
    job: int           # 作业编号
    stage: int         # 工序阶段
    machine: int       # 机器编号
    start: float       # 开始时间
    proc_end: float    # 加工结束时间
    end: float         # 占用结束时间（包含驻留时间）
    is_arm: bool = False      # 是否为机械手动作
    kind: int = -1            # 动作类型（0=取，1=放）
    from_loc: str = ""        # 取的位置
    to_loc: str = ""          # 放的位置
```

#### Scenario: 标准工序操作

- **GIVEN** 一个作业在某个阶段的加工操作
- **WHEN** 创建 Op 实例时
- **THEN** `is_arm` 应为 `False`
- **AND** `start` < `proc_end` <= `end`
- **AND** `proc_end` 到 `end` 之间表示驻留时间

#### Scenario: 机械手操作

- **GIVEN** 一个机械手搬运操作
- **WHEN** 创建 Op 实例时
- **THEN** `is_arm` 应为 `True`
- **AND** 应设置 `from_loc` 和 `to_loc` 表示搬运路径
- **AND** `kind` 为 0 表示"取"，1 表示"放"

## 甘特图绘制功能

### Requirement: 多阶段多机器甘特图

系统 SHALL 提供 `plot_gantt_hatched_residence()` 函数绘制多阶段多机器调度甘特图。

#### 函数签名

```python
def plot_gantt_hatched_residence(
    ops: List[Op],                           # 操作列表
    proc_time: Dict[int, float],             # 各阶段处理时间
    capacity: Dict[int, int],                # 各阶段机器数量
    n_jobs: int,                             # 作业总数
    out_path: str,                           # 输出路径
    arm_info: dict,                          # 机械手信息
    with_label: bool = True,                 # 是否显示作业标签
    no_arm: bool = True,                     # 是否隐藏机械手泳道
    policy: int = None,                      # 调度策略编号
    stage_module_names: Dict[int, List[str]] = None  # 阶段到模块名称映射
)
```

#### Scenario: 基本甘特图绘制

- **GIVEN** 一组操作列表和调度参数
- **WHEN** 调用 `plot_gantt_hatched_residence()` 时
- **THEN** 应生成包含所有工序的甘特图
- **AND** 每个工序用深色矩形表示加工时间
- **AND** 驻留时间用浅色斜线矩形表示
- **AND** 不同作业使用不同颜色（turbo 色谱）

#### Scenario: 动态图像尺寸

- **GIVEN** 不同时间跨度的调度结果
- **WHEN** 绘制甘特图时
- **THEN** 图像宽度应根据 makespan 动态调整
- **AND** 宽度范围为 16-30 英寸
- **AND** 高度根据泳道数量自动调整（每个泳道 0.8 英寸）

### Requirement: 泳道布局

系统 SHALL 为每个阶段-机器组合创建独立的泳道。

#### Scenario: 泳道命名

- **GIVEN** 阶段 s 和机器编号 m
- **WHEN** 生成泳道标签时
- **THEN** 默认格式为 `S{s}-M{m}(模块名)`
- **AND** 如果提供 `stage_module_names`，则使用对应的模块名称
- **AND** 泳道标签显示在左侧，右对齐

#### Scenario: 泳道顺序

- **GIVEN** 多个阶段和多台机器
- **WHEN** 排列泳道时
- **THEN** 按阶段从小到大排列
- **AND** 同一阶段内按机器编号从小到大排列

### Requirement: 加工与驻留时间可视化

系统 SHALL 区分加工时间和驻留时间，使用不同的视觉样式。

#### Scenario: 加工时间表示

- **GIVEN** 一个操作 `op`
- **WHEN** 绘制 `[op.start, op.proc_end)` 区间时
- **THEN** 应使用深色填充（alpha=0.9）
- **AND** 使用作业对应的颜色
- **AND** 不绘制内部边框（边框由统一外边框绘制）

#### Scenario: 驻留时间表示

- **GIVEN** 一个操作 `op` 且 `op.end > op.proc_end`
- **WHEN** 绘制 `[op.proc_end, op.end)` 区间时
- **THEN** 应使用浅色填充（alpha=0.35）
- **AND** 添加交叉线填充图案（hatch="xxx"），比斜线更明显，用于清晰区分加工段和驻留段
- **AND** 不绘制内部边框（边框由统一外边框绘制）

#### Scenario: 统一外边框

- **GIVEN** 一个操作 `op` 且 `op.end > op.start`
- **WHEN** 绘制完加工段和驻留段后
- **THEN** 应为整个区间 `[op.start, op.end)` 添加统一外边框
- **AND** 边框颜色为深灰色（#1E293B），线宽 0.8
- **AND** 外边框包裹整个区间，使加工段和驻留段看起来像一个统一的整体
- **AND** 在 `proc_end` 处不绘制分隔线，实现无缝连接

### Requirement: 机械手动作可视化

系统 SHALL 使用不同颜色表示机械手的取放路径。

#### Scenario: 机械手路径颜色分配

- **GIVEN** 一组机械手操作
- **WHEN** 绘制机械手动作时
- **THEN** 应收集所有唯一的 `(from_loc, to_loc)` 路径
- **AND** 为每条路径分配唯一颜色（tab20 色谱）
- **AND** 在图例中显示路径标签（格式：`from_loc → to_loc`）

#### Scenario: 机械手动作矩形

- **GIVEN** 一个机械手操作 `op` 且 `op.is_arm = True`
- **WHEN** 绘制该操作时
- **THEN** 应使用路径对应的颜色
- **AND** alpha=1.0（不透明）
- **AND** 不添加作业标签

### Requirement: 机械手占用泳道（可选）

系统 SHALL 支持显示机械手占用情况的额外泳道（当 `no_arm=False` 时）。

#### Scenario: 机械手占用区间计算

- **GIVEN** 一个作业的相邻工序 A 和 B
- **WHEN** 计算机械手占用区间时
- **THEN** 占用区间为 `[A.end, B.start)`
- **AND** 区间分配给对应的机械手（ARM1 或 ARM2）

#### Scenario: 机械手占用颜色

- **GIVEN** 某个时刻 t 的机械手占用数量 occ
- **WHEN** 绘制该时刻的机械手泳道时
- **THEN** occ=1 时使用绿色
- **AND** occ=2 时使用蓝色
- **AND** occ>=3 时使用红色（表示冲突）

### Requirement: 图表元数据

系统 SHALL 在图表中显示调度结果的元数据。

#### Scenario: 图表标题

- **GIVEN** 绘制完成的甘特图
- **WHEN** 设置标题时
- **THEN** 应包含阶段数量
- **AND** 应包含 makespan（完工时间）
- **AND** 格式：`{S}-Stage Mixed Routes (A:[...], B:[...])|makespan={t_max:.1f}s`

#### Scenario: 输出文件命名

- **GIVEN** 输出路径 `out_path` 和策略编号 `policy`
- **WHEN** 保存图像时
- **THEN** 文件名应附加策略和作业数：`{policy}_job{n_jobs}.png`
- **AND** 策略映射：0='pdr', 1='random', 2='rl'

### Requirement: 高分辨率输出

系统 SHALL 输出高分辨率图像以确保清晰度。

#### Scenario: DPI 设置

- **GIVEN** 生成的甘特图
- **WHEN** 保存为 PNG 文件时
- **THEN** 应使用 300 DPI（无机械手泳道）
- **OR** 使用 400 DPI（包含机械手泳道）
- **AND** 使用 `bbox_inches='tight'` 自动裁剪空白

### Requirement: 作业标签显示

系统 SHALL 根据作业数量自动决定是否显示标签。

#### Scenario: 作业数量少时显示标签

- **GIVEN** 作业数量 <= 50
- **WHEN** `with_label=True` 时
- **THEN** 应在每个操作矩形中心显示 `J{job}` 标签
- **AND** 字体颜色根据背景色亮度动态选择（白色或深色）
- **AND** 字体大小为 11，加粗显示
- **AND** 添加文字描边确保对比度≥4.5:1（WCAG标准）

#### Scenario: 作业数量多时隐藏标签

- **GIVEN** 作业数量 > 50
- **WHEN** 绘制甘特图时
- **THEN** 应自动禁用标签显示（避免重叠）

## 辅助功能

### Requirement: 阶段数量验证

系统 SHALL 提供 `_num_stages()` 函数验证阶段配置完整性。

#### Scenario: 完整阶段配置

- **GIVEN** `proc_time = {1: 10, 2: 20, 3: 15}`
- **WHEN** 调用 `_num_stages(proc_time)` 时
- **THEN** 应返回 3
- **AND** 不抛出异常

#### Scenario: 不完整阶段配置

- **GIVEN** `proc_time = {1: 10, 3: 15}`（缺少阶段 2）
- **WHEN** 调用 `_num_stages(proc_time)` 时
- **THEN** 应抛出 `ValueError`
- **AND** 错误消息应指明缺失的阶段编号

## 配置参数

### 默认配置

- **泳道高度**：4 单位
- **泳道间隙**：1 单位
- **机械手泳道高度**：6 单位
- **机械手泳道间隙**：6 单位
- **图像宽度**：30 英寸（固定）
- **图像高度**：8 英寸（固定）
- **背景色**：#F8FAFC（浅灰白）
- **图表背景色**：#FFFFFF（白色）
- **边框颜色**：#1E293B（深灰蓝）
- **网格线颜色**：#94A3B8（中灰色）

### 颜色方案

- **作业颜色**：turbo 色谱（连续渐变）
- **机械手路径颜色**：tab20 色谱（离散颜色）
- **机械手占用颜色**：
  - 绿色：单一占用
  - 蓝色：双重占用
  - 红色：三重及以上占用（冲突）
- **边框颜色**：深灰色（#1E293B），统一所有矩形边框
- **背景色**：浅色背景（#F8FAFC），图表区域白色（#FFFFFF）

### 视觉优化特性

#### 网格线

- **垂直网格线**：浅灰色虚线（#94A3B8），透明度30%
- **用途**：帮助读取时间轴，提高可读性
- **样式**：`linestyle='--', linewidth=0.5, alpha=0.3`

#### 字体和间距

- **标题字体**：17pt，深色（#0F172A），正常字重
- **轴标签字体**：14pt，深色（#1E293B）
- **时间刻度字体**：11pt，中灰色（#475569）
- **作业标签字体**：11pt，加粗，带描边增强对比度
- **图例字体**：12-13pt，标题13-14pt

#### 时间轴优化

- **主要刻度**：自动计算，间隔为 max(50, makespan/10)
- **标签格式**：整数秒数（如：100, 200, 300）
- **网格对齐**：网格线与主要刻度对齐

### 可访问性特性

#### 对比度优化

- **文字对比度**：所有文字与背景的对比度≥4.5:1（符合WCAG AA标准）
- **动态文字颜色**：根据背景色亮度自动选择白色或深色文字
- **文字描边**：为标签添加描边，确保在任何背景下都清晰可读

#### 辅助函数

系统提供以下辅助函数确保可访问性：

- `_get_luminance(rgb)` - 计算颜色的相对亮度
- `_get_contrast_ratio(color1, color2)` - 计算两个颜色的对比度比率
- `_get_text_color(bg_color)` - 根据背景色返回合适的文字颜色

#### 视觉区分

- **加工时间**：深色填充（alpha=0.9），清晰可见
- **驻留时间**：浅色填充（alpha=0.35）+ 交叉线图案（hatch="xxx"），比斜线更明显，清晰区分加工段和驻留段
- **边框统一**：所有矩形使用统一的深灰色外边框（#1E293B，线宽0.8），提高一致性

#### 统一视觉设计

- **整体感**：加工段和驻留段通过统一外边框形成一个视觉整体
- **无缝连接**：在 `proc_end` 处不绘制分隔线，两个段无缝连接
- **清晰区分**：驻留段使用交叉线图案（`hatch="xxx"`）与加工段形成明显视觉区分，alpha=0.35使图案更明显，同时保持整体感
- **边框策略**：只绘制外边框，内部不绘制边框，避免视觉分割

## 使用示例

### 基本用法

```python
from visualization.plot import plot_gantt_hatched_residence, Op

ops = [
    Op(job=1, stage=1, machine=0, start=0, proc_end=30, end=35),
    Op(job=1, stage=2, machine=0, start=40, proc_end=120, end=125),
    # ... 更多操作
]

proc_time = {1: 30, 2: 80}
capacity = {1: 2, 2: 1}
arm_info = {"ARM1": [], "ARM2": [], "STAGE2ACT": {}}

plot_gantt_hatched_residence(
    ops=ops,
    proc_time=proc_time,
    capacity=capacity,
    n_jobs=10,
    out_path="results/",
    arm_info=arm_info,
    no_arm=True,
    policy=2  # RL策略
)
# 输出文件：results/rl_job10.png
```

### 自定义模块名称

```python
stage_module_names = {
    1: ["PM1", "PM2"],
    2: ["PM3"],
}

plot_gantt_hatched_residence(
    ops=ops,
    proc_time=proc_time,
    capacity=capacity,
    n_jobs=10,
    out_path="results/",
    arm_info=arm_info,
    stage_module_names=stage_module_names,
    no_arm=True,
    policy=2
)
# 泳道标签：S1-M0(PM1), S1-M1(PM2), S2-M0(PM3)
```

### 显示机械手占用

```python
arm_info = {
    "ARM1": ["u_LP_PM1", "t_PM1"],
    "ARM2": ["u_PM1_PM2", "t_PM2"],
    "STAGE2ACT": {
        1: ["t_PM1", "u_PM1_PM2"],
        2: ["t_PM2", "u_PM2_LP_done"],
    }
}

plot_gantt_hatched_residence(
    ops=ops,
    proc_time=proc_time,
    capacity=capacity,
    n_jobs=10,
    out_path="results/",
    arm_info=arm_info,
    no_arm=False,  # 显示机械手泳道
    policy=2
)
```

## 实现文件

- **主文件**：`visualization/plot.py`
- **依赖**：matplotlib, numpy
- **核心函数**：
  - `plot_gantt_hatched_residence()` - 主绘制函数
  - `_num_stages()` - 阶段数量验证
  - `_get_luminance()` - 计算颜色亮度（可访问性）
  - `_get_contrast_ratio()` - 计算对比度比率（可访问性）
  - `_get_text_color()` - 动态选择文字颜色（可访问性）

## 扩展性

### 未来可能的扩展

- 支持更多颜色方案选择
- 交互式甘特图（鼠标悬停显示详情）
- 导出为 SVG 格式（矢量图）
- 支持时间轴缩放和平移
- 添加关键路径高亮显示
