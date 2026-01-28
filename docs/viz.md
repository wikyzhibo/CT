# viz.py 可视化模块文档

## 概述

`viz.py` 是一个用于交互式调试连续 Petri 网环境的可视化程序（Pygame 版本），采用 UI/UX 优化设计。该模块通过动画显示腔室、晶圆和运输状态，帮助用户直观地理解和调试 Petri 网环境。

**设计风格**: Terminal Green + Tactical Dark 工业控制面板风格

## 使用方法

```bash
python -m solutions.Continuous_model.viz
```

### 命令行参数

- `--model`, `-m`: 指定模型文件路径
- `--no-model`: 不加载模型

### 交互方式

- **鼠标**: 点击动作按钮执行动作
- **键盘快捷键**（仅控制按钮）:
  - `W`: 等待动作
  - `R`: 随机动作
  - `M`: 模型单步预测
  - `A`: 模型自动模式开关
  - `Space`: 重置环境
  - `ESC`: 退出程序
- **注意**: 变迁按钮不再使用键盘快捷键（1-9），只能通过鼠标点击

## 主要类

### ColorTheme

工业控制面板风格配色方案 - Cyberpunk Terminal + Industrial Control

#### 主要颜色

- **主色调**:
  - `primary`: Matrix Green (#00FF41) - 活跃/正常状态
  - `secondary`: 深绿 (#008F11) - 完成状态
  - `accent`: 青色 (#00C8FF) - 强调色

- **状态色**:
  - `success`: Matrix Green - 成功/正常
  - `warning`: 警告黄 (#FFB800)
  - `danger`: 警报红 (#FF3333)
  - `info`: 信息蓝 (#3B82F6)

- **背景层次**:
  - `bg_deepest`: 最深背景 (18, 22, 30)
  - `bg_deep`: 卡片背景 (28, 35, 45)
  - `bg_surface`: 表面层 (40, 48, 60)
  - `bg_elevated`: 提升层 (55, 65, 78)

- **按钮分类颜色**:
  - `btn_transition`: Cyan - 变迁操作
  - `btn_wait`: Yellow - 等待
  - `btn_random`: Purple - 随机
  - `btn_model`: Green - 模型单步
  - `btn_auto`: Orange - 自动模式
  - `btn_speed`: Blue - 速度控制
  - `btn_reset`: Red - 重置

#### 工具方法

- `lerp_color(c1, c2, t)`: 颜色线性插值
- `dim_color(color, factor)`: 降低颜色亮度
- `brighten_color(color, factor)`: 提高颜色亮度
- `with_alpha(color, alpha)`: 添加透明度

### AnimationManager

管理所有动画效果，支持 reduced-motion 和动画开关。

#### 主要功能

- **动画控制**:
  - `should_animate()`: 检查是否应该播放动画
  - `toggle_animations()`: 切换动画开关
  - `toggle_reduced_motion()`: 切换减少动画模式

- **动画效果**:
  - `pulse(frequency, min_val, max_val)`: 脉冲动画，使用 ease-out 曲线
  - `blink(frequency)`: 闪烁动画，返回 True/False
  - `flow_offset(speed)`: 流线动画偏移
  - `ease_out(progress)`: 缓出动画曲线
  - `ease_in_out(progress)`: 缓入缓出动画曲线

- **闪烁动画**:
  - `add_flash(key, duration)`: 添加闪烁动画
  - `get_flash_alpha(key)`: 获取闪烁透明度

### WaferRenderer

增强的晶圆渲染器，支持进度环、发光效果和状态显示。

#### 主要方法

- `get_status_color(stay_time, proc_time, place_type)`: 
  获取晶圆状态颜色和状态名
  - 返回: `(颜色, 状态名)`
  - 状态层次:
    - Level 1 (关键警报): 红色 + 脉冲动画
    - Level 2 (核心数据): Matrix Green
    - Level 3 (辅助信息): 灰色，无动画

- `draw(screen, x, y, stay_time, proc_time, place_type, token_id, wafer_color, font_small, font_tiny)`:
  绘制晶圆（带进度环和发光）
  - 显示内容:
    - 加工腔室: 倒计时数字 + Token ID
    - 非加工腔室: Token ID + 停留时间
    - 进度环: 仅加工腔室显示

### SparklineRenderer

迷你趋势图渲染器，用于显示产能和滞留时间趋势。

#### 主要方法

- `draw(screen, x, y, width, height, data, color, show_fill, show_current)`:
  绘制迷你趋势图
  - 最多显示 20 个数据点
  - 支持填充区域和当前值高亮

### ChamberRenderer

腔室渲染器，支持 LED 状态条、阴影和网格背景。

#### 主要方法

- `get_chamber_status(wafers, proc_time)`: 获取腔室状态
  - 返回: "idle", "active", "warning", "danger"

- `draw(screen, name, x, y, width, height, proc_time, wafers, font_large, font_small)`:
  绘制加工腔室
  - 显示 LED 状态条
  - 网格背景
  - 阴影效果

- `draw_transport(screen, name, x, y, width, height, wafers, font_tiny)`:
  绘制运输位
  - 流动指示线
  - 超时警告

### Button

按钮类，支持分组、快捷键、悬停动画和类型颜色。

#### 按钮类型

- `transition`: 变迁按钮 (Cyan)
- `wait`: 等待按钮 (Yellow)
- `random`: 随机按钮 (Purple)
- `model`: 模型单步 (Green)
- `auto`: 自动模式 (Orange)
- `speed`: 速度按钮 (Blue)
- `reset`: 重置按钮 (Red)

#### 主要方法

- `get_type_color()`: 获取按钮类型对应的颜色
- `update()`: 更新动画状态（悬停动画）
- `draw(screen, font, font_small)`: 绘制按钮
  - 支持悬停效果
  - 激活状态发光
  - 快捷键标签显示
- `check_hover(pos)`: 检查鼠标悬停
- `check_click(pos)`: 检查点击

### PetriVisualizer

Petri 网可视化器主类，UI/UX 优化版，支持多腔室网格布局。

#### 窗口配置

- **窗口尺寸**: 1820 x 1235 (适配 1440p 显示器)
- **左侧面板宽度**: 416
- **右侧面板宽度**: 300
- **腔室大小**: 130 x 130
- **中心机械手区域**: 170 x 170

#### 布局配置

**腔室映射**:
- LP1/LP2 -> LLA (起点)
- LP_done -> LLB (终点)
- s1 -> PM7 (machine=0) / PM8 (machine=1)
- s2 -> LLC
- s3 -> PM1/PM2/PM3/PM4 (machine=0/1/2/3)
- s4 -> LLD
- s5 -> PM9 (machine=0) / PM10 (machine=1)

**机械手分配**:
- TM2: d_s1, d_s2, d_s5, d_LP_done
- TM3: d_s3, d_s4

#### 主要方法

##### 初始化

- `__init__(env, model_path)`: 初始化可视化器
  - 设置窗口和字体
  - 创建渲染器
  - 设置布局和按钮
  - 加载模型（可选）

- `load_model(model_path)`: 加载训练好的 PPO 模型
- `get_model_action()`: 使用模型获取动作

##### 布局设置

- `_setup_layout()`: 设置布局位置 - 双机械手网格布局
- `_setup_buttons()`: 设置动作按钮 - 分组布局，带类型颜色
- `_setup_shortcuts()`: 设置快捷键映射

##### 数据收集

- `_collect_wafer_info()`: 收集各库所中晶圆信息，映射到显示名称
- `_update_trend_data()`: 更新趋势数据（用于迷你图）
  - 吞吐量
  - 平均滞留时间
  - 设备利用率

##### 绘制方法

- `draw()`: 绘制整个界面
  - 左侧面板: 系统监控
  - 右侧面板: 控制面板
  - 中心区域: 腔室和机械手

- `_draw_left_panel()`: 绘制左侧面板
  - 关键指标（TIME, PROGRESS）
  - 可折叠统计分组（系统/腔室/机械手滞留时间）
  - 动作历史（最近6步）
  - 详细奖励分解（最近一步的非零奖励/惩罚项）
  - 趋势数据

- `_draw_right_panel()`: 绘制右侧面板
  - 控制按钮组
  - 速度控制
  - 快捷键提示

- `_draw_active_chamber()`: 绘制活跃腔室
- `_draw_idle_chamber()`: 绘制闲置腔室
- `_draw_robot_buffer()`: 绘制双机械手缓冲区
- `_draw_single_robot()`: 绘制单个机械手
- `_draw_robot_connections()`: 绘制机械手到腔室的折线连接

##### 事件处理

- `handle_events()`: 处理事件，返回选中的动作或 None
  - 鼠标点击
  - 键盘快捷键
  - 折叠区域点击

- `_handle_collapse_click(pos)`: 处理折叠区域的点击

##### 动作执行

- `step_action(action)`: 执行动作并返回结果
- `_execute_action(action)`: 执行动作并更新状态
- `reset()`: 重置环境

##### 主循环

- `run()`: 主循环
  - 事件处理
  - 自动模式执行
  - 界面绘制
  - 60 FPS 刷新率

#### 状态管理

- `step_count`: 步数计数
- `total_reward`: 总奖励
- `last_reward`: 上次奖励
- `auto_mode`: 自动模式开关
- `auto_speed`: 速度倍率 (1x/2x/5x/10x)
- `done`: 是否完成
- `action_history`: 动作历史记录
- `trend_data`: 趋势数据

## 功能特性

### 1. 实时状态监控

- **关键指标**: TIME, PROGRESS (完成百分比)
- **次要指标**: STEP, REWARD, TPT (吞吐量)
- **统计信息**: 
  - 系统滞留时间（平均/最大/差值）
  - 腔室滞留时间（按腔室分组）
  - 机械手滞留时间（TM2/TM3）
- **动作历史**: 显示最近6步的动作和总奖励
- **详细奖励分解**: 在 HISTORY 下方显示最近一步的详细奖励和惩罚项
  - 显示步数和动作名称（格式：Step #XX - ACTION_NAME）
  - 只显示非零的奖励/惩罚项，过滤掉所有值为0的项
  - 奖励项（正值）显示为绿色，惩罚项（负值）显示为红色
  - 惩罚项始终显示为负数（如果原始值为正数会自动转换为负数）
  - 包含的奖励/惩罚类型：
    - 加工奖励 (proc_reward)
    - 安全裕量 (safe_reward)
    - 完工奖励 (wafer_done_bonus)
    - 完成奖励 (finish_bonus)
    - 超时惩罚 (penalty)
    - 预警惩罚 (warn_penalty)
    - 运输惩罚 (transport_penalty)
    - 堵塞惩罚 (congestion_penalty)
    - 时间成本 (time_cost)
    - 释放违规 (release_violation_penalty)
    - 报废惩罚 (scrap_penalty)

### 2. 可视化元素

- **晶圆显示**:
  - 进度环（加工腔室）
  - 状态颜色（正常/警告/危险）
  - 倒计时/停留时间
  - Token ID
  - 路线颜色标识

- **腔室显示**:
  - LED 状态条
  - 网格背景
  - 阴影效果
  - 状态指示

- **机械手显示**:
  - 状态指示灯（BUSY/IDLE）
  - 连接线（到各腔室）
  - 运输中晶圆显示

### 3. 交互功能

- **动作按钮**: 分组显示，带类型颜色
  - 变迁按钮显示友好格式（如 `LP1→PM7/PM8`, `PM7/PM8`），使用简短格式提高可读性
  - 变迁按钮不再使用键盘快捷键，只能通过鼠标点击
  - 控制按钮（WAIT、Random、Model等）保留键盘快捷键
- **自动模式**: 模型自动执行，可调节速度
- **折叠面板**: 可折叠统计分组，节省空间
- **动作历史**: 显示最近动作和奖励

### 4. 动画效果

- **脉冲动画**: 关键警报状态
- **闪烁动画**: 危险状态提示
- **流线动画**: 运输指示
- **悬停动画**: 按钮交互反馈
- **发光效果**: 激活状态指示

### 5. 模型集成

- 支持加载训练好的 PPO 模型
- 模型单步预测
- 模型自动模式
- 动作掩码支持

## 技术细节

### 依赖库

- `pygame`: 图形界面
- `numpy`: 数值计算
- `torch`: 深度学习框架
- `tensordict`: 张量字典
- `torchrl`: 强化学习工具

### 性能优化

- 60 FPS 刷新率
- 动画开关支持（减少动画模式）
- 数据点限制（趋势图最多 20 个点）
- 渲染优化（减少不必要的重绘）

### 可访问性

- 支持 reduced-motion 模式
- 动画开关
- 清晰的视觉层次
- 颜色对比度优化

## 使用示例

### 基本使用

```python
from solutions.PPO.enviroment import Env_PN
from solutions.Continuous_model.viz import PetriVisualizer

# 创建环境
env = Env_PN()

# 创建可视化器（不加载模型）
visualizer = PetriVisualizer(env)

# 运行可视化
visualizer.run()
```

### 加载模型

```python
# 创建可视化器并加载模型
model_path = "path/to/model.pt"
visualizer = PetriVisualizer(env, model_path=model_path)

# 运行可视化
visualizer.run()
```

### 命令行使用

```bash
# 使用默认模型
python -m solutions.Continuous_model.viz

# 指定模型路径
python -m solutions.Continuous_model.viz --model path/to/model.pt

# 不加载模型
python -m solutions.Continuous_model.viz --no-model
```

## 注意事项

1. **窗口大小**: 默认窗口大小为 1820x1235，适配 1440p 显示器
2. **字体要求**: 需要系统安装 Microsoft YaHei 和 Consolas 字体
3. **模型格式**: 模型文件必须是 PyTorch 格式，且与环境的动作空间匹配
4. **性能**: 在低性能设备上可以关闭动画以提高性能

## 更新日志

- **UI/UX 优化版**: 
  - Terminal Green + Tactical Dark 工业控制面板风格
  - 1.3x 放大版本，适配 1440p 显示器
  - 可折叠统计分组
  - 增强的动画效果
  - 双机械手支持
