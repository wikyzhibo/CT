"""
UI 组件可调参数

集中存放布局、字号、尺寸等，便于统一调整风格而无需改各组件实现。
各组件从本模块的 ui_params 单例读取；启动时也可替换为从 JSON 等加载的实例。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class MainWindowParams:
    """主窗口：三栏布局宽度、初始几何、中央间距。"""

    left_panel_width: int = 300      # 左栏固定宽度 (px)
    right_panel_width: int = 300     # 右栏固定宽度 (px)
    initial_x: int = 100             # 窗口初始左上角 x
    initial_y: int = 100             # 窗口初始左上角 y
    initial_width: int = 1350        # 窗口初始宽度
    initial_height: int = 1000       # 窗口初始高度
    central_spacing: int = 12        # 左/中/右三栏之间的间距 (px)
    
    # ───────── 无边框窗口参数 ─────────
    window_corner_radius: int = 20   # 窗口圆角半径 (px)
    shadow_blur_radius: int = 20     # 阴影模糊半径 (px)
    shadow_offset: int = 5           # 阴影偏移 (px)
    shadow_margin: int = 40          # 窗口边距（为阴影留空）(px)
    drag_region_height: int = 80     # 顶部拖动区域高度 (px)


@dataclass
class StatsPanelParams:
    """左侧统计面板：KPI 仪表盘、摘要、RELEASE TIME、HISTORY。"""

    # ───────── 布局间距 ─────────
    layout_spacing: int = 12         # 整体垂直间距
    layout_margins: Tuple[int, int, int, int] = (10, 12, 10, 12)  # 内容边距 (左,上,右,下)
    section_spacing: int = 16        # 主要区块之间的间距
    group_spacing: int = 8           # 组内控件间距
    
    # ───────── KPI 卡片参数 ─────────
    kpi_card_padding: int = 10       # KPI 卡片内边距
    kpi_card_radius: int = 6         # KPI 卡片圆角
    kpi_card_height: int = 60        # KPI 卡片固定高度
    kpi_card_spacing: int = 8        # KPI 卡片之间间距
    kpi_grid_columns: int = 2        # KPI 网格列数
    
    # ───────── 字号 (pt) ─────────
    section_title_font_pt: int = 14  # 区块标题
    kpi_label_font_pt: int = 10      # KPI 标签（如 TIME, STEP）
    kpi_value_font_pt: int = 18      # KPI 数值（大号）
    label_font_pt: int = 11          # 普通标签文字
    summary_key_font_pt: int = 10    # 摘要 Key 字号
    summary_value_font_pt: int = 11  # 摘要 Value 字号
    toolbox_tab_font_pt: int = 12    # GroupBox 标题
    release_font_pt: int = 12        # RELEASE TIME
    history_font_pt: int = 12        # HISTORY
    reward_detail_font_pt: int = 12  # REWARD 明细
    
    # ───────── 摘要布局参数 ─────────
    summary_row_spacing: int = 4     # 摘要行间距
    summary_key_width: int = 80      # 摘要 Key 列宽度
    summary_frame_padding: int = 8   # 摘要区域内边距
    
    # ───────── 进度条参数 ─────────
    progress_bar_height: int = 16    # 进度条高度 (px)
    progress_label_spacing: int = 4  # 进度条与标签间距
    
    # ───────── 固定高度区域 ─────────
    release_fixed_height: int = 80   # RELEASE TIME 固定高度
    history_fixed_height: int = 100  # HISTORY 固定高度
    history_line_count: int = 8      # HISTORY 显示最近 N 条
    
    # ───────── Rewards 区块参数 ─────────
    rewards_fixed_height: int = 100  # REWARDS 区块固定高度 (px)
    rewards_max_items: int = 6       # REWARDS 最多显示条目数
    rewards_item_font_pt: int = 12   # REWARDS 条目字号
    rewards_item_spacing: int = 2    # REWARDS 条目行间距
    
    # ───────── 视觉效果 ─────────
    font_family: str = "Consolas"    # 左栏统一字体
    use_metric_cards: bool = True    # 是否使用 KPI 卡片组件
    use_status_badges: bool = True   # 是否使用 StatusBadge 组件
    show_trend_indicators: bool = False  # 是否显示趋势指示器
    animate_values: bool = False     # 是否启用数值动画


@dataclass
class ControlPanelParams:
    """右侧控制面板：TRANSITIONS 按钮组、CONTROL 按钮与速度倍率。"""

    # ───────── 标题 - 基于 Typography 系统 ─────────
    title_font_size_px: int = 20           # 「TRANSITIONS」「CONTROL」(H3)

    # ───────── 布局间距 ─────────
    spacing_after_transitions: int = 8     # 减小间距
    spacing_after_control_title: int = 4   # 减小间距
    spacing_before_speed: int = 6          # 减小间距
    spacing_before_reset: int = 6          # 减小间距

    # ───────── 按钮样式 - 基于 Typography 系统 ─────────
    button_font_size_px: int = 18          # 普通控制按钮 - 减小以节省空间
    transition_button_font_size_px: int = 18  # TRANSITIONS 动作按钮 - 减小以节省空间
    button_padding_v: int = 6              # 垂直内边距 - 减小
    button_padding_h: int = 10             # 水平内边距 - 减小
    button_min_height: int = 28            # 最小高度 - 减小

    # ───────── 速度倍率 ─────────
    speed_options: List[int] = field(default_factory=lambda: [1, 2, 5, 10])


@dataclass
class CenterCanvasParams:
    """中心画布：7×4 网格格宽格高，与腔室/机械手卡片尺寸一致时用于对齐。"""

    cell_w: int = 140    # 网格格宽 (px)，与 chamber_item.w 接近
    cell_h: int = 140   # 网格格高 (px)，与 chamber_item.h 一致
    chamber_w: int = 130  # 腔室占位宽（与 ChamberItem 一致，布局用）
    chamber_h: int = 130  # 腔室占位高（与 ChamberItem 一致，布局用）


@dataclass
class ChamberItemParams:
    """腔室卡片 (QGraphicsItem)：尺寸、状态闪烁、网格、晶圆与进度环。"""

    w: int = 140              # 卡片宽度 (px)
    h: int = 140              # 卡片高度 (px)
    flash_ms: int = 450       # 状态变化时边框高亮持续时间 (ms)
    corner_radius: int = 10    # 圆角半径 (px)
    grid_step: int = 15       # 背景网格线间距 (px)
    led_size: int = 10        # 状态灯圆点直径 (px)
    font_family: str = "Consolas"
    
    # ───────── 字号 - 基于 Typography 系统 ─────────
    name_font_pt: int = 18    # 腔室名称 (H3)
    wafer_font_pt: int = 14   # 晶圆内主数字（剩余时间）(BODY)
    wafer_id_font_pt: int = 12   # 晶圆内 #token_id 次要字号 (CAPTION)
    extra_count_font_pt: int = 11   # 「+N」多晶圆提示 (SMALL)
    
    progress_ring_width: int = 3   # 进度环线宽 (px)
    progress_ring_offset: int = 4 # 进度环相对晶圆半径的外扩 (px)
    wafer_radius: int = 45    # 晶圆圆半径 (px)
    inner_margin: int = 2     # 卡片内边距（相对边框）
    text_margin: int = 6      # 文字与边缘间距


@dataclass
class RobotItemParams:
    """机械手卡片 (QGraphicsItem)：与腔室同尺寸，腔室式布局（名称在上、晶圆居中）。"""

    w: int = 140              # 与腔室一致 (px)
    h: int = 140              # 与腔室一致 (px)
    flash_ms: int = 450       # IDLE↔BUSY 变化时边框高亮持续时间 (ms)
    corner_radius: int = 10    # 圆角半径 (px)
    grid_step: int = 15       # 背景网格线间距 (px)
    led_size: int = 10        # 状态灯圆点直径 (px)
    font_family: str = "Consolas"
    
    # ───────── 字号 - 基于 Typography 系统 ─────────
    title_font_pt: int = 14   # 机械手名称（TM2/TM3）(H3)
    status_font_pt: int = 11  # BUSY/IDLE (SMALL)
    wafers_font_pt: int = 10  # 「Wafers: N」(CAPTION)
    
    led_offset: int = 8       # 状态灯距左上角偏移 (px)
    title_left_offset: int = 24  # 标题文字左偏移（为 LED 留空）(px)
    inner_margin: int = 2    # 卡片内边距
    text_margin: int = 8     # 文字与边缘间距


@dataclass
class UIParams:
    """所有 UI 可调参数入口：聚合各子配置，各组件按需读取对应子类。"""

    main_window: MainWindowParams = field(default_factory=MainWindowParams)
    stats_panel: StatsPanelParams = field(default_factory=StatsPanelParams)
    control_panel: ControlPanelParams = field(default_factory=ControlPanelParams)
    center_canvas: CenterCanvasParams = field(default_factory=CenterCanvasParams)
    chamber_item: ChamberItemParams = field(default_factory=ChamberItemParams)
    robot_item: RobotItemParams = field(default_factory=RobotItemParams)


# 默认单例，各组件从此读取；也可在启动时替换为从文件加载的实例
ui_params = UIParams()
