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

    left_panel_width: int = 460      # 左栏固定宽度 (px)，放大字号后略增以容纳
    right_panel_width: int = 300     # 右栏固定宽度 (px)
    initial_x: int = 100             # 窗口初始左上角 x
    initial_y: int = 100             # 窗口初始左上角 y
    initial_width: int = 1520        # 窗口初始宽度
    initial_height: int = 1235       # 窗口初始高度
    central_spacing: int = 12        # 左/中/右三栏之间的间距 (px)


@dataclass
class StatsPanelParams:
    """左侧统计面板：KPI、System/Chambers/Robots 摘要、RELEASE TIME、HISTORY。"""

    layout_spacing: int = 14         # 整体垂直间距
    layout_margins: Tuple[int, int, int, int] = (14, 16, 14, 16)  # 内容边距 (左,上,右,下)
    group_spacing: int = 12          # 组内控件间距
    # 字号 (pt)，放大以提升可读性
    label_font_pt: int = 50          # KPI 下方标签（如 PROGRESS）
    kpi_font_pt: int = 30           # KPI 数字（TIME/STEP/REWARD）
    summary_font_pt: int = 30       # System/Chambers/Robots 摘要正文
    toolbox_tab_font_pt: int = 30    # ToolBox 标签（System/Chambers/Robots）
    release_font_pt: int = 20       # RELEASE TIME 文本框
    history_font_pt: int = 20       # HISTORY 文本框
    reward_detail_font_pt: int = 20  # REWARD 明细行字号
    # 尺寸与行数
    progress_bar_height: int = 16   # 进度条高度 (px)
    release_min_height: int = 100   # RELEASE TIME 最小高度 (px)
    history_min_height: int = 100   # HISTORY 最小高度 (px)
    summary_frame_padding: int = 10 # 摘要每页内边距
    history_line_count: int = 6     # HISTORY 显示最近 N 条
    font_family: str = "Consolas"   # 左栏统一字体


@dataclass
class ControlPanelParams:
    """右侧控制面板：TRANSITIONS 按钮组、CONTROL 按钮与速度倍率。"""

    # ───────── 标题 ─────────
    title_font_size_px: int = 20           # 「TRANSITIONS」「CONTROL」

    # ───────── 布局间距 ─────────
    spacing_after_transitions: int = 12
    spacing_after_control_title: int = 6
    spacing_before_speed: int = 8
    spacing_before_reset: int = 8

    # ───────── 按钮样式（你现在缺的就是这些） ─────────
    button_font_size_px: int = 18          # 普通控制按钮
    transition_button_font_size_px: int = 18  # TRANSITIONS 动作按钮
    button_padding_v: int = 8
    button_padding_h: int = 12
    button_min_height: int = 32

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
    corner_radius: int = 6    # 圆角半径 (px)
    grid_step: int = 12       # 背景网格线间距 (px)
    led_size: int = 10        # 状态灯圆点直径 (px)
    font_family: str = "Consolas"
    name_font_pt: int = 17    # 腔室名称字号
    wafer_radius: int = 42    # 晶圆圆半径 (px)
    progress_ring_width: int = 3   # 进度环线宽 (px)
    progress_ring_offset: int = 4 # 进度环相对晶圆半径的外扩 (px)
    wafer_font_pt: int = 18   # 晶圆内 token_id 字号
    extra_count_font_pt: int = 14   # 「+N」多晶圆提示字号
    inner_margin: int = 2     # 卡片内边距（相对边框）
    text_margin: int = 6      # 文字与边缘间距


@dataclass
class RobotItemParams:
    """机械手卡片 (QGraphicsItem)：与腔室同尺寸，腔室式布局（名称在上、晶圆居中）。"""

    w: int = 140              # 与腔室一致 (px)
    h: int = 140              # 与腔室一致 (px)
    flash_ms: int = 450       # IDLE↔BUSY 变化时边框高亮持续时间 (ms)
    corner_radius: int = 6    # 圆角半径 (px)
    grid_step: int = 12       # 背景网格线间距 (px)
    led_size: int = 10        # 状态灯圆点直径 (px)
    font_family: str = "Consolas"
    title_font_pt: int = 14   # 机械手名称（TM2/TM3）字号
    status_font_pt: int = 12  # BUSY/IDLE 字号
    wafers_font_pt: int = 10  # 「Wafers: N」字号
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
