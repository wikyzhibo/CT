"""
Env_PN 可视化测试程序 (Pygame 版本) - UI/UX 优化版
用于交互式调试连续 Petri 网环境，通过动画显示腔室、晶圆和运输状态。

采用 Terminal Green + Tactical Dark 工业控制面板风格

用法：
    python -m solutions.Continuous_model.test_env

交互：
    鼠标点击动作按钮执行动作
    键盘快捷键: 1-9 变迁, W 等待, R 随机, M 模型, Space 重置
    ESC 或关闭窗口退出
"""

import sys
import os
import argparse
import math
import time as pytime
import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.insert(0, r"C:\Users\khand\OneDrive\code\dqn\CT")

import pygame
from torchrl.modules import ProbabilisticActor, MaskedCategorical
from torchrl.envs.utils import set_exploration_type, ExplorationType
from solutions.PPO.enviroment import Env_PN
from solutions.PPO.network.models import MaskedPolicyHead


# ============ 颜色主题类 ============
@dataclass
class ColorTheme:
    """工业控制面板风格配色方案 - Terminal Green + Tactical Dark"""
    # 主色调
    primary: Tuple[int, int, int] = (0, 255, 65)      # #00FF41 Matrix Green - 活跃/正常
    secondary: Tuple[int, int, int] = (0, 143, 17)    # #008F11 深绿 - 完成
    accent: Tuple[int, int, int] = (0, 200, 255)      # #00C8FF 青色强调
    
    # 状态色
    success: Tuple[int, int, int] = (0, 255, 65)      # 成功/正常 - Matrix Green
    warning: Tuple[int, int, int] = (255, 184, 0)     # #FFB800 警告黄
    danger: Tuple[int, int, int] = (255, 51, 51)      # #FF3333 警报红
    info: Tuple[int, int, int] = (59, 130, 246)       # #3B82F6 信息蓝
    
    # 背景层次 (调亮版本)
    bg_deepest: Tuple[int, int, int] = (30, 35, 45)   # 调亮的主背景
    bg_deep: Tuple[int, int, int] = (40, 48, 60)      # 调亮的卡片背景
    bg_surface: Tuple[int, int, int] = (55, 65, 78)   # 调亮的表面
    bg_elevated: Tuple[int, int, int] = (70, 80, 95)  # 调亮的提升层
    
    # 边框
    border: Tuple[int, int, int] = (80, 95, 115)      # 更亮的边框
    border_muted: Tuple[int, int, int] = (60, 72, 88) # 调亮的弱边框
    border_active: Tuple[int, int, int] = (0, 255, 65)# 活跃边框
    
    # 文字
    text_primary: Tuple[int, int, int] = (230, 237, 243)  # #E6EDF3 主文字
    text_secondary: Tuple[int, int, int] = (139, 148, 158)# #8B949E 次要文字
    text_muted: Tuple[int, int, int] = (110, 118, 129)    # #6E7681 弱化文字
    
    # 特殊效果
    glow_green: Tuple[int, int, int] = (0, 255, 65)
    glow_yellow: Tuple[int, int, int] = (255, 184, 0)
    glow_red: Tuple[int, int, int] = (255, 51, 51)
    
    @staticmethod
    def lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
        """颜色线性插值"""
        t = max(0, min(1, t))
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t)
        )
    
    @staticmethod
    def dim_color(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        """降低颜色亮度"""
        return (
            int(color[0] * factor),
            int(color[1] * factor),
            int(color[2] * factor)
        )
    
    @staticmethod
    def brighten_color(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        """提高颜色亮度"""
        return (
            min(255, int(color[0] * factor)),
            min(255, int(color[1] * factor)),
            min(255, int(color[2] * factor))
        )


# 全局主题实例
THEME = ColorTheme()


# ============ 动画管理器 ============
class AnimationManager:
    """管理所有动画效果"""
    
    def __init__(self):
        self.start_time = pytime.time()
        self.animations: Dict[str, Dict[str, Any]] = {}
        
    def get_time(self) -> float:
        """获取动画时间（秒）"""
        return pytime.time() - self.start_time
    
    def pulse(self, frequency: float = 1.0, min_val: float = 0.7, max_val: float = 1.0) -> float:
        """脉冲动画 - 返回 min_val 到 max_val 之间的值"""
        t = self.get_time()
        # sin 波在 0-1 之间
        wave = (math.sin(t * frequency * 2 * math.pi) + 1) / 2
        return min_val + wave * (max_val - min_val)
    
    def blink(self, frequency: float = 2.0) -> bool:
        """闪烁动画 - 返回 True/False"""
        t = self.get_time()
        return (t * frequency) % 1.0 < 0.5
    
    def flow_offset(self, speed: float = 50.0) -> float:
        """流线动画偏移"""
        t = self.get_time()
        return (t * speed) % 20  # 20像素周期
    
    def ease_out(self, progress: float) -> float:
        """缓出动画曲线"""
        return 1 - (1 - progress) ** 3
    
    def add_flash(self, key: str, duration: float = 0.3):
        """添加闪烁动画"""
        self.animations[key] = {
            'type': 'flash',
            'start': self.get_time(),
            'duration': duration
        }
    
    def get_flash_alpha(self, key: str) -> float:
        """获取闪烁透明度 (0-1, 1表示最亮)"""
        if key not in self.animations:
            return 0
        anim = self.animations[key]
        elapsed = self.get_time() - anim['start']
        if elapsed > anim['duration']:
            del self.animations[key]
            return 0
        # 快速淡出
        return 1 - (elapsed / anim['duration']) ** 0.5


# ============ 晶圆渲染器 ============
class WaferRenderer:
    """增强的晶圆渲染器 - 进度环、发光、状态"""
    
    def __init__(self, theme: ColorTheme, anim: AnimationManager):
        self.theme = theme
        self.anim = anim
        self.base_radius = 32
    
    def get_status_color(self, stay_time: int, proc_time: int, place_type: int) -> Tuple[Tuple[int, int, int], str]:
        """
        获取晶圆状态颜色和状态名
        返回: (颜色, 状态名)
        """
        if place_type == 1:  # 加工腔室
            if stay_time < proc_time:
                return self.theme.success, "processing"
            elif stay_time < proc_time + 15:
                return self.theme.warning, "done"
            elif stay_time < proc_time + 20:
                return self.theme.danger, "critical"
            else:
                return self.theme.danger, "scrapped"
        elif place_type == 2:  # 运输位
            if stay_time < 7:
                return self.theme.success, "normal"
            elif stay_time < 10:
                return self.theme.warning, "warning"
            else:
                return self.theme.danger, "overtime"
        else:
            return self.theme.success, "idle"
    
    def draw(self, screen: pygame.Surface, x: int, y: int, 
             stay_time: int, proc_time: int, place_type: int,
             token_id: int = -1, font_small: pygame.font.Font = None,
             font_tiny: pygame.font.Font = None):
        """绘制晶圆（带进度环和发光）"""
        color, status = self.get_status_color(stay_time, proc_time, place_type)
        radius = self.base_radius
        
        # 计算进度 (仅加工腔室显示)
        progress = 0.0
        if place_type == 1 and proc_time > 0:
            progress = min(1.0, stay_time / proc_time)
        
        # 发光效果 (警告/危险状态脉冲)
        glow_intensity = 0
        if status in ["warning", "critical", "overtime"]:
            glow_intensity = self.anim.pulse(2.0, 0.3, 1.0)
        elif status == "scrapped":
            glow_intensity = 1.0 if self.anim.blink(3.0) else 0.5
        
        # 绘制外发光
        if glow_intensity > 0:
            for i in range(3, 0, -1):
                glow_radius = radius + i * 4
                glow_alpha = int(50 * glow_intensity * (4 - i) / 3)
                glow_surface = pygame.Surface((glow_radius * 2 + 10, glow_radius * 2 + 10), pygame.SRCALPHA)
                glow_color = (*color, glow_alpha)
                pygame.draw.circle(glow_surface, glow_color, 
                                 (glow_radius + 5, glow_radius + 5), glow_radius)
                screen.blit(glow_surface, (x - glow_radius - 5, y - glow_radius - 5))
        
        # 绘制进度环背景
        if place_type == 1 and proc_time > 0:
            # 背景环
            pygame.draw.circle(screen, self.theme.bg_elevated, (x, y), radius + 4, 4)
            # 进度环 (使用弧线)
            if progress > 0:
                self._draw_arc(screen, x, y, radius + 4, 4, progress, color)
        
        # 绘制晶圆主体
        # 外边框
        pygame.draw.circle(screen, self.theme.border, (x, y), radius + 1)
        # 主体渐变效果 (简化为实心)
        inner_color = self.theme.dim_color(color, 0.7)
        pygame.draw.circle(screen, inner_color, (x, y), radius)
        # 高光
        highlight_color = self.theme.brighten_color(color, 1.3)
        pygame.draw.circle(screen, highlight_color, (x, y), radius, 2)
        
        # 内部信息
        if font_tiny:
            # Token ID
            if token_id >= 0:
                id_text = f"#{token_id}"
                id_surf = font_tiny.render(id_text, True, self.theme.text_primary)
                id_rect = id_surf.get_rect(centerx=x, centery=y - 7)
                screen.blit(id_surf, id_rect)
            
            # 停留时间
            time_text = f"{stay_time}s"
            time_color = self.theme.text_primary if status != "scrapped" else self.theme.danger
            time_surf = font_tiny.render(time_text, True, time_color)
            time_rect = time_surf.get_rect(centerx=x, centery=y + 7)
            screen.blit(time_surf, time_rect)
    
    def _draw_arc(self, screen: pygame.Surface, cx: int, cy: int, 
                  radius: int, width: int, progress: float, color: Tuple[int, int, int]):
        """绘制进度弧线"""
        # 使用多边形近似圆弧
        if progress <= 0:
            return
        
        points = []
        start_angle = -math.pi / 2  # 从顶部开始
        end_angle = start_angle + 2 * math.pi * progress
        
        # 外圈点
        outer_r = radius
        inner_r = radius - width
        
        steps = max(10, int(progress * 60))
        for i in range(steps + 1):
            angle = start_angle + (end_angle - start_angle) * i / steps
            points.append((
                cx + outer_r * math.cos(angle),
                cy + outer_r * math.sin(angle)
            ))
        
        # 内圈点（反向）
        for i in range(steps, -1, -1):
            angle = start_angle + (end_angle - start_angle) * i / steps
            points.append((
                cx + inner_r * math.cos(angle),
                cy + inner_r * math.sin(angle)
            ))
        
        if len(points) >= 3:
            pygame.draw.polygon(screen, color, points)


# ============ 腔室渲染器 ============
class ChamberRenderer:
    """腔室渲染器 - LED状态条、阴影、网格背景"""
    
    def __init__(self, theme: ColorTheme, anim: AnimationManager):
        self.theme = theme
        self.anim = anim
    
    def get_chamber_status(self, wafers: List[Dict], proc_time: int) -> str:
        """获取腔室状态"""
        if not wafers:
            return "idle"
        
        # 检查是否有危险状态的晶圆
        for wafer in wafers:
            stay = wafer["stay_time"]
            if wafer["type"] == 1:  # 加工腔室
                if stay >= proc_time + 15:
                    return "danger"
                elif stay >= proc_time:
                    return "warning"
        return "active"
    
    def draw(self, screen: pygame.Surface, name: str, x: int, y: int,
             width: int, height: int, proc_time: int,
             wafers: List[Dict], font_large: pygame.font.Font,
             font_small: pygame.font.Font):
        """绘制腔室"""
        status = self.get_chamber_status(wafers, proc_time)
        
        # LED 状态条颜色
        if status == "danger":
            led_color = self.theme.danger
            if self.anim.blink(2.0):
                led_color = self.theme.brighten_color(led_color, 1.3)
        elif status == "warning":
            led_color = self.theme.warning
        elif status == "active":
            led_color = self.theme.success
        else:
            led_color = self.theme.text_muted
        
        # 绘制阴影
        shadow_offset = 4
        shadow_rect = pygame.Rect(x + shadow_offset, y + shadow_offset, width, height)
        shadow_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (0, 0, 0, 80), (0, 0, width, height), border_radius=8)
        screen.blit(shadow_surface, (x + shadow_offset, y + shadow_offset))
        
        # 主体背景
        main_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(screen, self.theme.bg_deep, main_rect, border_radius=8)
        
        # 网格背景
        self._draw_grid(screen, x + 4, y + 20, width - 8, height - 24)
        
        # 边框
        border_color = self.theme.border if status == "idle" else led_color
        pygame.draw.rect(screen, border_color, main_rect, 2, border_radius=8)
        
        # 顶部 LED 状态条
        led_height = 6
        led_rect = pygame.Rect(x + 10, y + 6, width - 20, led_height)
        pygame.draw.rect(screen, self.theme.bg_elevated, led_rect, border_radius=3)
        
        # LED 活跃部分
        led_segments = 5
        seg_width = (width - 24) // led_segments
        active_segs = led_segments if status == "active" else (3 if status == "warning" else (led_segments if status == "danger" else 1))
        for i in range(active_segs):
            seg_x = x + 12 + i * (seg_width + 2)
            seg_rect = pygame.Rect(seg_x, y + 7, seg_width - 2, led_height - 2)
            pygame.draw.rect(screen, led_color, seg_rect, border_radius=2)
        
        # 腔室名称（上方）
        name_surf = font_large.render(name, True, self.theme.text_primary)
        name_rect = name_surf.get_rect(centerx=x + width // 2, bottom=y - 10)
        screen.blit(name_surf, name_rect)
        
        # 加工时间（名称下方）
        #if proc_time > 0:
        #    time_text = f"[{proc_time}s]"
        #    time_surf = font_small.render(time_text, True, self.theme.text_secondary)
        #    time_rect = time_surf.get_rect(centerx=x + width // 2, bottom=y - 30)
        #    screen.blit(time_surf, time_rect)
    
    def _draw_grid(self, screen: pygame.Surface, x: int, y: int, w: int, h: int):
        """绘制科技感网格"""
        grid_color = self.theme.border_muted
        spacing = 18
        
        # 垂直线
        for gx in range(x, x + w, spacing):
            pygame.draw.line(screen, grid_color, (gx, y), (gx, y + h), 1)
        
        # 水平线
        for gy in range(y, y + h, spacing):
            pygame.draw.line(screen, grid_color, (x, gy), (x + w, gy), 1)
    
    def draw_transport(self, screen: pygame.Surface, name: str, x: int, y: int,
                       width: int, height: int, wafers: List[Dict],
                       font_tiny: pygame.font.Font):
        """绘制运输位"""
        has_wafer = len(wafers) > 0
        is_overtime = has_wafer and wafers[0]["stay_time"] >= 10
        is_warning = has_wafer and wafers[0]["stay_time"] >= 7
        
        # 背景色
        if is_overtime:
            bg_color = self.theme.dim_color(self.theme.danger, 0.3)
            border_color = self.theme.danger
        elif is_warning:
            bg_color = self.theme.dim_color(self.theme.warning, 0.2)
            border_color = self.theme.warning
        else:
            bg_color = self.theme.bg_surface
            border_color = self.theme.border
        
        # 主体
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(screen, bg_color, rect, border_radius=6)
        pygame.draw.rect(screen, border_color, rect, 2, border_radius=6)
        
        # 流动指示线
        if has_wafer:
            self._draw_flow_indicator(screen, x, y, width, height)
        
        # 名称
        short_name = name.replace("d_", "")
        name_surf = font_tiny.render(short_name, True, self.theme.text_secondary)
        name_rect = name_surf.get_rect(centerx=x + width // 2, bottom=y - 4)
        screen.blit(name_surf, name_rect)
    
    def _draw_flow_indicator(self, screen: pygame.Surface, x: int, y: int, w: int, h: int):
        """绘制流动指示"""
        offset = self.anim.flow_offset(30)
        color = self.theme.accent
        
        # 绘制移动的虚线
        dash_len = 6
        gap_len = 4
        total = dash_len + gap_len
        
        for dx in range(int(-offset) % total, w, total):
            start_x = x + dx
            end_x = min(x + dx + dash_len, x + w)
            if start_x < x + w:
                pygame.draw.line(screen, color, 
                               (start_x, y + h // 2), 
                               (end_x, y + h // 2), 2)


# ============ 按钮类 (增强版) ============
class Button:
    """按钮类 - 支持分组、快捷键、悬停动画"""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 text: str, action_id: int, shortcut: str = "",
                 theme: ColorTheme = None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action_id = action_id
        self.shortcut = shortcut
        self.enabled = False
        self.hovered = False
        self.theme = theme or THEME
        self.hover_progress = 0.0  # 悬停动画进度
        self.last_update = pytime.time()
    
    def update(self):
        """更新动画状态"""
        now = pytime.time()
        dt = now - self.last_update
        self.last_update = now
        
        # 悬停动画
        target = 1.0 if (self.hovered and self.enabled) else 0.0
        speed = 8.0  # 动画速度
        if self.hover_progress < target:
            self.hover_progress = min(target, self.hover_progress + dt * speed)
        else:
            self.hover_progress = max(target, self.hover_progress - dt * speed)
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font, 
             font_small: pygame.font.Font = None):
        """绘制按钮"""
        self.update()
        
        # 根据状态选择颜色
        if not self.enabled:
            bg_color = self.theme.bg_elevated
            text_color = self.theme.text_muted
            border_color = self.theme.border_muted
        else:
            # 插值悬停效果
            base_bg = self.theme.bg_surface
            hover_bg = self.theme.dim_color(self.theme.accent, 0.3)
            bg_color = self.theme.lerp_color(base_bg, hover_bg, self.hover_progress)
            
            text_color = self.theme.lerp_color(
                self.theme.text_primary, 
                self.theme.accent,
                self.hover_progress
            )
            border_color = self.theme.lerp_color(
                self.theme.border,
                self.theme.accent,
                self.hover_progress
            )
        
        # 绘制按钮
        pygame.draw.rect(screen, bg_color, self.rect, border_radius=6)
        pygame.draw.rect(screen, border_color, self.rect, 2, border_radius=6)
        
        # 绘制文字
        if self.shortcut and font_small:
            # 左侧主文字
            text_surf = font.render(self.text, True, text_color)
            text_rect = text_surf.get_rect(midleft=(self.rect.left + 10, self.rect.centery))
            screen.blit(text_surf, text_rect)
            
            # 右侧快捷键
            key_text = f"[{self.shortcut}]"
            key_color = self.theme.text_muted if not self.enabled else self.theme.text_secondary
            key_surf = font_small.render(key_text, True, key_color)
            key_rect = key_surf.get_rect(midright=(self.rect.right - 8, self.rect.centery))
            screen.blit(key_surf, key_rect)
        else:
            text_surf = font.render(self.text, True, text_color)
            text_rect = text_surf.get_rect(center=self.rect.center)
            screen.blit(text_surf, text_rect)
    
    def check_hover(self, pos: Tuple[int, int]) -> bool:
        self.hovered = self.rect.collidepoint(pos)
        return self.hovered
    
    def check_click(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos) and self.enabled


# ============ 主可视化器 ============
class PetriVisualizer:
    """Petri 网可视化器 - UI/UX 优化版 - 多腔室网格布局"""
    
    # 布局配置
    WINDOW_WIDTH = 1300
    WINDOW_HEIGHT = 950
    
    LEFT_PANEL_WIDTH = 260      # 左侧面板
    RIGHT_PANEL_WIDTH = 230     # 右侧面板
    CENTER_PADDING = 15         # 中间区域左右边距
    
    # 腔室配置（网格布局）
    CHAMBER_SIZE = 100           # 腔室大小（正方形）
    CENTER_ROBOT_SIZE = 130     # 中心机械手区域大小
    CHAMBER_GAP_X = 115         # 腔室水平间距
    CHAMBER_GAP_Y = 105         # 腔室垂直间距
    
    # 晶圆配置
    WAFER_RADIUS = 24
    MAX_VISIBLE_WAFERS = 1      # 每个腔室最多显示1个晶圆
    
    def __init__(self, env: Env_PN, model_path: Optional[str] = None):
        self.env = env
        self.net = env.net
        self.theme = THEME
        
        pygame.init()
        pygame.display.set_caption("晶圆加工控制台")
        
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # 动画管理器
        self.anim = AnimationManager()
        
        # 渲染器
        self.wafer_renderer = WaferRenderer(self.theme, self.anim)
        self.chamber_renderer = ChamberRenderer(self.theme, self.anim)
        
        # 字体 (增大以提高清晰度)
        self.font_large = pygame.font.SysFont("Microsoft YaHei", 22)
        self.font_medium = pygame.font.SysFont("Microsoft YaHei", 18)
        self.font_small = pygame.font.SysFont("Microsoft YaHei", 15)
        self.font_tiny = pygame.font.SysFont("Microsoft YaHei", 13)
        self.font_mono = pygame.font.SysFont("Consolas", 15)
        
        # 计算腔室位置
        self._setup_layout()
        
        # 创建动作按钮
        self._setup_buttons()
        
        # 状态变量
        self.step_count = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.td = None
        self.running = True
        self.done = False
        
        # 动作历史记录
        self.action_history: List[Dict] = []
        
        # 模型相关
        self.policy = None
        if model_path is not None:
            self.load_model(model_path)
        
        # 快捷键映射
        self._setup_shortcuts()
    
    def _setup_shortcuts(self):
        """设置快捷键映射"""
        self.shortcuts = {
            pygame.K_1: 0,
            pygame.K_2: 1,
            pygame.K_3: 2,
            pygame.K_4: 3,
            pygame.K_5: 4,
            pygame.K_6: 5,
            pygame.K_7: 6,
            pygame.K_8: 7,
            pygame.K_9: 8,
            pygame.K_w: self.net.T,  # WAIT
            pygame.K_r: -1,  # Random
            pygame.K_m: -3,  # Model
            pygame.K_SPACE: -2,  # Reset
        }
    
    def load_model(self, model_path: str):
        """加载训练好的 PPO 模型"""
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在 {model_path}")
            return False
        
        n_actions = self.env.n_actions
        n_obs = self.env.observation_spec["observation"].shape[0]
        
        policy_backbone = MaskedPolicyHead(hidden=256, n_obs=n_obs, n_actions=n_actions, n_layers=4)
        td_module = TensorDictModule(
            policy_backbone, in_keys=["observation_f"], out_keys=["logits"]
        )
        self.policy = ProbabilisticActor(
            module=td_module,
            in_keys={"logits": "logits", "mask": "action_mask"},
            out_keys=["action"],
            distribution_class=MaskedCategorical,
            return_log_prob=True,
        )
        
        state_dict = torch.load(model_path, map_location="cpu")
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        
        print(f"模型加载成功: {model_path}")
        return True
    
    def get_model_action(self) -> Optional[int]:
        """使用模型获取动作"""
        if self.policy is None:
            return None
        
        obs = self.env._build_obs()
        action_mask = self._get_current_action_mask()
        
        td = TensorDict({
            "observation": torch.as_tensor(obs, dtype=torch.int64).unsqueeze(0),
            "observation_f": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
            "action_mask": torch.as_tensor(action_mask, dtype=torch.bool).unsqueeze(0),
        }, batch_size=[1])
        
        with torch.no_grad():
            with set_exploration_type(ExplorationType.MODE):
                td = self.policy(td)
        
        return td["action"].item()
    
    def _get_current_action_mask(self) -> np.ndarray:
        """获取当前的动作掩码"""
        action_mask_indices = self.net.get_enable_t()
        mask = np.zeros(self.env.n_actions, dtype=bool)
        mask[action_mask_indices] = True
        mask[self.net.T] = True
        return mask
    
    def _setup_layout(self):
        """设置布局位置 - 双机械手网格布局"""
        
        # ========== 腔室映射配置 ==========
        # 定义显示名称到实际 Petri 库所的映射
        self.chamber_config = {
            # TM2 区域（活跃）
            "LLA": {"source": "LP", "active": True, "proc_time": 0, "robot": "TM2"},
            "LLB": {"source": "LP_done", "active": True, "proc_time": 0, "robot": "TM2"},
            "PM7": {"source": "PM1", "machine": 0, "active": True, "proc_time": 80, "robot": "TM2"},
            "PM8": {"source": "PM1", "machine": 1, "active": True, "proc_time": 80, "robot": "TM2"},
            "PM9": {"source": "PM2", "machine": 0, "active": True, "proc_time": 30, "robot": "TM2"},
            "PM10": {"source": None, "active": False, "proc_time": 0, "robot": "TM2"},
            # TM3 区域（闲置）
            "LLC": {"source": None, "active": False, "proc_time": 0, "robot": "TM3"},
            "LLD": {"source": None, "active": False, "proc_time": 0, "robot": "TM3"},
            "PM1": {"source": None, "active": False, "proc_time": 0, "robot": "TM3"},
            "PM2": {"source": None, "active": False, "proc_time": 0, "robot": "TM3"},
            "PM3": {"source": None, "active": False, "proc_time": 0, "robot": "TM3"},
            "PM4": {"source": None, "active": False, "proc_time": 0, "robot": "TM3"},
            "PM5": {"source": None, "active": False, "proc_time": 0, "robot": "TM3"},
            "PM6": {"source": None, "active": False, "proc_time": 0, "robot": "TM3"},
        }
        
        # 显示腔室列表
        self.display_chambers = list(self.chamber_config.keys())
        
        # 运输库所（显示在 TM2 机械手区域）
        self.transports = ["d_PM1", "d_PM2", "d_LP_done"]
        
        # ========== 计算网格基准坐标 ==========
        center_start = self.LEFT_PANEL_WIDTH + self.CENTER_PADDING
        center_end = self.WINDOW_WIDTH - self.RIGHT_PANEL_WIDTH - self.CENTER_PADDING
        grid_center_x = (center_start + center_end) // 2
        grid_center_y = self.WINDOW_HEIGHT // 2
        
        # 网格单元格大小
        cell_w = self.CHAMBER_GAP_X  # 列间距
        cell_h = self.CHAMBER_GAP_Y  # 行间距
        cs = self.CHAMBER_SIZE       # 腔室尺寸
        
        # ========== 根据相对坐标计算位置 ==========
        # 坐标格式: (行, 列)，从1开始
        # 行1-3是TM2区域，行4-7是TM3区域
        # 列1-4
        
        # 网格原点（左上角，对应行1列1）
        # 让整个布局居中
        total_cols = 4
        total_rows = 7
        origin_x = grid_center_x - (total_cols * cell_w) // 2
        origin_y = grid_center_y - (total_rows * cell_h) // 2
        
        def grid_pos(row: int, col: int) -> Tuple[int, int]:
            """将网格坐标转换为像素坐标（返回腔室左上角）"""
            x = origin_x + (col - 1) * cell_w + (cell_w - cs) // 2
            y = origin_y + (row - 1) * cell_h + (cell_h - cs) // 2
            return (int(x), int(y))
        
        # 根据用户提供的相对坐标设置位置（已翻转，LLA/LLB在底部）
        self.chamber_positions = {
            # TM3 区域（行1-4，在上方）
            "PM3": grid_pos(1, 2),
            "PM4": grid_pos(1, 3),
            "PM2": grid_pos(2, 1),
            "PM1": grid_pos(3, 1),
            "PM5": grid_pos(2, 4),
            "PM6": grid_pos(3, 4),
            "LLC": grid_pos(4, 2),
            "LLD": grid_pos(4, 3),
            # TM2 区域（行5-7，在下方）
            "PM8": grid_pos(5, 1),
            "PM7": grid_pos(6, 1),
            "PM9": grid_pos(5, 4),
            "PM10": grid_pos(6, 4),
            "LLA": grid_pos(7, 2),
            "LLB": grid_pos(7, 3),
        }
        
        # ========== 计算两个机械手的中心位置 ==========
        # TM3 在行2-3、列2-3的中心（上方）
        tm3_x = origin_x + int(1.5 * cell_w) + cell_w // 2
        tm3_y = origin_y + int(1.5 * cell_h) + cell_h // 2
        self.tm3_pos = (tm3_x, tm3_y)
        
        # TM2 在行5-6、列2-3的中心（下方）
        tm2_x = origin_x + int(1.5 * cell_w) + cell_w // 2
        tm2_y = origin_y + int(4.5 * cell_h) + cell_h // 2
        self.tm2_pos = (tm2_x, tm2_y)
        self.center_pos = self.tm2_pos  # 保持向后兼容
    
    def _setup_buttons(self):
        """设置动作按钮 - 分组布局"""
        self.buttons = []
        self.button_groups = {}
        
        t_names = self.net.id2t_name
        
        panel_x = self.WINDOW_WIDTH - self.RIGHT_PANEL_WIDTH + 10
        button_width = self.RIGHT_PANEL_WIDTH - 20
        button_height = 32
        button_gap = 4
        
        y = 72
        
        # 变迁动作组
        self.button_groups["transitions"] = {"start_y": y, "buttons": []}
        for i, t_name in enumerate(t_names):
            shortcut = str(i + 1) if i < 9 else ""
            btn = Button(panel_x, y, button_width, button_height, 
                        t_name, i, shortcut, self.theme)
            self.buttons.append(btn)
            self.button_groups["transitions"]["buttons"].append(btn)
            y += button_height + button_gap
        
        y += 15
        
        # 系统控制组
        self.button_groups["control"] = {"start_y": y, "buttons": []}
        
        # WAIT 按钮
        self.wait_button = Button(panel_x, y, button_width, button_height,
                                  "WAIT", self.net.T, "W", self.theme)
        self.buttons.append(self.wait_button)
        self.button_groups["control"]["buttons"].append(self.wait_button)
        y += button_height + button_gap
        
        # Random 按钮
        self.random_button = Button(panel_x, y, button_width, button_height,
                                    "Random", -1, "R", self.theme)
        self.buttons.append(self.random_button)
        self.button_groups["control"]["buttons"].append(self.random_button)
        y += button_height + button_gap
        
        # Model 按钮
        self.model_button = Button(panel_x, y, button_width, button_height,
                                   "Model", -3, "M", self.theme)
        self.buttons.append(self.model_button)
        self.button_groups["control"]["buttons"].append(self.model_button)
        
        y += button_height + 20
        
        # Reset 按钮
        self.reset_button = Button(panel_x, y, button_width, button_height,
                                   "Reset", -2, "Space", self.theme)
        self.reset_button.enabled = True
    
    def _update_buttons(self, action_mask: np.ndarray):
        """更新按钮状态"""
        for btn in self.buttons:
            if btn.action_id >= 0 and btn.action_id < len(action_mask):
                btn.enabled = bool(action_mask[btn.action_id])
            elif btn.action_id == -1:  # Random
                btn.enabled = True
            elif btn.action_id == -3:  # Model
                btn.enabled = self.policy is not None
    
    def _collect_wafer_info(self) -> Dict[str, List[Dict]]:
        """收集各库所中晶圆信息 - 映射到显示名称"""
        wafer_info = {}
        
        # 初始化所有显示腔室
        for chamber_name in self.display_chambers:
            wafer_info[chamber_name] = []
        
        # 初始化运输库所
        for t_name in self.transports:
            wafer_info[t_name] = []
        
        for p_idx, place in enumerate(self.net.marks):
            p_name = place.name
            if p_name.startswith("r_"):
                continue
            
            for tok in place.tokens:
                if tok.token_id < 0:
                    continue
                
                wafer_data = {
                    "token_id": tok.token_id,
                    "stay_time": int(tok.stay_time),
                    "enter_time": int(tok.enter_time),
                    "proc_time": place.processing_time,
                    "type": place.type,
                    "machine": getattr(tok, 'machine', -1),
                }
                
                # 根据实际库所映射到显示名称
                if p_name == "PM1":
                    # PM1 根据 machine 属性分配到 PM7 或 PM8
                    machine = getattr(tok, 'machine', 0)
                    display_name = "PM7" if machine == 0 else "PM8"
                    wafer_info[display_name].append(wafer_data)
                elif p_name == "PM2":
                    # PM2 映射到 PM9
                    wafer_info["PM9"].append(wafer_data)
                elif p_name == "LP":
                    # LP 映射到 LLA
                    wafer_info["LLA"].append(wafer_data)
                elif p_name == "LP_done":
                    # LP_done 映射到 LLB
                    wafer_info["LLB"].append(wafer_data)
                elif p_name.startswith("d_"):
                    # 运输库所保持原名，并记录目标腔室
                    # 根据运输库所名称确定目标腔室
                    if p_name == "d_PM1":
                        # d_PM1 -> PM1 -> PM7 (machine=0) 或 PM8 (machine=1)
                        machine = getattr(tok, 'machine', 0)
                        wafer_data["target_chamber"] = "PM7" if machine == 0 else "PM8"
                    elif p_name == "d_PM2":
                        # d_PM2 -> PM2 -> PM9
                        wafer_data["target_chamber"] = "PM9"
                    elif p_name == "d_LP_done":
                        # d_LP_done -> LP_done -> LLB
                        wafer_data["target_chamber"] = "LLB"
                    else:
                        wafer_data["target_chamber"] = None
                    wafer_info[p_name].append(wafer_data)
        
        return wafer_info
    
    def _draw_left_panel(self):
        """绘制左侧面板 - 状态卡片 + 时间轴历史"""
        panel_x = 8
        panel_y = 8
        panel_width = self.LEFT_PANEL_WIDTH - 16
        panel_height = self.WINDOW_HEIGHT - 16
        
        # 面板背景
        rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, self.theme.bg_deep, rect, border_radius=12)
        pygame.draw.rect(self.screen, self.theme.border, rect, 2, border_radius=12)
        
        y = panel_y + 18
        
        # ===== 系统状态区域 =====
        section_title = self.font_medium.render("SYSTEM STATUS", True, self.theme.accent)
        self.screen.blit(section_title, (panel_x + 12, y))
        y += 32
        
        # 状态卡片
        card_height = 32
        card_gap = 6
        
        stats = [
            ("TIME", f"{self.net.time}s", self.theme.accent),
            ("STEP", str(self.step_count), self.theme.text_primary),
            ("REWARD", f"{self.total_reward:+.1f}", 
             self.theme.success if self.total_reward >= 0 else self.theme.danger),
            ("LAST", f"{self.last_reward:+.1f}",
             self.theme.success if self.last_reward >= 0 else self.theme.danger),
        ]
        
        for label, value, color in stats:
            # 卡片背景
            card_rect = pygame.Rect(panel_x + 8, y, panel_width - 16, card_height)
            pygame.draw.rect(self.screen, self.theme.bg_surface, card_rect, border_radius=6)
            
            # 标签
            label_surf = self.font_small.render(label, True, self.theme.text_secondary)
            self.screen.blit(label_surf, (panel_x + 14, y + 7))
            
            # 值
            value_surf = self.font_medium.render(value, True, color)
            value_rect = value_surf.get_rect(midright=(panel_x + panel_width - 14, y + card_height // 2))
            self.screen.blit(value_surf, value_rect)
            
            y += card_height + card_gap
        
        # 进度条（假设总共需要处理的晶圆数）
        y += 8
        # 查找 LP 和 LP_done 库所
        lp_place = next((p for p in self.net.marks if p.name == "LP"), None)
        lp_done_place = next((p for p in self.net.marks if p.name == "LP_done"), None)
        total_wafers = len(lp_place.tokens) if lp_place else 3
        done_wafers = len(lp_done_place.tokens) if lp_done_place else 0
        
        progress_label = self.font_small.render(f"PROGRESS: {done_wafers}/{total_wafers + done_wafers}", True, self.theme.text_secondary)
        self.screen.blit(progress_label, (panel_x + 12, y))
        y += 22
        
        # 进度条
        bar_rect = pygame.Rect(panel_x + 12, y, panel_width - 24, 10)
        pygame.draw.rect(self.screen, self.theme.bg_elevated, bar_rect, border_radius=5)
        
        progress = done_wafers / max(1, total_wafers + done_wafers)
        if progress > 0:
            fill_width = int((panel_width - 24) * progress)
            fill_rect = pygame.Rect(panel_x + 12, y, fill_width, 10)
            pygame.draw.rect(self.screen, self.theme.success, fill_rect, border_radius=5)
        
        y += 22
        
        # 分隔线
        pygame.draw.line(self.screen, self.theme.border,
                        (panel_x + 12, y), (panel_x + panel_width - 12, y), 1)
        y += 12
        
        # ===== 图例 =====
        legend_title = self.font_small.render("LEGEND", True, self.theme.text_muted)
        self.screen.blit(legend_title, (panel_x + 12, y))
        y += 20
        
        legends = [
            (self.theme.success, "Normal"),
            (self.theme.warning, "Warning"),
            (self.theme.danger, "Critical"),
        ]
        
        for color, text in legends:
            pygame.draw.circle(self.screen, color, (panel_x + 20, y + 7), 6)
            text_surf = self.font_small.render(text, True, self.theme.text_primary)
            self.screen.blit(text_surf, (panel_x + 34, y))
            y += 20
        
        y += 8
        pygame.draw.line(self.screen, self.theme.border,
                        (panel_x + 12, y), (panel_x + panel_width - 12, y), 1)
        y += 12
        
        # ===== 动作历史 (时间轴样式) =====
        history_title = self.font_medium.render("HISTORY", True, self.theme.accent)
        self.screen.blit(history_title, (panel_x + 12, y))
        y += 28
        
        # 时间轴
        timeline_x = panel_x + 20
        max_y = panel_y + panel_height - 15
        
        for i, entry in enumerate(reversed(self.action_history[-12:])):  # 最近12条
            if y >= max_y - 25:
                break
            
            step = entry['step']
            action_name = entry['action']
            total = entry['total']
            
            # 时间轴节点
            node_color = self.theme.success if total > 0 else (
                self.theme.danger if total < -5 else self.theme.warning
            )
            pygame.draw.circle(self.screen, node_color, (timeline_x, y + 9), 6)
            
            # 时间轴线
            if i < len(self.action_history) - 1 and y + 28 < max_y:
                pygame.draw.line(self.screen, self.theme.border,
                               (timeline_x, y + 15), (timeline_x, y + 30), 2)
            
            # 步数和动作
            step_text = f"{step:02d}"
            step_surf = self.font_mono.render(step_text, True, self.theme.text_secondary)
            self.screen.blit(step_surf, (timeline_x + 14, y + 1))
            
            action_surf = self.font_small.render(action_name, True, self.theme.text_primary)
            self.screen.blit(action_surf, (timeline_x + 42, y + 1))
            
            # 奖励
            reward_text = f"{total:+.1f}"
            reward_color = node_color
            reward_surf = self.font_small.render(reward_text, True, reward_color)
            reward_rect = reward_surf.get_rect(right=panel_x + panel_width - 12, top=y + 1)
            self.screen.blit(reward_surf, reward_rect)
            
            y += 28
    
    def _draw_right_panel(self):
        """绘制右侧面板 - 按钮分组"""
        panel_x = self.WINDOW_WIDTH - self.RIGHT_PANEL_WIDTH
        panel_y = 8
        panel_width = self.RIGHT_PANEL_WIDTH - 8
        panel_height = self.WINDOW_HEIGHT - 16
        
        # 面板背景
        rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, self.theme.bg_deep, rect, border_radius=12)
        pygame.draw.rect(self.screen, self.theme.border, rect, 2, border_radius=12)
        
        # 标题
        title_surf = self.font_medium.render("ACTIONS", True, self.theme.accent)
        self.screen.blit(title_surf, (panel_x + 12, panel_y + 15))
        
        # 变迁组标题
        trans_y = self.button_groups["transitions"]["start_y"] - 22
        trans_label = self.font_small.render("TRANSITIONS", True, self.theme.text_secondary)
        self.screen.blit(trans_label, (panel_x + 12, trans_y))
        
        # 控制组标题
        ctrl_y = self.button_groups["control"]["start_y"] - 22
        ctrl_label = self.font_small.render("CONTROL", True, self.theme.text_secondary)
        self.screen.blit(ctrl_label, (panel_x + 12, ctrl_y))
        
        # 绘制所有按钮
        for btn in self.buttons:
            btn.draw(self.screen, self.font_small, self.font_tiny)
        
        # Reset 按钮
        self.reset_button.draw(self.screen, self.font_small, self.font_tiny)
        
        # 快捷键提示
        hint_y = panel_y + panel_height - 25
        hint_surf = self.font_tiny.render("Keys: [1-6] [W] [R] [M]", True, self.theme.text_muted)
        hint_rect = hint_surf.get_rect(centerx=panel_x + panel_width // 2, centery=hint_y)
        self.screen.blit(hint_surf, hint_rect)
    
    def _draw_flow_arrow(self, x1: int, y1: int, x2: int, y2: int):
        """绘制流动箭头"""
        color = self.theme.text_muted
        
        # 主线
        mid_x = (x1 + x2) // 2
        pygame.draw.line(self.screen, color, (x1, y1), (mid_x - 10, y1), 2)
        pygame.draw.line(self.screen, color, (mid_x + 10, y1), (x2, y1), 2)
        
        # 箭头
        arrow_points = [
            (mid_x + 8, y1),
            (mid_x - 4, y1 - 6),
            (mid_x - 4, y1 + 6)
        ]
        pygame.draw.polygon(self.screen, self.theme.accent, arrow_points)
    
    def _draw_robot_buffer(self, wafer_info: Dict[str, List[Dict]]):
        """绘制双机械手缓冲区（TM2 和 TM3）"""
        # 收集运输中的晶圆（TM2）
        transport_wafers = []
        for t_name in self.transports:
            transport_wafers.extend(wafer_info.get(t_name, []))
        
        # 收集目标腔室（用于高亮连接线）
        target_chambers = set()
        for wafer in transport_wafers:
            target = wafer.get("target_chamber")
            if target:
                target_chambers.add(target)
        
        # ========== 先绘制连接线（在底层） ==========
        self._draw_robot_connections(target_chambers)
        
        # ========== 绘制 TM2（活跃） ==========
        self._draw_single_robot(
            self.tm2_pos[0], self.tm2_pos[1], 
            "TM2", is_busy=len(transport_wafers) > 0, 
            is_active=True, wafers=transport_wafers
        )
        
        # ========== 绘制 TM3（闲置） ==========
        self._draw_single_robot(
            self.tm3_pos[0], self.tm3_pos[1],
            "TM3", is_busy=False, is_active=False, wafers=[]
        )
    
    def _draw_single_robot(self, cx: int, cy: int, name: str, 
                           is_busy: bool, is_active: bool, wafers: List[Dict]):
        """绘制单个机械手"""
        size = self.CENTER_ROBOT_SIZE
        
        # 背景
        rect = pygame.Rect(cx - size // 2, cy - size // 2, size, size)
        bg_color = self.theme.bg_deep if is_active else self.theme.bg_surface
        pygame.draw.rect(self.screen, bg_color, rect, border_radius=10)
        
        # 边框
        if is_active:
            border_color = self.theme.accent if is_busy else self.theme.border
        else:
            border_color = self.theme.border_muted
        pygame.draw.rect(self.screen, border_color, rect, 2, border_radius=10)
        
        # 网格
        grid_color = self.theme.border_muted if is_active else self.theme.dim_color(self.theme.border_muted, 0.5)
        grid_spacing = 16
        for gx in range(cx - size // 2 + grid_spacing, cx + size // 2, grid_spacing):
            pygame.draw.line(self.screen, grid_color, 
                           (gx, cy - size // 2 + 8), (gx, cy + size // 2 - 8), 1)
        for gy in range(cy - size // 2 + grid_spacing, cy + size // 2, grid_spacing):
            pygame.draw.line(self.screen, grid_color, 
                           (cx - size // 2 + 8, gy), (cx + size // 2 - 8, gy), 1)
        
        # 标题
        title_color = self.theme.accent if is_active else self.theme.text_muted
        title_surf = self.font_large.render(name, True, title_color)
        title_rect = title_surf.get_rect(centerx=cx, top=cy - size // 2 + 5)
        self.screen.blit(title_surf, title_rect)
        
        # 状态指示灯
        led_radius = 6
        led_x = cx + size // 2 - 15
        led_y = cy - size // 2 + 15
        
        if is_active and is_busy:
            led_color = self.theme.success
            pulse = self.anim.pulse(2.0, 0.6, 1.0)
            glow_radius = int(led_radius + 3 * pulse)
            glow_surface = pygame.Surface((glow_radius * 2 + 10, glow_radius * 2 + 10), pygame.SRCALPHA)
            glow_color = (*led_color, int(80 * pulse))
            pygame.draw.circle(glow_surface, glow_color, (glow_radius + 5, glow_radius + 5), glow_radius)
            self.screen.blit(glow_surface, (led_x - glow_radius - 5, led_y - glow_radius - 5))
        else:
            led_color = self.theme.text_muted
        
        pygame.draw.circle(self.screen, led_color, (led_x, led_y), led_radius)
        pygame.draw.circle(self.screen, self.theme.border, (led_x, led_y), led_radius, 1)
        
        # 状态文字
        if is_active:
            status_text = "BUSY" if is_busy else "IDLE"
            status_color = self.theme.success if is_busy else self.theme.text_muted
        else:
            status_text = "IDLE"
            status_color = self.theme.text_muted
        status_surf = self.font_small.render(status_text, True, status_color)
        status_rect = status_surf.get_rect(centerx=cx, centery=cy - 8)
        self.screen.blit(status_surf, status_rect)
        
        # 绘制晶圆
        if wafers and is_active:
            wafer_y = cy + 15
            wafer_spacing = self.WAFER_RADIUS * 2 + 4
            total_width = len(wafers[:2]) * wafer_spacing - 4
            start_x = cx - total_width // 2 + self.WAFER_RADIUS
            
            for i, wafer in enumerate(wafers[:2]):
                wafer_x = start_x + i * wafer_spacing
                self.wafer_renderer.draw(
                    self.screen, wafer_x, wafer_y,
                    wafer["stay_time"], 0, 2,
                    wafer["token_id"], None, self.font_tiny
                )
    
    def _draw_robot_connections(self, target_chambers: set = None):
        """绘制机械手到腔室的折线连接
        
        Args:
            target_chambers: 当前运输中晶圆的目标腔室集合，这些连接线会有脉冲发光效果
        """
        robot_size = self.CENTER_ROBOT_SIZE
        if target_chambers is None:
            target_chambers = set()
        
        for chamber_name, pos in self.chamber_positions.items():
            config = self.chamber_config[chamber_name]
            is_active = config["active"]
            robot_name = config.get("robot", "TM2")
            is_target = chamber_name in target_chambers
            
            # 根据腔室所属的机械手选择中心点
            if robot_name == "TM2":
                cx, cy = self.tm2_pos
            else:  # TM3
                cx, cy = self.tm3_pos
            
            # 腔室中心
            chamber_cx = pos[0] + self.CHAMBER_SIZE // 2
            chamber_cy = pos[1] + self.CHAMBER_SIZE // 2
            
            # 计算折线路径
            points = self._calc_polyline_path(cx, cy, robot_size, chamber_cx, chamber_cy, chamber_name)
            
            # 选择颜色和样式
            if is_target:
                # 目标腔室：脉冲发光效果（从虚线变实线 + 脉冲亮度）
                pulse = self.anim.pulse(frequency=1.5, min_val=0.7, max_val=1.0)
                line_color = self.theme.brighten_color(self.theme.accent, pulse)
                line_width = 3
                self._draw_polyline(points, line_color, line_width)
            elif is_active:
                # 普通活跃腔室：实线
                line_color = self.theme.dim_color(self.theme.accent, 0.6)
                line_width = 2
                self._draw_polyline(points, line_color, line_width)
            else:
                # 闲置腔室：虚线
                line_color = self.theme.dim_color(self.theme.border_muted, 0.5)
                line_width = 1
                self._draw_dashed_polyline(points, line_color, line_width, 5, 4)
    
    def _calc_polyline_path(self, cx: int, cy: int, robot_size: int,
                            target_x: int, target_y: int, chamber_name: str) -> List[Tuple[int, int]]:
        """计算从机械手中心到腔室的折线路径（翻转后布局）"""
        half_robot = robot_size // 2 + 5
        half_chamber = self.CHAMBER_SIZE // 2 + 3
        
        # ===== TM2 区域腔室（下方）=====
        if chamber_name in ["LLA", "LLB"]:
            # 下方腔室：垂直向下
            start_pt = (cx, cy + half_robot)
            mid_pt = (cx, target_y - half_chamber - 10)
            mid_pt2 = (target_x, target_y - half_chamber - 10)
            end_pt = (target_x, target_y - half_chamber)
            return [start_pt, mid_pt, mid_pt2, end_pt]
        
        elif chamber_name in ["PM7", "PM8"]:
            # 左侧腔室：水平向左
            start_pt = (cx - half_robot, cy)
            mid_pt = (target_x + half_chamber + 10, cy)
            mid_pt2 = (target_x + half_chamber + 10, target_y)
            end_pt = (target_x + half_chamber, target_y)
            return [start_pt, mid_pt, mid_pt2, end_pt]
        
        elif chamber_name in ["PM9", "PM10"]:
            # 右侧腔室：水平向右
            start_pt = (cx + half_robot, cy)
            mid_pt = (target_x - half_chamber - 10, cy)
            mid_pt2 = (target_x - half_chamber - 10, target_y)
            end_pt = (target_x - half_chamber, target_y)
            return [start_pt, mid_pt, mid_pt2, end_pt]
        
        # ===== TM3 区域腔室（上方）=====
        elif chamber_name in ["PM3", "PM4"]:
            # 上方腔室：垂直向上
            start_pt = (cx, cy - half_robot)
            mid_pt = (cx, target_y + half_chamber + 10)
            mid_pt2 = (target_x, target_y + half_chamber + 10)
            end_pt = (target_x, target_y + half_chamber)
            return [start_pt, mid_pt, mid_pt2, end_pt]
        
        elif chamber_name in ["PM1", "PM2"]:
            # 左侧腔室：水平向左
            start_pt = (cx - half_robot, cy)
            mid_pt = (target_x + half_chamber + 10, cy)
            mid_pt2 = (target_x + half_chamber + 10, target_y)
            end_pt = (target_x + half_chamber, target_y)
            return [start_pt, mid_pt, mid_pt2, end_pt]
        
        elif chamber_name in ["PM5", "PM6"]:
            # 右侧腔室：水平向右
            start_pt = (cx + half_robot, cy)
            mid_pt = (target_x - half_chamber - 10, cy)
            mid_pt2 = (target_x - half_chamber - 10, target_y)
            end_pt = (target_x - half_chamber, target_y)
            return [start_pt, mid_pt, mid_pt2, end_pt]
        
        elif chamber_name in ["LLC", "LLD"]:
            # 下方腔室（相对于TM3）：垂直向下
            start_pt = (cx, cy + half_robot)
            mid_pt = (cx, target_y - half_chamber - 10)
            mid_pt2 = (target_x, target_y - half_chamber - 10)
            end_pt = (target_x, target_y - half_chamber)
            return [start_pt, mid_pt, mid_pt2, end_pt]
        
        # 默认直线
        return [(cx, cy), (target_x, target_y)]
    
    def _draw_polyline(self, points: List[Tuple[int, int]], color: Tuple[int, int, int], width: int):
        """绘制折线"""
        if len(points) < 2:
            return
        for i in range(len(points) - 1):
            pygame.draw.line(self.screen, color, points[i], points[i + 1], width)
    
    def _draw_dashed_polyline(self, points: List[Tuple[int, int]], color: Tuple[int, int, int], 
                               width: int, dash_len: int, gap_len: int):
        """绘制虚线折线"""
        if len(points) < 2:
            return
        for i in range(len(points) - 1):
            self._draw_dashed_line(
                points[i][0], points[i][1],
                points[i + 1][0], points[i + 1][1],
                color, width, dash_len, gap_len
            )
    
    def _draw_dashed_line(self, x1: int, y1: int, x2: int, y2: int, 
                          color: Tuple[int, int, int], width: int,
                          dash_len: int, gap_len: int):
        """绘制虚线"""
        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx * dx + dy * dy)
        if dist == 0:
            return
        
        nx, ny = dx / dist, dy / dist
        total = dash_len + gap_len
        pos = 0
        
        while pos < dist:
            start = pos
            end = min(pos + dash_len, dist)
            pygame.draw.line(self.screen, color,
                           (int(x1 + nx * start), int(y1 + ny * start)),
                           (int(x1 + nx * end), int(y1 + ny * end)), width)
            pos += total
    
    def _draw_idle_chamber(self, name: str, x: int, y: int):
        """绘制闲置腔室"""
        size = self.CHAMBER_SIZE
        
        # 阴影
        shadow_offset = 3
        shadow_rect = pygame.Rect(x + shadow_offset, y + shadow_offset, size, size)
        shadow_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (0, 0, 0, 40), (0, 0, size, size), border_radius=8)
        self.screen.blit(shadow_surface, (x + shadow_offset, y + shadow_offset))
        
        # 背景（更暗）
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(self.screen, self.theme.bg_surface, rect, border_radius=8)
        
        # 网格（更淡）
        grid_color = self.theme.dim_color(self.theme.border_muted, 0.5)
        for gx in range(x + 15, x + size, 15):
            pygame.draw.line(self.screen, grid_color, (gx, y + 10), (gx, y + size - 10), 1)
        for gy in range(y + 15, y + size, 15):
            pygame.draw.line(self.screen, grid_color, (x + 10, gy), (x + size - 10, gy), 1)
        
        # 边框
        pygame.draw.rect(self.screen, self.theme.border_muted, rect, 1, border_radius=8)
        
        # 名称
        name_surf = self.font_medium.render(name, True, self.theme.text_muted)
        name_rect = name_surf.get_rect(centerx=x + size // 2, centery=y + size // 2 - 10)
        self.screen.blit(name_surf, name_rect)
        
        # IDLE 标签
        idle_surf = self.font_tiny.render("IDLE", True, self.theme.text_muted)
        idle_rect = idle_surf.get_rect(centerx=x + size // 2, centery=y + size // 2 + 15)
        self.screen.blit(idle_surf, idle_rect)
    
    def _draw_active_chamber(self, name: str, x: int, y: int, proc_time: int, 
                             wafers: List[Dict], place_type: int):
        """绘制活跃腔室"""
        size = self.CHAMBER_SIZE
        
        # 获取状态
        status = self.chamber_renderer.get_chamber_status(wafers, proc_time)
        
        # LED 颜色
        if status == "danger":
            led_color = self.theme.danger
            if self.anim.blink(2.0):
                led_color = self.theme.brighten_color(led_color, 1.3)
        elif status == "warning":
            led_color = self.theme.warning
        elif status == "active":
            led_color = self.theme.success
        else:
            led_color = self.theme.text_muted
        
        # 阴影
        shadow_offset = 4
        shadow_rect = pygame.Rect(x + shadow_offset, y + shadow_offset, size, size)
        shadow_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (0, 0, 0, 80), (0, 0, size, size), border_radius=8)
        self.screen.blit(shadow_surface, (x + shadow_offset, y + shadow_offset))
        
        # 背景
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(self.screen, self.theme.bg_deep, rect, border_radius=8)
        
        # 网格
        grid_color = self.theme.border_muted
        for gx in range(x + 15, x + size, 15):
            pygame.draw.line(self.screen, grid_color, (gx, y + 18), (gx, y + size - 10), 1)
        for gy in range(y + 18, y + size, 15):
            pygame.draw.line(self.screen, grid_color, (x + 10, gy), (x + size - 10, gy), 1)
        
        # 边框
        border_color = self.theme.border if status == "idle" else led_color
        pygame.draw.rect(self.screen, border_color, rect, 2, border_radius=8)
        
        # 顶部 LED 条
        led_height = 5
        led_rect = pygame.Rect(x + 8, y + 5, size - 16, led_height)
        pygame.draw.rect(self.screen, self.theme.bg_elevated, led_rect, border_radius=2)
        
        # LED 活跃段
        led_segments = 4
        seg_width = (size - 20) // led_segments
        active_segs = led_segments if status in ["active", "danger"] else (2 if status == "warning" else 1)
        for i in range(active_segs):
            seg_x = x + 10 + i * (seg_width + 1)
            seg_rect = pygame.Rect(seg_x, y + 6, seg_width - 2, led_height - 2)
            pygame.draw.rect(self.screen, led_color, seg_rect, border_radius=1)
        
        # 名称（上方外部）
        name_surf = self.font_medium.render(name, True, self.theme.text_primary)
        name_rect = name_surf.get_rect(centerx=x + size // 2, bottom=y - 5)
        self.screen.blit(name_surf, name_rect)
        
        # 绘制晶圆
        if wafers:
            wafer = wafers[0]  # 只显示第一个
            wafer_x = x + size // 2
            wafer_y = y + size // 2 + 10
            
            self.wafer_renderer.draw(
                self.screen, wafer_x, wafer_y,
                wafer["stay_time"], wafer["proc_time"], place_type,
                wafer["token_id"], None, self.font_tiny
            )
        
        # 显示晶圆数量（如果超过1个）
        if len(wafers) > 1:
            count_text = f"+{len(wafers) - 1}"
            count_surf = self.font_tiny.render(count_text, True, self.theme.warning)
            count_rect = count_surf.get_rect(right=x + size - 5, bottom=y + size - 5)
            self.screen.blit(count_surf, count_rect)
    
    def draw(self):
        """绘制整个界面 - 正方形布局版本"""
        # 深色背景
        self.screen.fill(self.theme.bg_deepest)
        
        # 绘制左侧面板
        self._draw_left_panel()
        
        # 绘制右侧面板
        self._draw_right_panel()
        
        # 中间区域标题
        center_x = self.center_pos[0]
        
        # 主标题
        title_text = "WAFER PROCESSING CONSOLE"
        title_surf = self.font_large.render(title_text, True, self.theme.text_primary)
        title_rect = title_surf.get_rect(centerx=center_x, top=15)
        self.screen.blit(title_surf, title_rect)
        
        # 副标题 - 状态指示
        status_text = f"Time: {self.net.time}s  |  Step: {self.step_count}"
        if self.done:
            status_text += "  |  ENDED"
        status_color = self.theme.accent if not self.done else self.theme.warning
        status_surf = self.font_medium.render(status_text, True, status_color)
        status_rect = status_surf.get_rect(centerx=center_x, top=42)
        self.screen.blit(status_surf, status_rect)
        
        # 收集晶圆信息
        wafer_info = self._collect_wafer_info()
        
        # ========== 绘制中心机械手缓冲区 ==========
        self._draw_robot_buffer(wafer_info)
        
        # ========== 绘制各腔室 ==========
        for chamber_name in self.display_chambers:
            config = self.chamber_config[chamber_name]
            x, y = self.chamber_positions[chamber_name]
            wafers = wafer_info.get(chamber_name, [])
            
            if config["active"]:
                # 获取实际库所类型
                source = config["source"]
                if source and source in self.net.id2p_name:
                    p_idx = self.net.id2p_name.index(source)
                    place_type = self.net.marks[p_idx].type
                else:
                    place_type = 4  # 默认类型
                
                # 绘制活跃腔室
                self._draw_active_chamber(
                    chamber_name, x, y, 
                    config["proc_time"], wafers, place_type
                )
            else:
                # 绘制闲置腔室
                self._draw_idle_chamber(chamber_name, x, y)
        
        pygame.display.flip()
    
    def handle_events(self) -> Optional[int]:
        """处理事件，返回选中的动作或 None"""
        mouse_pos = pygame.mouse.get_pos()
        
        # 更新按钮悬停状态
        for btn in self.buttons:
            btn.check_hover(mouse_pos)
        self.reset_button.check_hover(mouse_pos)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return None
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return None
                
                # 检查快捷键
                if event.key in self.shortcuts:
                    action_id = self.shortcuts[event.key]
                    
                    # 检查动作是否可用
                    if action_id == -2:  # Reset 始终可用
                        return -2
                    elif action_id == -1:  # Random 始终可用
                        return -1
                    elif action_id == -3:  # Model
                        if self.policy is not None:
                            return -3
                    elif action_id >= 0:
                        # 检查变迁/WAIT 是否可用
                        if self.td is not None:
                            mask = self.td["action_mask"].numpy()
                            if action_id < len(mask) and mask[action_id]:
                                return action_id
                            elif action_id == self.net.T:  # WAIT
                                return action_id
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.reset_button.check_click(mouse_pos):
                        return -2
                    
                    for btn in self.buttons:
                        if btn.check_click(mouse_pos):
                            return btn.action_id
        
        return None
    
    def step_action(self, action: int) -> Tuple[Dict, bool, bool, bool]:
        """执行动作并返回结果"""
        if action == self.net.T:
            done, reward_dict, scrap = self.net.step(
                wait=True, with_reward=True, detailed_reward=True
            )
        else:
            done, reward_dict, scrap = self.net.step(
                t=action, wait=False, with_reward=True, detailed_reward=True
            )
        
        if not isinstance(reward_dict, dict):
            reward_dict = {'total': reward_dict}
        
        finish = done and not scrap
        return reward_dict, done, finish, scrap
    
    def reset(self):
        """重置环境"""
        self.td = self.env.reset()
        self.step_count = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.action_history = []
        self.done = False
        
        action_mask = self.td["action_mask"].numpy()
        self._update_buttons(action_mask)
    
    def run(self):
        """主循环"""
        self.reset()
        
        while self.running:
            action = self.handle_events()
            
            if action is not None:
                if action == -2:
                    self.reset()
                elif action == -1:
                    action_mask = self.td["action_mask"].numpy()
                    valid_actions = np.where(action_mask)[0]
                    if len(valid_actions) > 0:
                        action = int(np.random.choice(valid_actions))
                        self._execute_action(action)
                elif action == -3:
                    model_action = self.get_model_action()
                    if model_action is not None:
                        self._execute_action(model_action)
                else:
                    self._execute_action(action)
            
            self.draw()
            self.clock.tick(60)  # 60 FPS for smoother animations
        
        pygame.quit()
    
    def _execute_action(self, action: int):
        """执行动作并更新状态"""
        if self.done:
            return
        
        if action == self.net.T:
            action_name = "WAIT"
        else:
            action_name = self.net.id2t_name[action]
        
        reward_dict, done, finish, scrap = self.step_action(action)
        reward = reward_dict.get('total', 0)
        
        self.last_reward = reward
        self.total_reward += reward
        self.step_count += 1
        
        # 添加动作执行闪烁动画
        self.anim.add_flash(f"action_{action}", 0.3)
        
        self.action_history.append({
            'step': self.step_count,
            'action': action_name,
            'total': reward,
            'details': reward_dict
        })
        
        action_mask_indices = self.net.get_enable_t()
        mask = np.zeros(self.env.n_actions, dtype=bool)
        mask[action_mask_indices] = True
        mask[self.net.T] = True
        
        self.td = TensorDict({
            "observation": torch.zeros(1),
            "action_mask": torch.as_tensor(mask, dtype=torch.bool),
            "time": torch.tensor([self.net.time], dtype=torch.int64),
        })
        
        self._update_buttons(mask)
        
        if done:
            self.done = True
            self._show_end_message(finish, scrap)
    
    def _show_end_message(self, finish: bool, scrap: bool):
        """显示结束消息"""
        overlay = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        if finish:
            msg = "MISSION COMPLETE"
            sub_msg = "All wafers processed successfully!"
            color = self.theme.success
        elif scrap:
            msg = "MISSION FAILED"
            sub_msg = "Wafer scrapped due to timeout!"
            color = self.theme.danger
        else:
            msg = "EPISODE ENDED"
            sub_msg = ""
            color = self.theme.warning
        
        # 消息框
        box_width, box_height = 450, 220
        box_x = (self.WINDOW_WIDTH - box_width) // 2
        box_y = (self.WINDOW_HEIGHT - box_height) // 2
        
        # 框背景
        box_rect = pygame.Rect(box_x, box_y, box_width, box_height)
        pygame.draw.rect(self.screen, self.theme.bg_deep, box_rect, border_radius=12)
        pygame.draw.rect(self.screen, color, box_rect, 3, border_radius=12)
        
        # 主消息
        msg_surf = self.font_large.render(msg, True, color)
        msg_rect = msg_surf.get_rect(centerx=self.WINDOW_WIDTH // 2, centery=box_y + 40)
        self.screen.blit(msg_surf, msg_rect)
        
        # 副消息
        if sub_msg:
            sub_surf = self.font_small.render(sub_msg, True, self.theme.text_secondary)
            sub_rect = sub_surf.get_rect(centerx=self.WINDOW_WIDTH // 2, centery=box_y + 70)
            self.screen.blit(sub_surf, sub_rect)
        
        # 统计信息
        stats = [
            f"Total Steps: {self.step_count}",
            f"Total Reward: {self.total_reward:.2f}",
            f"Final Time: {self.net.time}s",
        ]
        
        y_offset = box_y + 100
        for stat in stats:
            stat_surf = self.font_medium.render(stat, True, self.theme.text_primary)
            stat_rect = stat_surf.get_rect(centerx=self.WINDOW_WIDTH // 2, centery=y_offset)
            self.screen.blit(stat_surf, stat_rect)
            y_offset += 28
        
        # 提示
        hint_surf = self.font_small.render("Press [Space] or click Reset to restart", True, self.theme.text_muted)
        hint_rect = hint_surf.get_rect(centerx=self.WINDOW_WIDTH // 2, centery=box_y + box_height - 25)
        self.screen.blit(hint_surf, hint_rect)
        
        pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description="Petri Net 可视化测试环境")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="模型文件路径"
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="不加载模型"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("  PETRI NET WAFER PROCESSING CONSOLE")
    print("  UI/UX Optimized Version - Terminal Green Theme")
    print("=" * 60)
    print()
    print("Keyboard Shortcuts:")
    print("  1-9    : Transition actions")
    print("  W      : WAIT action")
    print("  R      : Random action")
    print("  M      : Model prediction")
    print("  Space  : Reset environment")
    print("  ESC    : Exit")
    print()
    
    model_path = None
    if not args.no_model:
        if args.model:
            model_path = args.model
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            default_model = os.path.join(current_dir, "..", "PPO", "saved_models", "CT_phase2_latest.pt")
            if os.path.exists(default_model):
                model_path = default_model
                print(f"Loading model: {model_path}")
            else:
                print("Warning: Default model not found, Model button will be disabled")
    
    print()
    env = Env_PN()
    visualizer = PetriVisualizer(env, model_path=model_path)
    visualizer.run()


if __name__ == "__main__":
    main()
