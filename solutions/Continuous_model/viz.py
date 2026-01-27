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
    """工业控制面板风格配色方案 - Cyberpunk Terminal + Industrial Control"""
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
    bg_deepest: Tuple[int, int, int] = (18, 22, 30)   # 更深的主背景
    bg_deep: Tuple[int, int, int] = (28, 35, 45)      # 卡片背景
    bg_surface: Tuple[int, int, int] = (40, 48, 60)   # 表面
    bg_elevated: Tuple[int, int, int] = (55, 65, 78)  # 提升层
    
    # 边框
    border: Tuple[int, int, int] = (70, 85, 105)      # 边框
    border_muted: Tuple[int, int, int] = (50, 62, 78) # 弱边框
    border_active: Tuple[int, int, int] = (0, 255, 65)# 活跃边框
    
    # 文字
    text_primary: Tuple[int, int, int] = (230, 237, 243)  # #E6EDF3 主文字
    text_secondary: Tuple[int, int, int] = (139, 148, 158)# #8B949E 次要文字
    text_muted: Tuple[int, int, int] = (100, 110, 125)    # #6E7681 弱化文字
    
    # 特殊效果
    glow_green: Tuple[int, int, int] = (0, 255, 65)
    glow_yellow: Tuple[int, int, int] = (255, 184, 0)
    glow_red: Tuple[int, int, int] = (255, 51, 51)
    glow_cyan: Tuple[int, int, int] = (0, 200, 255)
    glow_orange: Tuple[int, int, int] = (255, 107, 53)
    glow_purple: Tuple[int, int, int] = (168, 85, 247)
    
    # 按钮分类颜色 - Cyberpunk 风格
    btn_transition: Tuple[int, int, int] = (0, 200, 255)    # Cyan - 变迁操作
    btn_wait: Tuple[int, int, int] = (255, 184, 0)          # Yellow - 等待
    btn_random: Tuple[int, int, int] = (168, 85, 247)       # Purple - 随机
    btn_model: Tuple[int, int, int] = (0, 255, 65)          # Green - 模型单步
    btn_auto: Tuple[int, int, int] = (255, 107, 53)         # Orange - 自动模式
    btn_speed: Tuple[int, int, int] = (59, 130, 246)        # Blue - 速度控制
    btn_reset: Tuple[int, int, int] = (255, 51, 51)         # Red - 重置
    
    # 渐变色（用于进度条等）
    gradient_start: Tuple[int, int, int] = (0, 200, 255)    # Cyan
    gradient_end: Tuple[int, int, int] = (0, 255, 65)       # Green
    
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
    
    @staticmethod
    def with_alpha(color: Tuple[int, int, int], alpha: int) -> Tuple[int, int, int, int]:
        """添加透明度"""
        return (color[0], color[1], color[2], alpha)


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
    """按钮类 - 支持分组、快捷键、悬停动画、类型颜色"""
    
    # 按钮类型映射
    BUTTON_TYPES = {
        "transition": "btn_transition",  # 变迁按钮
        "wait": "btn_wait",              # 等待按钮
        "random": "btn_random",          # 随机按钮
        "model": "btn_model",            # 模型单步
        "auto": "btn_auto",              # 自动模式
        "speed": "btn_speed",            # 速度按钮
        "reset": "btn_reset",            # 重置按钮
    }
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 text: str, action_id: int, shortcut: str = "",
                 theme: ColorTheme = None, button_type: str = "transition"):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action_id = action_id
        self.shortcut = shortcut
        self.enabled = False
        self.hovered = False
        self.theme = theme or THEME
        self.button_type = button_type
        self.hover_progress = 0.0  # 悬停动画进度
        self.last_update = pytime.time()
        self.is_active = False  # 用于自动模式等激活状态
    
    def get_type_color(self) -> Tuple[int, int, int]:
        """获取按钮类型对应的颜色"""
        color_attr = self.BUTTON_TYPES.get(self.button_type, "btn_transition")
        return getattr(self.theme, color_attr, self.theme.accent)
    
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
        """绘制按钮 - 带类型颜色和发光效果"""
        self.update()
        
        type_color = self.get_type_color()
        
        # 根据状态选择颜色
        if not self.enabled:
            bg_color = self.theme.bg_elevated
            text_color = self.theme.text_muted
            border_color = self.theme.border_muted
            accent_color = self.theme.dim_color(type_color, 0.3)
        else:
            # 基础背景
            base_bg = self.theme.bg_surface
            hover_bg = self.theme.dim_color(type_color, 0.25)
            bg_color = self.theme.lerp_color(base_bg, hover_bg, self.hover_progress)
            
            # 文字颜色
            text_color = self.theme.lerp_color(
                self.theme.text_primary, 
                type_color,
                self.hover_progress * 0.6
            )
            
            # 边框颜色
            border_color = self.theme.lerp_color(
                self.theme.dim_color(type_color, 0.5),
                type_color,
                self.hover_progress
            )
            accent_color = type_color
        
        # 激活状态发光效果
        if self.is_active and self.enabled:
            # 绘制外发光
            glow_surface = pygame.Surface((self.rect.width + 8, self.rect.height + 8), pygame.SRCALPHA)
            glow_color = (*type_color, 60)
            pygame.draw.rect(glow_surface, glow_color, 
                           (0, 0, self.rect.width + 8, self.rect.height + 8), 
                           border_radius=10)
            screen.blit(glow_surface, (self.rect.x - 4, self.rect.y - 4))
        
        # 绘制按钮背景
        pygame.draw.rect(screen, bg_color, self.rect, border_radius=6)
        
        # 绘制左侧彩色边条
        if self.enabled:
            accent_rect = pygame.Rect(self.rect.x, self.rect.y + 4, 3, self.rect.height - 8)
            pygame.draw.rect(screen, accent_color, accent_rect, border_radius=2)
        
        # 绘制边框
        pygame.draw.rect(screen, border_color, self.rect, 2, border_radius=6)
        
        # 绘制文字
        if self.shortcut and font_small:
            # 左侧主文字（考虑彩色边条偏移）
            text_surf = font.render(self.text, True, text_color)
            text_rect = text_surf.get_rect(midleft=(self.rect.left + 12, self.rect.centery))
            screen.blit(text_surf, text_rect)
            
            # 右侧快捷键标签
            key_text = self.shortcut
            key_bg_color = self.theme.dim_color(type_color, 0.3) if self.enabled else self.theme.bg_deep
            key_text_color = type_color if self.enabled else self.theme.text_muted
            
            key_surf = font_small.render(key_text, True, key_text_color)
            key_width = key_surf.get_width() + 8
            key_height = 18
            key_x = self.rect.right - key_width - 6
            key_y = self.rect.centery - key_height // 2
            
            # 快捷键背景
            key_rect = pygame.Rect(key_x, key_y, key_width, key_height)
            pygame.draw.rect(screen, key_bg_color, key_rect, border_radius=3)
            
            # 快捷键文字
            key_text_rect = key_surf.get_rect(center=key_rect.center)
            screen.blit(key_surf, key_text_rect)
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
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 950
    
    LEFT_PANEL_WIDTH = 320      # 左侧面板（扩大以容纳统计信息）
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
        
        # 自动模式状态变量
        self.auto_mode = False          # 自动模式开关
        self.auto_speed = 1.0           # 速度倍率 (1x=1s间隔)
        self.last_auto_step_time = 0.0  # 上次自动执行时间
        
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
            pygame.K_m: -3,  # Model(1步)
            pygame.K_a: -4,  # Model(auto) 开关
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
        # 新路线：LP -> s1(PM7/PM8) -> s2(LLC) -> s3(PM1/PM2) -> s4(LLD) -> s5(PM9/PM10) -> LP_done
        self.chamber_config = {
            # TM2 区域（活跃）- LP/s1/s2放入/s4取出/s5/LP_done
            "LLA": {"source": "LP", "active": True, "proc_time": 0, "robot": "TM2"},
            "LLB": {"source": "LP_done", "active": True, "proc_time": 0, "robot": "TM2"},
            "PM7": {"source": "s1", "machine": 0, "active": True, "proc_time": 70, "robot": "TM2"},
            "PM8": {"source": "s1", "machine": 1, "active": True, "proc_time": 70, "robot": "TM2"},
            "PM9": {"source": "s5", "machine": 0, "active": True, "proc_time": 70, "robot": "TM2"},
            "PM10": {"source": "s5", "machine": 1, "active": True, "proc_time": 70, "robot": "TM2"},
            # TM3 区域（活跃）- s2取出/s3/s4放入
            "LLC": {"source": "s2", "active": True, "proc_time": 0, "robot": "TM3"},
            "LLD": {"source": "s4", "active": True, "proc_time": 70, "robot": "TM3"},
            "PM1": {"source": "s3", "machine": 0, "active": True, "proc_time": 300, "robot": "TM3"},
            "PM2": {"source": "s3", "machine": 1, "active": True, "proc_time": 300, "robot": "TM3"},
            # TM3 区域（闲置）
            "PM3": {"source": None, "active": False, "proc_time": 0, "robot": "TM3"},
            "PM4": {"source": None, "active": False, "proc_time": 0, "robot": "TM3"},
            "PM5": {"source": None, "active": False, "proc_time": 0, "robot": "TM3"},
            "PM6": {"source": None, "active": False, "proc_time": 0, "robot": "TM3"},
        }
        
        # 显示腔室列表
        self.display_chambers = list(self.chamber_config.keys())
        
        # 运输库所配置（按机械手分组）
        # TM2: d_s1, d_s2, d_s5, d_LP_done
        # TM3: d_s3, d_s4
        self.transports_tm2 = ["d_s1", "d_s2", "d_s5", "d_LP_done"]
        self.transports_tm3 = ["d_s3", "d_s4"]
        self.transports = self.transports_tm2 + self.transports_tm3  # 保持向后兼容
        
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
        """设置动作按钮 - 分组布局，带类型颜色"""
        self.buttons = []
        self.button_groups = {}
        
        t_names = self.net.id2t_name
        
        panel_x = self.WINDOW_WIDTH - self.RIGHT_PANEL_WIDTH + 10
        button_width = self.RIGHT_PANEL_WIDTH - 20
        button_height = 32
        button_gap = 4
        
        y = 72
        
        # 变迁动作组 - Cyan 色
        self.button_groups["transitions"] = {"start_y": y, "buttons": []}
        for i, t_name in enumerate(t_names):
            shortcut = str(i + 1) if i < 9 else ""
            btn = Button(panel_x, y, button_width, button_height, 
                        t_name, i, shortcut, self.theme, button_type="transition")
            self.buttons.append(btn)
            self.button_groups["transitions"]["buttons"].append(btn)
            y += button_height + button_gap
        
        y += 15
        
        # 系统控制组
        self.button_groups["control"] = {"start_y": y, "buttons": []}
        
        # WAIT 按钮 - Yellow 色
        self.wait_button = Button(panel_x, y, button_width, button_height,
                                  "WAIT", self.net.T, "W", self.theme, button_type="wait")
        self.buttons.append(self.wait_button)
        self.button_groups["control"]["buttons"].append(self.wait_button)
        y += button_height + button_gap
        
        # Random 按钮 - Purple 色
        self.random_button = Button(panel_x, y, button_width, button_height,
                                    "Random", -1, "R", self.theme, button_type="random")
        self.buttons.append(self.random_button)
        self.button_groups["control"]["buttons"].append(self.random_button)
        y += button_height + button_gap
        
        # Model(1步) 按钮 - Green 色
        self.model_button = Button(panel_x, y, button_width, button_height,
                                   "Model(1步)", -3, "M", self.theme, button_type="model")
        self.buttons.append(self.model_button)
        self.button_groups["control"]["buttons"].append(self.model_button)
        y += button_height + button_gap
        
        # Model(auto) 按钮 - Orange 色
        self.model_auto_button = Button(panel_x, y, button_width, button_height,
                                        "Model(auto)", -4, "A", self.theme, button_type="auto")
        self.buttons.append(self.model_auto_button)
        self.button_groups["control"]["buttons"].append(self.model_auto_button)
        
        y += button_height + 15
        
        # 速度控制组 - Blue 色
        self.button_groups["speed"] = {"start_y": y, "buttons": []}
        speed_btn_width = (button_width - 12) // 4  # 4个按钮横向排列
        speed_values = [(1, -5), (2, -6), (5, -7), (10, -8)]  # (倍率, action_id)
        self.speed_buttons = []
        
        for i, (speed, action_id) in enumerate(speed_values):
            btn_x = panel_x + i * (speed_btn_width + 4)
            btn = Button(btn_x, y, speed_btn_width, button_height,
                        f"{speed}x", action_id, "", self.theme, button_type="speed")
            self.buttons.append(btn)
            self.speed_buttons.append(btn)
            self.button_groups["speed"]["buttons"].append(btn)
        
        y += button_height + 15
        
        # Reset 按钮 - Red 色
        self.reset_button = Button(panel_x, y, button_width, button_height,
                                   "Reset", -2, "Space", self.theme, button_type="reset")
        self.reset_button.enabled = True
    
    def _update_buttons(self, action_mask: np.ndarray):
        """更新按钮状态"""
        for btn in self.buttons:
            if btn.action_id >= 0 and btn.action_id < len(action_mask):
                btn.enabled = bool(action_mask[btn.action_id])
            elif btn.action_id == -1:  # Random
                btn.enabled = True
            elif btn.action_id == -3:  # Model(1步)
                btn.enabled = self.policy is not None
            elif btn.action_id == -4:  # Model(auto)
                btn.enabled = self.policy is not None
                btn.is_active = self.auto_mode  # 自动模式激活时发光
            elif btn.action_id in [-5, -6, -7, -8]:  # 速度按钮
                btn.enabled = True  # 速度按钮始终可用
        
        # 更新速度按钮高亮状态（当前选中的速度）
        speed_map = {-5: 1, -6: 2, -7: 5, -8: 10}
        for btn in self.speed_buttons:
            if speed_map.get(btn.action_id) == self.auto_speed:
                btn.hovered = True  # 使用 hovered 状态来高亮当前速度
            else:
                btn.hovered = False
    
    def _collect_wafer_info(self) -> Dict[str, List[Dict]]:
        """收集各库所中晶圆信息 - 映射到显示名称
        
        新路线映射：
        - LP -> LLA
        - LP_done -> LLB
        - s1 -> PM7 (machine=0) / PM8 (machine=1)
        - s2 -> LLC
        - s3 -> PM1 (machine=0) / PM2 (machine=1)
        - s4 -> LLD
        - s5 -> PM9 (machine=0) / PM10 (machine=1)
        
        运输库所：
        - d_s1, d_s2, d_s5, d_LP_done -> TM2
        - d_s3, d_s4 -> TM3
        """
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
                if p_name == "LP":
                    # LP 映射到 LLA
                    wafer_info["LLA"].append(wafer_data)
                elif p_name == "LP_done":
                    # LP_done 映射到 LLB
                    wafer_info["LLB"].append(wafer_data)
                elif p_name == "s1":
                    # s1 根据 machine 属性分配到 PM7 或 PM8
                    machine = getattr(tok, 'machine', 0)
                    display_name = "PM7" if machine == 0 else "PM8"
                    wafer_info[display_name].append(wafer_data)
                elif p_name == "s2":
                    # s2 映射到 LLC
                    wafer_info["LLC"].append(wafer_data)
                elif p_name == "s3":
                    # s3 根据 machine 属性分配到 PM1 或 PM2
                    machine = getattr(tok, 'machine', 0)
                    display_name = "PM1" if machine == 0 else "PM2"
                    wafer_info[display_name].append(wafer_data)
                elif p_name == "s4":
                    # s4 映射到 LLD
                    wafer_info["LLD"].append(wafer_data)
                elif p_name == "s5":
                    # s5 根据 machine 属性分配到 PM9 或 PM10
                    machine = getattr(tok, 'machine', 0)
                    display_name = "PM9" if machine == 0 else "PM10"
                    wafer_info[display_name].append(wafer_data)
                elif p_name.startswith("d_"):
                    # 运输库所保持原名，并记录目标腔室
                    if p_name == "d_s1":
                        # d_s1 -> s1 -> PM7 (machine=0) 或 PM8 (machine=1)
                        machine = getattr(tok, 'machine', 0)
                        wafer_data["target_chamber"] = "PM7" if machine == 0 else "PM8"
                    elif p_name == "d_s2":
                        # d_s2 -> s2 -> LLC
                        wafer_data["target_chamber"] = "LLC"
                    elif p_name == "d_s3":
                        # d_s3 -> s3 -> PM1 (machine=0) 或 PM2 (machine=1)
                        machine = getattr(tok, 'machine', 0)
                        wafer_data["target_chamber"] = "PM1" if machine == 0 else "PM2"
                    elif p_name == "d_s4":
                        # d_s4 -> s4 -> LLD
                        wafer_data["target_chamber"] = "LLD"
                    elif p_name == "d_s5":
                        # d_s5 -> s5 -> PM9 (machine=0) 或 PM10 (machine=1)
                        machine = getattr(tok, 'machine', 0)
                        wafer_data["target_chamber"] = "PM9" if machine == 0 else "PM10"
                    elif p_name == "d_LP_done":
                        # d_LP_done -> LP_done -> LLB
                        wafer_data["target_chamber"] = "LLB"
                    else:
                        wafer_data["target_chamber"] = None
                    if p_name in wafer_info:
                        wafer_info[p_name].append(wafer_data)
        
        return wafer_info
    
    # ============ 辅助绘制方法 ============
    
    def _draw_gradient_rect(self, x: int, y: int, width: int, height: int,
                           color1: Tuple[int, int, int], color2: Tuple[int, int, int],
                           horizontal: bool = True, border_radius: int = 0):
        """绘制渐变矩形"""
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        if horizontal:
            for i in range(width):
                t = i / max(1, width - 1)
                color = self.theme.lerp_color(color1, color2, t)
                pygame.draw.line(surface, color, (i, 0), (i, height))
        else:
            for i in range(height):
                t = i / max(1, height - 1)
                color = self.theme.lerp_color(color1, color2, t)
                pygame.draw.line(surface, color, (0, i), (width, i))
        
        if border_radius > 0:
            # 创建圆角遮罩
            mask = pygame.Surface((width, height), pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, width, height), border_radius=border_radius)
            surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        
        self.screen.blit(surface, (x, y))
    
    def _draw_glow_text(self, text: str, font: pygame.font.Font, x: int, y: int,
                       color: Tuple[int, int, int], glow_radius: int = 3):
        """绘制带发光效果的文字"""
        # 发光层
        glow_color = (*self.theme.dim_color(color, 0.4), 80)
        for offset in range(glow_radius, 0, -1):
            alpha = int(60 * (glow_radius - offset + 1) / glow_radius)
            glow_surface = pygame.Surface((font.size(text)[0] + offset * 4, 
                                          font.size(text)[1] + offset * 4), pygame.SRCALPHA)
            glow_text = font.render(text, True, (*color, alpha))
            glow_surface.blit(glow_text, (offset * 2, offset * 2))
            self.screen.blit(glow_surface, (x - offset * 2, y - offset * 2))
        
        # 主文字
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, (x, y))
    
    def _draw_status_indicator(self, x: int, y: int, radius: int, 
                               status: str, pulse: bool = True):
        """绘制状态指示灯"""
        if status == "success" or status == "active":
            color = self.theme.success
        elif status == "warning":
            color = self.theme.warning
        elif status == "danger" or status == "error":
            color = self.theme.danger
        elif status == "info":
            color = self.theme.info
        else:
            color = self.theme.text_muted
        
        # 发光效果
        if pulse and status in ["success", "active", "warning", "danger"]:
            pulse_val = self.anim.pulse(1.5, 0.5, 1.0)
            glow_radius = int(radius + 4 * pulse_val)
            glow_surface = pygame.Surface((glow_radius * 2 + 10, glow_radius * 2 + 10), pygame.SRCALPHA)
            glow_color = (*color, int(60 * pulse_val))
            pygame.draw.circle(glow_surface, glow_color, (glow_radius + 5, glow_radius + 5), glow_radius)
            self.screen.blit(glow_surface, (x - glow_radius - 5, y - glow_radius - 5))
        
        # 主指示灯
        pygame.draw.circle(self.screen, color, (x, y), radius)
        pygame.draw.circle(self.screen, self.theme.brighten_color(color, 1.3), (x, y), radius, 1)
    
    def _draw_stat_card(self, x: int, y: int, width: int, height: int,
                       label: str, value: str, accent_color: Tuple[int, int, int],
                       icon: str = None):
        """绘制统计卡片"""
        # 背景
        card_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.theme.bg_surface, card_rect, border_radius=6)
        
        # 左侧彩色边条
        accent_rect = pygame.Rect(x, y + 3, 3, height - 6)
        pygame.draw.rect(self.screen, accent_color, accent_rect, border_radius=2)
        
        # 标签
        label_surf = self.font_tiny.render(label, True, self.theme.text_secondary)
        self.screen.blit(label_surf, (x + 10, y + 4))
        
        # 值
        value_surf = self.font_small.render(value, True, accent_color)
        value_rect = value_surf.get_rect(right=x + width - 8, centery=y + height // 2)
        self.screen.blit(value_surf, value_rect)
    
    def _draw_section_header(self, x: int, y: int, width: int, title: str,
                            accent_color: Tuple[int, int, int] = None):
        """绘制区域标题（带渐变下划线）"""
        if accent_color is None:
            accent_color = self.theme.accent
        
        # 标题文字
        title_surf = self.font_small.render(title, True, accent_color)
        self.screen.blit(title_surf, (x, y))
        
        # 渐变下划线
        line_y = y + title_surf.get_height() + 2
        self._draw_gradient_rect(x, line_y, width, 2, 
                                accent_color, self.theme.bg_deep, horizontal=True)
        
        return line_y + 6  # 返回下一个元素的 y 坐标
    
    def _draw_mini_progress_bar(self, x: int, y: int, width: int, height: int,
                               progress: float, color: Tuple[int, int, int] = None):
        """绘制迷你进度条"""
        if color is None:
            color = self.theme.success
        
        # 背景
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.theme.bg_elevated, bg_rect, border_radius=height // 2)
        
        # 进度
        if progress > 0:
            fill_width = max(height, int(width * min(1.0, progress)))
            fill_rect = pygame.Rect(x, y, fill_width, height)
            pygame.draw.rect(self.screen, color, fill_rect, border_radius=height // 2)
    
    def _draw_left_panel(self):
        """绘制左侧面板 - Cyberpunk 风格状态面板"""
        panel_x = 8
        panel_y = 8
        panel_width = self.LEFT_PANEL_WIDTH - 16
        panel_height = self.WINDOW_HEIGHT - 16
        content_x = panel_x + 12
        content_width = panel_width - 24
        
        # ===== 面板背景（带微妙渐变）=====
        rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        # 深色背景
        pygame.draw.rect(self.screen, self.theme.bg_deep, rect, border_radius=12)
        # 顶部渐变高光
        highlight_surface = pygame.Surface((panel_width, 60), pygame.SRCALPHA)
        for i in range(60):
            alpha = int(15 * (1 - i / 60))
            pygame.draw.line(highlight_surface, (*self.theme.accent, alpha), (0, i), (panel_width, i))
        self.screen.blit(highlight_surface, (panel_x, panel_y))
        # 边框
        pygame.draw.rect(self.screen, self.theme.border, rect, 2, border_radius=12)
        
        y = panel_y + 12
        
        # ===== 面板标题（带发光效果）=====
        title_text = "SYSTEM MONITOR"
        # 发光背景
        glow_rect = pygame.Rect(content_x - 4, y - 2, content_width + 8, 28)
        glow_surface = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.theme.accent, 20), (0, 0, glow_rect.width, glow_rect.height), border_radius=4)
        self.screen.blit(glow_surface, glow_rect.topleft)
        # 标题文字
        title_surf = self.font_medium.render(title_text, True, self.theme.accent)
        self.screen.blit(title_surf, (content_x, y))
        # 状态指示灯
        status = "active" if not self.done else "warning"
        self._draw_status_indicator(panel_x + panel_width - 20, y + 10, 6, status)
        y += 32
        
        # ===== 核心指标卡片 =====
        card_height = 38
        card_gap = 6
        
        # TIME 卡片 - Cyan 色
        self._draw_metric_card(content_x, y, content_width, card_height,
                              "TIME", f"{self.net.time}s", self.theme.accent,
                              icon_type="clock")
        y += card_height + card_gap
        
        # STEP 卡片 - Green 色
        self._draw_metric_card(content_x, y, content_width, card_height,
                              "STEP", str(self.step_count), self.theme.success,
                              icon_type="counter")
        y += card_height + card_gap
        
        # REWARD 卡片 - 根据值变色
        reward_color = self.theme.success if self.total_reward >= 0 else self.theme.danger
        self._draw_metric_card(content_x, y, content_width, card_height,
                              "REWARD", f"{self.total_reward:+.1f}", reward_color,
                              icon_type="star")
        y += card_height + 8
        
        # ===== 进度条区域 =====
        lp_done_place = next((p for p in self.net.marks if p.name == "LP_done"), None)
        done_wafers = len(lp_done_place.tokens) if lp_done_place else 0
        n_wafer = self.net.n_wafer
        progress = done_wafers / max(1, n_wafer)
        
        # 进度标签
        progress_label = f"WAFER PROGRESS: {done_wafers}/{n_wafer}"
        label_surf = self.font_tiny.render(progress_label, True, self.theme.text_secondary)
        self.screen.blit(label_surf, (content_x, y))
        
        # 百分比
        pct_text = f"{progress * 100:.0f}%"
        pct_surf = self.font_tiny.render(pct_text, True, self.theme.success if progress > 0.5 else self.theme.warning)
        pct_rect = pct_surf.get_rect(right=content_x + content_width, centery=y + 7)
        self.screen.blit(pct_surf, pct_rect)
        y += 18
        
        # 渐变进度条
        bar_height = 10
        bar_rect = pygame.Rect(content_x, y, content_width, bar_height)
        pygame.draw.rect(self.screen, self.theme.bg_elevated, bar_rect, border_radius=5)
        
        if progress > 0:
            fill_width = max(bar_height, int(content_width * progress))
            # 渐变填充
            self._draw_gradient_rect(content_x, y, fill_width, bar_height,
                                    self.theme.gradient_start, self.theme.gradient_end,
                                    horizontal=True, border_radius=5)
            # 发光边缘
            if progress < 1.0:
                glow_x = content_x + fill_width - 2
                pulse = self.anim.pulse(2.0, 0.5, 1.0)
                glow_surface = pygame.Surface((8, bar_height), pygame.SRCALPHA)
                glow_color = (*self.theme.success, int(150 * pulse))
                pygame.draw.rect(glow_surface, glow_color, (0, 0, 8, bar_height), border_radius=3)
                self.screen.blit(glow_surface, (glow_x, y))
        
        y += bar_height + 12
        
        # ===== 分隔线 =====
        self._draw_separator(content_x, y, content_width)
        y += 10
        
        # ===== 产能统计 =====
        stats_data = self.net.calc_wafer_statistics()
        
        y = self._draw_section_header(content_x, y, content_width, "CAPACITY", self.theme.glow_cyan)
        
        # 产能计算
        if stats_data["system_avg"] > 0 and done_wafers > 0:
            throughput = 3600 / stats_data["system_avg"]
        else:
            throughput = 0.0
        
        # 三列布局
        col_width = content_width // 3
        metrics = [
            ("TPT", f"{throughput:.1f}", self.theme.accent),
            ("DONE", f"{stats_data['completed_count']}", self.theme.success),
            ("ACTIVE", f"{stats_data['in_progress_count']}", self.theme.warning),
        ]
        
        for i, (label, value, color) in enumerate(metrics):
            col_x = content_x + i * col_width
            # 小标签
            label_surf = self.font_tiny.render(label, True, self.theme.text_muted)
            label_rect = label_surf.get_rect(centerx=col_x + col_width // 2, top=y)
            self.screen.blit(label_surf, label_rect)
            # 值
            value_surf = self.font_small.render(value, True, color)
            value_rect = value_surf.get_rect(centerx=col_x + col_width // 2, top=y + 14)
            self.screen.blit(value_surf, value_rect)
        
        y += 36
        self._draw_separator(content_x, y, content_width)
        y += 10
        
        # ===== 系统滞留时间 =====
        y = self._draw_section_header(content_x, y, content_width, "SYSTEM RESIDENCE", self.theme.glow_green)
        
        # 横向三列
        sys_metrics = [
            ("AVG", f"{stats_data['system_avg']:.0f}s"),
            ("MAX", f"{stats_data['system_max']}s"),
            ("DIFF", f"{stats_data['system_diff']:.0f}s"),
        ]
        
        for i, (label, value) in enumerate(sys_metrics):
            col_x = content_x + i * col_width
            text = f"{label}: {value}"
            text_surf = self.font_tiny.render(text, True, self.theme.text_primary)
            text_rect = text_surf.get_rect(centerx=col_x + col_width // 2, top=y)
            self.screen.blit(text_surf, text_rect)
        
        y += 20
        self._draw_separator(content_x, y, content_width)
        y += 10
        
        # ===== 腔室滞留时间 =====
        y = self._draw_section_header(content_x, y, content_width, "CHAMBER RESIDENCE", self.theme.glow_orange)
        
        chambers_display = [
            ("PM7/8", self.theme.btn_transition),
            ("PM1/2", self.theme.btn_model),
            ("PM9/10", self.theme.btn_auto),
        ]
        
        for chamber_name, color in chambers_display:
            chamber_data = stats_data["chambers"].get(chamber_name, {"avg": 0, "max": 0})
            avg_val = chamber_data.get("avg", 0)
            max_val = chamber_data.get("max", 0)
            
            # 左侧彩色点 + 名称
            pygame.draw.circle(self.screen, color, (content_x + 5, y + 7), 4)
            name_surf = self.font_tiny.render(chamber_name, True, self.theme.text_secondary)
            self.screen.blit(name_surf, (content_x + 14, y))
            
            # 右侧值
            value_text = f"{avg_val:.0f}s / {max_val}s"
            value_surf = self.font_tiny.render(value_text, True, self.theme.text_primary)
            value_rect = value_surf.get_rect(right=content_x + content_width, centery=y + 7)
            self.screen.blit(value_surf, value_rect)
            y += 16
        
        y += 4
        self._draw_separator(content_x, y, content_width)
        y += 10
        
        # ===== 机械手滞留时间 =====
        y = self._draw_section_header(content_x, y, content_width, "ROBOT RESIDENCE", self.theme.glow_purple)
        
        # TM2 统计
        tm2_times = []
        for t_name in ["d_s1", "d_s2", "d_s5", "d_LP_done"]:
            t_data = stats_data["transports_detail"].get(t_name, {})
            if t_data.get("count", 0) > 0:
                tm2_times.extend([t_data["avg"]] * t_data["count"])
        tm2_avg = sum(tm2_times) / len(tm2_times) if tm2_times else 0
        tm2_max = max([stats_data["transports_detail"].get(t, {}).get("max", 0) 
                       for t in ["d_s1", "d_s2", "d_s5", "d_LP_done"]])
        
        # TM3 统计
        tm3_times = []
        for t_name in ["d_s3", "d_s4"]:
            t_data = stats_data["transports_detail"].get(t_name, {})
            if t_data.get("count", 0) > 0:
                tm3_times.extend([t_data["avg"]] * t_data["count"])
        tm3_avg = sum(tm3_times) / len(tm3_times) if tm3_times else 0
        tm3_max = max([stats_data["transports_detail"].get(t, {}).get("max", 0) 
                       for t in ["d_s3", "d_s4"]])
        
        robot_stats = [
            ("TM2", tm2_avg, tm2_max, self.theme.accent),
            ("TM3", tm3_avg, tm3_max, self.theme.btn_random),
        ]
        
        for robot_name, avg_val, max_val, color in robot_stats:
            pygame.draw.circle(self.screen, color, (content_x + 5, y + 7), 4)
            name_surf = self.font_tiny.render(robot_name, True, self.theme.text_secondary)
            self.screen.blit(name_surf, (content_x + 14, y))
            
            value_text = f"{avg_val:.0f}s / {max_val}s"
            value_surf = self.font_tiny.render(value_text, True, self.theme.text_primary)
            value_rect = value_surf.get_rect(right=content_x + content_width, centery=y + 7)
            self.screen.blit(value_surf, value_rect)
            y += 16
        
        y += 4
        self._draw_separator(content_x, y, content_width)
        y += 10
        
        # ===== 图例 =====
        legend_title = self.font_tiny.render("STATUS LEGEND", True, self.theme.text_muted)
        self.screen.blit(legend_title, (content_x, y))
        y += 16
        
        legends = [
            (self.theme.success, "OK"),
            (self.theme.warning, "WARN"),
            (self.theme.danger, "CRIT"),
        ]
        
        x_offset = content_x
        for color, text in legends:
            # 发光圆点
            pygame.draw.circle(self.screen, self.theme.dim_color(color, 0.3), (x_offset + 5, y + 5), 7)
            pygame.draw.circle(self.screen, color, (x_offset + 5, y + 5), 4)
            text_surf = self.font_tiny.render(text, True, self.theme.text_primary)
            self.screen.blit(text_surf, (x_offset + 14, y))
            x_offset += 14 + text_surf.get_width() + 16
        y += 18
        
        self._draw_separator(content_x, y, content_width)
        y += 10
        
        # ===== 动作历史 (时间轴样式) =====
        y = self._draw_section_header(content_x, y, content_width, "ACTION HISTORY", self.theme.accent)
        
        timeline_x = content_x + 8
        max_y = panel_y + panel_height - 10
        
        for i, entry in enumerate(reversed(self.action_history[-7:])):
            if y >= max_y - 20:
                break
            
            step = entry['step']
            action_name = entry['action']
            total = entry['total']
            
            # 时间轴节点颜色
            node_color = self.theme.success if total > 0 else (
                self.theme.danger if total < -5 else self.theme.warning
            )
            
            # 发光节点
            glow_surface = pygame.Surface((16, 16), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*node_color, 60), (8, 8), 7)
            self.screen.blit(glow_surface, (timeline_x - 8, y - 1))
            pygame.draw.circle(self.screen, node_color, (timeline_x, y + 7), 4)
            
            # 时间轴线
            if i < len(self.action_history) - 1 and y + 22 < max_y:
                pygame.draw.line(self.screen, self.theme.border_muted,
                               (timeline_x, y + 12), (timeline_x, y + 22), 1)
            
            # 步数
            step_text = f"#{step:02d}"
            step_surf = self.font_tiny.render(step_text, True, self.theme.text_muted)
            self.screen.blit(step_surf, (timeline_x + 10, y))
            
            # 动作名
            action_surf = self.font_tiny.render(action_name, True, self.theme.text_primary)
            self.screen.blit(action_surf, (timeline_x + 40, y))
            
            # 奖励（带背景）
            reward_text = f"{total:+.1f}"
            reward_surf = self.font_tiny.render(reward_text, True, node_color)
            reward_rect = reward_surf.get_rect(right=content_x + content_width, top=y)
            
            # 奖励背景
            reward_bg = pygame.Rect(reward_rect.x - 4, reward_rect.y - 1, 
                                   reward_rect.width + 8, reward_rect.height + 2)
            pygame.draw.rect(self.screen, self.theme.dim_color(node_color, 0.15), reward_bg, border_radius=3)
            self.screen.blit(reward_surf, reward_rect)
            
            y += 22
    
    def _draw_metric_card(self, x: int, y: int, width: int, height: int,
                         label: str, value: str, color: Tuple[int, int, int],
                         icon_type: str = None):
        """绘制指标卡片（带左侧彩色边条和图标）"""
        # 背景
        card_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.theme.bg_surface, card_rect, border_radius=6)
        
        # 左侧彩色边条
        accent_rect = pygame.Rect(x, y + 4, 4, height - 8)
        pygame.draw.rect(self.screen, color, accent_rect, border_radius=2)
        
        # 图标区域（简化为彩色方块）
        icon_x = x + 12
        icon_size = 20
        icon_rect = pygame.Rect(icon_x, y + (height - icon_size) // 2, icon_size, icon_size)
        pygame.draw.rect(self.screen, self.theme.dim_color(color, 0.2), icon_rect, border_radius=4)
        
        # 图标符号
        if icon_type == "clock":
            symbol = "◷"
        elif icon_type == "counter":
            symbol = "▣"
        elif icon_type == "star":
            symbol = "★"
        else:
            symbol = "●"
        symbol_surf = self.font_small.render(symbol, True, color)
        symbol_rect = symbol_surf.get_rect(center=icon_rect.center)
        self.screen.blit(symbol_surf, symbol_rect)
        
        # 标签
        label_surf = self.font_tiny.render(label, True, self.theme.text_secondary)
        self.screen.blit(label_surf, (icon_x + icon_size + 8, y + 6))
        
        # 值
        value_surf = self.font_medium.render(value, True, color)
        value_rect = value_surf.get_rect(right=x + width - 10, centery=y + height // 2)
        self.screen.blit(value_surf, value_rect)
    
    def _draw_separator(self, x: int, y: int, width: int):
        """绘制渐变分隔线"""
        for i in range(width):
            t = abs(i - width // 2) / (width // 2)
            alpha = int(60 * (1 - t))
            color = (*self.theme.border, alpha)
            surface = pygame.Surface((1, 1), pygame.SRCALPHA)
            surface.fill(color)
            self.screen.blit(surface, (x + i, y))
    
    def _draw_right_panel(self):
        """绘制右侧面板 - Cyberpunk 风格控制面板"""
        panel_x = self.WINDOW_WIDTH - self.RIGHT_PANEL_WIDTH
        panel_y = 8
        panel_width = self.RIGHT_PANEL_WIDTH - 8
        panel_height = self.WINDOW_HEIGHT - 16
        content_x = panel_x + 10
        content_width = panel_width - 20
        
        # ===== 面板背景（带微妙渐变）=====
        rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, self.theme.bg_deep, rect, border_radius=12)
        
        # 顶部渐变高光
        highlight_surface = pygame.Surface((panel_width, 50), pygame.SRCALPHA)
        for i in range(50):
            alpha = int(12 * (1 - i / 50))
            pygame.draw.line(highlight_surface, (*self.theme.btn_transition, alpha), (0, i), (panel_width, i))
        self.screen.blit(highlight_surface, (panel_x, panel_y))
        
        # 边框
        pygame.draw.rect(self.screen, self.theme.border, rect, 2, border_radius=12)
        
        # ===== 面板标题 =====
        title_text = "CONTROL PANEL"
        # 标题背景
        title_bg_rect = pygame.Rect(content_x - 2, panel_y + 10, content_width + 4, 26)
        pygame.draw.rect(self.screen, (*self.theme.btn_transition, 15), title_bg_rect, border_radius=4)
        # 标题文字
        title_surf = self.font_medium.render(title_text, True, self.theme.btn_transition)
        self.screen.blit(title_surf, (content_x, panel_y + 13))
        # 状态指示灯
        self._draw_status_indicator(panel_x + panel_width - 18, panel_y + 23, 5, 
                                   "active" if not self.done else "warning")
        
        # ===== 变迁组标题（带彩色边条）=====
        trans_y = self.button_groups["transitions"]["start_y"] - 24
        self._draw_group_header(content_x, trans_y, content_width, "TRANSITIONS", 
                               self.theme.btn_transition)
        
        # ===== 控制组标题 =====
        ctrl_y = self.button_groups["control"]["start_y"] - 24
        self._draw_group_header(content_x, ctrl_y, content_width, "CONTROL", 
                               self.theme.btn_model)
        
        # ===== 速度组标题 =====
        speed_y = self.button_groups["speed"]["start_y"] - 20
        self._draw_group_header(content_x, speed_y, content_width, "SPEED", 
                               self.theme.btn_speed)
        
        # 自动模式状态指示（带背景）
        if self.auto_mode:
            auto_status = f"AUTO: {int(self.auto_speed)}x"
            status_color = self.theme.btn_auto
            # 发光背景
            pulse = self.anim.pulse(1.5, 0.6, 1.0)
            glow_surface = pygame.Surface((70, 18), pygame.SRCALPHA)
            glow_color = (*status_color, int(40 * pulse))
            pygame.draw.rect(glow_surface, glow_color, (0, 0, 70, 18), border_radius=4)
            self.screen.blit(glow_surface, (panel_x + panel_width - 78, speed_y - 1))
        else:
            auto_status = "AUTO: OFF"
            status_color = self.theme.text_muted
        
        # 状态背景
        status_bg_rect = pygame.Rect(panel_x + panel_width - 76, speed_y, 66, 16)
        pygame.draw.rect(self.screen, self.theme.dim_color(status_color, 0.15), status_bg_rect, border_radius=3)
        
        auto_surf = self.font_tiny.render(auto_status, True, status_color)
        auto_rect = auto_surf.get_rect(center=status_bg_rect.center)
        self.screen.blit(auto_surf, auto_rect)
        
        # ===== 绘制所有按钮 =====
        for btn in self.buttons:
            btn.draw(self.screen, self.font_small, self.font_tiny)
        
        # ===== Reset 按钮 =====
        self.reset_button.draw(self.screen, self.font_small, self.font_tiny)
        
        # ===== 快捷键提示区域 =====
        hint_y = panel_y + panel_height - 35
        
        # 提示背景
        hint_bg_rect = pygame.Rect(content_x, hint_y, content_width, 28)
        pygame.draw.rect(self.screen, self.theme.bg_surface, hint_bg_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.theme.border_muted, hint_bg_rect, 1, border_radius=6)
        
        # 提示标题
        hint_title = self.font_tiny.render("SHORTCUTS", True, self.theme.text_muted)
        hint_title_rect = hint_title.get_rect(centerx=panel_x + panel_width // 2, centery=hint_y + 8)
        self.screen.blit(hint_title, hint_title_rect)
        
        # 快捷键列表
        shortcuts_text = "1-9 W R M A Space"
        shortcuts_surf = self.font_tiny.render(shortcuts_text, True, self.theme.accent)
        shortcuts_rect = shortcuts_surf.get_rect(centerx=panel_x + panel_width // 2, centery=hint_y + 20)
        self.screen.blit(shortcuts_surf, shortcuts_rect)
    
    def _draw_group_header(self, x: int, y: int, width: int, title: str,
                          accent_color: Tuple[int, int, int]):
        """绘制按钮组标题（带左侧彩色边条）"""
        # 左侧彩色边条
        bar_rect = pygame.Rect(x - 6, y + 2, 3, 14)
        pygame.draw.rect(self.screen, accent_color, bar_rect, border_radius=2)
        
        # 标题文字
        title_surf = self.font_small.render(title, True, self.theme.text_secondary)
        self.screen.blit(title_surf, (x, y))
    
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
        """绘制双机械手缓冲区（TM2 和 TM3）
        
        TM2 负责: d_s1, d_s2, d_s5, d_LP_done
        TM3 负责: d_s3, d_s4
        """
        # 收集 TM2 运输中的晶圆
        tm2_wafers = []
        for t_name in self.transports_tm2:
            tm2_wafers.extend(wafer_info.get(t_name, []))
        
        # 收集 TM3 运输中的晶圆
        tm3_wafers = []
        for t_name in self.transports_tm3:
            tm3_wafers.extend(wafer_info.get(t_name, []))
        
        # 收集目标腔室（用于高亮连接线）
        target_chambers = set()
        for wafer in tm2_wafers + tm3_wafers:
            target = wafer.get("target_chamber")
            if target:
                target_chambers.add(target)
        
        # ========== 先绘制连接线（在底层） ==========
        self._draw_robot_connections(target_chambers)
        
        # ========== 绘制 TM2（活跃） ==========
        self._draw_single_robot(
            self.tm2_pos[0], self.tm2_pos[1], 
            "TM2", is_busy=len(tm2_wafers) > 0, 
            is_active=True, wafers=tm2_wafers
        )
        
        # ========== 绘制 TM3（活跃） ==========
        self._draw_single_robot(
            self.tm3_pos[0], self.tm3_pos[1],
            "TM3", is_busy=len(tm3_wafers) > 0, 
            is_active=True, wafers=tm3_wafers
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
        
        # 更新按钮悬停状态（但保留速度按钮的高亮状态）
        for btn in self.buttons:
            if btn not in self.speed_buttons:
                btn.check_hover(mouse_pos)
            else:
                # 速度按钮：只在鼠标悬停时临时高亮，否则保持选中状态
                is_mouse_hover = btn.rect.collidepoint(mouse_pos)
                speed_map = {-5: 1, -6: 2, -7: 5, -8: 10}
                is_selected = speed_map.get(btn.action_id) == self.auto_speed
                btn.hovered = is_mouse_hover or is_selected
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
                    elif action_id == -3:  # Model(1步)
                        if self.policy is not None:
                            return -3
                    elif action_id == -4:  # Model(auto) 开关
                        if self.policy is not None:
                            return -4
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
        self.auto_mode = False  # 重置自动模式
        self.last_auto_step_time = 0.0  # 重置自动执行计时器
        
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
                    self.auto_mode = False  # 重置时关闭自动模式
                elif action == -1:
                    action_mask = self.td["action_mask"].numpy()
                    valid_actions = np.where(action_mask)[0]
                    if len(valid_actions) > 0:
                        action = int(np.random.choice(valid_actions))
                        self._execute_action(action)
                elif action == -3:  # Model(1步)
                    model_action = self.get_model_action()
                    if model_action is not None:
                        self._execute_action(model_action)
                elif action == -4:  # Model(auto) 开关
                    self.auto_mode = not self.auto_mode
                    if self.auto_mode:
                        self.last_auto_step_time = pytime.time()  # 重置计时器
                elif action in [-5, -6, -7, -8]:  # 速度按钮
                    speed_map = {-5: 1, -6: 2, -7: 5, -8: 10}
                    self.auto_speed = speed_map[action]
                    # 更新速度按钮高亮
                    if self.td is not None:
                        self._update_buttons(self.td["action_mask"].numpy())
                else:
                    self._execute_action(action)
            
            # 自动模式执行
            if self.auto_mode and not self.done and self.policy is not None:
                current_time = pytime.time()
                interval = 1.0 / self.auto_speed  # 1x=1s, 2x=0.5s, 5x=0.2s, 10x=0.1s
                if current_time - self.last_auto_step_time >= interval:
                    model_action = self.get_model_action()
                    if model_action is not None:
                        self._execute_action(model_action)
                        self.last_auto_step_time = current_time
            
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
            default_model = os.path.join(current_dir, "..", "PPO", "saved_models","keepmodels", "CT_phase2_best.pt")
            if os.path.exists(default_model):
                model_path = default_model
                print(f"Loading model: {model_path}")
            else:
                print("Warning: Default model not found, Model button will be disabled")
    
    print()
    # 为可视化禁用极速模式（需要详细奖励信息和完整统计追踪）
    env = Env_PN(enable_turbo=False)
    print("Note: Turbo mode disabled for visualization (enables detailed reward tracking)")
    print()
    visualizer = PetriVisualizer(env, model_path=model_path)
    visualizer.run()


if __name__ == "__main__":
    main()
