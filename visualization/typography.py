"""
字体层级系统 (Typography System)

基于 UI/UX Pro Max 设计方法的专业字体层级定义。
提供统一的字体大小、粗细、行高标准，确保整个应用的视觉一致性和可读性。

设计原则：
1. 清晰的层级关系 - 每个层级有明确的用途
2. 适当的对比度 - 相邻层级有足够的视觉区分
3. 可读性优先 - 所有字号都在舒适阅读范围内
4. 响应式考虑 - 使用 pt 单位确保跨 DPI 一致性
"""

from __future__ import annotations
from dataclasses import dataclass
from PySide6.QtGui import QFont


@dataclass
class FontLevel:
    """单个字体层级定义"""
    size_pt: int          # 字号（pt）
    weight: QFont.Weight  # 字重
    line_height: float    # 行高倍数
    description: str      # 用途说明


class TypographySystem:
    """
    字体层级系统
    
    定义了从 H1 到 Caption 的完整字体层级，确保整个应用的字体使用一致性。
    
    层级说明：
    - H1: 主标题（很少使用）
    - H2: 次级标题（区块标题）
    - H3: 三级标题（小节标题）
    - Large: 大号正文（KPI 数值）
    - Body: 标准正文（主要内容）
    - Small: 小号正文（次要信息）
    - Caption: 说明文字（辅助信息）
    - Tiny: 极小文字（标注、角标）
    """
    
    # 字体族
    FONT_FAMILY_MONO = "Consolas"      # 等宽字体（数字、代码）
    FONT_FAMILY_SANS = "Microsoft YaHei UI"  # 无衬线（中文、界面）
    
    # ========== 字体层级定义 ==========
    
    # H1 - 主标题（24pt）
    H1 = FontLevel(
        size_pt=24,
        weight=QFont.Bold,
        line_height=1.2,
        description="主标题 - 页面级标题（很少使用）"
    )
    
    # H2 - 次级标题（18pt）
    H2 = FontLevel(
        size_pt=18,
        weight=QFont.Bold,
        line_height=1.3,
        description="次级标题 - 区块标题（GroupBox 标题）"
    )
    
    # H3 - 三级标题（14pt）
    H3 = FontLevel(
        size_pt=14,
        weight=QFont.DemiBold,
        line_height=1.4,
        description="三级标题 - 小节标题"
    )
    
    # Large - 大号正文（16pt）
    LARGE = FontLevel(
        size_pt=16,
        weight=QFont.DemiBold,
        line_height=1.4,
        description="大号正文 - KPI 数值、重要信息"
    )
    
    # Body - 标准正文（12pt）
    BODY = FontLevel(
        size_pt=12,
        weight=QFont.Normal,
        line_height=1.5,
        description="标准正文 - 主要内容、列表项"
    )
    
    # Small - 小号正文（11pt）
    SMALL = FontLevel(
        size_pt=11,
        weight=QFont.Normal,
        line_height=1.5,
        description="小号正文 - 次要信息、详细说明"
    )
    
    # Caption - 说明文字（10pt）
    CAPTION = FontLevel(
        size_pt=10,
        weight=QFont.Normal,
        line_height=1.4,
        description="说明文字 - 辅助信息、标签"
    )
    
    # Tiny - 极小文字（9pt）
    TINY = FontLevel(
        size_pt=9,
        weight=QFont.Normal,
        line_height=1.3,
        description="极小文字 - 标注、角标、状态徽章"
    )
    
    # ========== 便捷方法 ==========
    
    @classmethod
    def create_font(cls, level: FontLevel, family: str = None) -> QFont:
        """
        根据字体层级创建 QFont 对象
        
        Args:
            level: 字体层级
            family: 字体族（默认使用等宽字体）
        
        Returns:
            配置好的 QFont 对象
        """
        if family is None:
            family = cls.FONT_FAMILY_MONO
        
        font = QFont(family, level.size_pt)
        font.setWeight(level.weight)
        return font
    
    @classmethod
    def get_size(cls, level: FontLevel) -> int:
        """获取字体层级的字号"""
        return level.size_pt
    
    @classmethod
    def get_weight_name(cls, weight: QFont.Weight) -> str:
        """获取字重的 CSS 名称（用于 QSS）"""
        weight_map = {
            QFont.Thin: "100",
            QFont.ExtraLight: "200",
            QFont.Light: "300",
            QFont.Normal: "400",
            QFont.Medium: "500",
            QFont.DemiBold: "600",
            QFont.Bold: "700",
            QFont.ExtraBold: "800",
            QFont.Black: "900",
        }
        return weight_map.get(weight, "400")


# 全局单例
typography = TypographySystem()
