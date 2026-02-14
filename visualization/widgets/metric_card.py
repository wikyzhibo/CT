"""
Metric Card Widget - Reusable KPI display component

A polished card component for displaying key metrics with:
- Title, value, and optional subtitle
- Icon support
- Gradient background
- Border accent
- Trend indicator
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QFont, QPainter, QLinearGradient, QPen, QColor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel

from ..theme import ColorTheme


class MetricCard(QWidget):
    """A card widget for displaying a metric with title, value, and optional trend."""

    def __init__(
        self,
        title: str,
        theme: ColorTheme,
        value_color: tuple[int, int, int] | None = None,
        parent=None
    ):
        super().__init__(parent)
        self.theme = theme
        self.title_text = title
        self.value_color = value_color or theme.text_kpi
        self._animated_value = 0.0
        
        self.setMinimumHeight(80)
        self._setup_ui()
        
    def _setup_ui(self) -> None:
        """Setup the card layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        
        # Title (CAPTION - 10pt)
        self.title_label = QLabel(self.title_text)
        self.title_label.setFont(QFont("Consolas", 10))
        self.title_label.setStyleSheet(f"color: rgb{self.theme.text_secondary}; font-weight: 600;")
        layout.addWidget(self.title_label)
        
        # Value row (value + trend)
        value_row = QHBoxLayout()
        value_row.setContentsMargins(0, 0, 0, 0)
        
        # Value (H2 - 18pt)
        self.value_label = QLabel("0")
        self.value_label.setFont(QFont("Consolas", 18, QFont.Bold))
        self.value_label.setStyleSheet(f"color: rgb{self.value_color};")
        value_row.addWidget(self.value_label)
        
        value_row.addStretch()
        
        # Trend indicator (H3 - 14pt)
        self.trend_label = QLabel("")
        self.trend_label.setFont(QFont("Consolas", 14))
        value_row.addWidget(self.trend_label)
        
        layout.addLayout(value_row)
        
        # Subtitle (TINY - 9pt)
        self.subtitle_label = QLabel("")
        self.subtitle_label.setFont(QFont("Consolas", 9))
        self.subtitle_label.setStyleSheet(f"color: rgb{self.theme.text_muted};")
        layout.addWidget(self.subtitle_label)
        
    def set_value(self, value: str | int | float, animate: bool = False) -> None:
        """Set the metric value."""
        if isinstance(value, (int, float)) and animate:
            # Animate numeric values
            self._animate_to_value(float(value))
        else:
            self.value_label.setText(str(value))
            
    def set_subtitle(self, text: str) -> None:
        """Set the subtitle text."""
        self.subtitle_label.setText(text)
        self.subtitle_label.setVisible(bool(text))
        
    def set_trend(self, direction: str, color: tuple[int, int, int] | None = None) -> None:
        """Set trend indicator: 'up', 'down', or empty."""
        if direction == 'up':
            self.trend_label.setText("↑")
            color = color or self.theme.success
        elif direction == 'down':
            self.trend_label.setText("↓")
            color = color or self.theme.danger
        else:
            self.trend_label.setText("")
            return
            
        self.trend_label.setStyleSheet(f"color: rgb{color};")
        
    def _animate_to_value(self, target: float) -> None:
        """Animate value change."""
        # Simple implementation - just set the value for now
        # Full animation would require QPropertyAnimation on a custom property
        self.value_label.setText(f"{target:.0f}")
        
    def paintEvent(self, event) -> None:
        """Custom paint for gradient background and border."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        
        # Gradient background
        gradient = QLinearGradient(0, 0, 0, rect.height())
        gradient.setColorAt(0, QColor(*self.theme.bg_surface))
        gradient.setColorAt(1, QColor(*self.theme.bg_deep))
        
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor(*self.theme.border), 1))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 8, 8)
        
        super().paintEvent(event)
