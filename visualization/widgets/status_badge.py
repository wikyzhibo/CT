"""
Status Badge Widget - Reusable status indicator component

A compact badge for displaying status with:
- Colored background
- Icon/emoji support
- Rounded corners
- Auto-sizing
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPainter, QColor, QPen
from PySide6.QtWidgets import QLabel

from ..theme import ColorTheme


class StatusBadge(QLabel):
    """A badge widget for displaying status with color coding."""

    def __init__(
        self,
        text: str,
        theme: ColorTheme,
        status_type: str = "default",
        parent=None
    ):
        super().__init__(text, parent)
        self.theme = theme
        self.status_type = status_type
        self.bg_color = self._get_bg_color()
        self.text_color = self._get_text_color()
        
        self._setup_style()
        
    def _get_bg_color(self) -> tuple[int, int, int]:
        """Get background color based on status type."""
        color_map = {
            "idle": self.theme.bg_elevated,
            "busy": self.theme.dim_color(self.theme.warning, 0.3),
            "processing": self.theme.dim_color(self.theme.info, 0.3),
            "complete": self.theme.dim_color(self.theme.success, 0.3),
            "error": self.theme.dim_color(self.theme.danger, 0.3),
            "waiting": self.theme.dim_color(self.theme.warning, 0.2),
            "default": self.theme.bg_surface,
        }
        return color_map.get(self.status_type.lower(), color_map["default"])
        
    def _get_text_color(self) -> tuple[int, int, int]:
        """Get text color based on status type."""
        color_map = {
            "idle": self.theme.text_secondary,
            "busy": self.theme.warning,
            "processing": self.theme.info,
            "complete": self.theme.success,
            "error": self.theme.danger,
            "waiting": self.theme.warning,
            "default": self.theme.text_primary,
        }
        return color_map.get(self.status_type.lower(), color_map["default"])
        
    def _setup_style(self) -> None:
        """Setup badge styling."""
        self.setFont(QFont("Consolas", 9, QFont.Bold))
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"""
            QLabel {{
                color: rgb{self.text_color};
                background-color: rgb{self.bg_color};
                border: 1px solid rgb{self.theme.border_muted};
                border-radius: 10px;
                padding: 3px 10px;
            }}
        """)
        self.setFixedHeight(22)
        
    def update_status(self, text: str, status_type: str) -> None:
        """Update badge text and status type."""
        self.setText(text)
        self.status_type = status_type
        self.bg_color = self._get_bg_color()
        self.text_color = self._get_text_color()
        self._setup_style()


class StatusDot(QLabel):
    """A simple colored dot indicator."""

    def __init__(
        self,
        theme: ColorTheme,
        status_type: str = "default",
        size: int = 8,
        parent=None
    ):
        super().__init__(parent)
        self.theme = theme
        self.status_type = status_type
        self.dot_size = size
        
        self.setFixedSize(size, size)
        self._setup_style()
        
    def _get_color(self) -> tuple[int, int, int]:
        """Get dot color based on status type."""
        color_map = {
            "idle": self.theme.text_muted,
            "busy": self.theme.warning,
            "processing": self.theme.info,
            "complete": self.theme.success,
            "error": self.theme.danger,
            "active": self.theme.accent_cyan,
            "default": self.theme.text_secondary,
        }
        return color_map.get(self.status_type.lower(), color_map["default"])
        
    def _setup_style(self) -> None:
        """Setup dot styling."""
        color = self._get_color()
        self.setStyleSheet(f"""
            QLabel {{
                background-color: rgb{color};
                border-radius: {self.dot_size // 2}px;
            }}
        """)
        
    def update_status(self, status_type: str) -> None:
        """Update dot status."""
        self.status_type = status_type
        self._setup_style()
