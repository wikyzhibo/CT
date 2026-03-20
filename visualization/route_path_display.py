"""
级联画布顶栏：将 routes.path 格式化为带颜色的 HTML（腔室 / 括号时间 / 方括号清洁）。
"""

from __future__ import annotations

import html

from .theme import ColorTheme


def format_route_path_html(
    path: str,
    theme: ColorTheme,
    *,
    font_family: str,
    font_size_px: int = 15,
) -> str:
    """
    按字符扫描 path，为以下片段着色：
    - 腔室名（含 PM7/PM8、LP_done 等连续段）
    - (…) 加工时间
    - […] 清洁参数
    - -> 分隔符（弱化）
    """
    s = path.strip()
    if not s:
        return ""

    ch = "rgb({},{},{})".format(*theme.accent_cyan)
    tm = "rgb({},{},{})".format(*theme.warning)
    cl = "rgb({},{},{})".format(*theme.complete_orange)
    sep = "rgb({},{},{})".format(*theme.text_muted)

    i = 0
    parts: list[str] = []
    n = len(s)

    while i < n:
        if s.startswith("->", i):
            parts.append(f'<span style="color:{sep};">-&gt;</span>')
            i += 2
            continue
        if s[i] == "(":
            j = s.find(")", i)
            if j < 0:
                parts.append(html.escape(s[i:]))
                break
            parts.append(f'<span style="color:{tm};">{html.escape(s[i : j + 1])}</span>')
            i = j + 1
            continue
        if s[i] == "[":
            j = s.find("]", i)
            if j < 0:
                parts.append(html.escape(s[i:]))
                break
            parts.append(f'<span style="color:{cl};">{html.escape(s[i : j + 1])}</span>')
            i = j + 1
            continue

        j = i
        while j < n:
            if s.startswith("->", j):
                break
            if s[j] in "([":
                break
            j += 1
        if j > i:
            parts.append(f'<span style="color:{ch};">{html.escape(s[i:j])}</span>')
        i = j

    inner = "".join(parts)
    ff = font_family.replace("'", "").replace('"', "") or "Consolas"
    fs = int(font_size_px)
    return (
        f"<div style=\"font-family:'{ff}',monospace;font-size:{fs}px;line-height:1.5;"
        f'font-weight:700;">{inner}</div>'
    )
