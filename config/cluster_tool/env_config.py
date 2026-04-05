"""
Petri 网环境配置。

支持通过 routes（字典或列表）、end_place_name、no_residence_place_names 等配置
多条路线与无驻留腔室。奖励分项开关已固定为全开，不再在配置中暴露。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PetriEnvConfig(BaseModel):
    """
    Petri 网环境配置。

    打印：
        - str(config) / print(config)：简略
        - repr(config) / config.format(detailed=True)：详细
    """

    model_config = ConfigDict(extra="ignore")

    MAX_TIME: int = 2000
    n_wafer: int = 13

    done_event_reward: int = 10
    finish_event_reward: int = 800
    scrap_event_penalty: int = -500
    warn_coef_penalty: float = 2.0
    idle_event_penalty: int = 1000

    transport_overtime_coef_penalty: float = 1.0
    processing_coef_reward: float = 3.0
    in_system_time_penalty_coef: float = 0.0
    time_coef_penalty: float = 1.0

    D_Residual_time: int = 20
    P_Residual_time: int = 15
    max_wafers1_in_system: int = 7
    max_wafers2_in_system: int = 7

    dual_arm: bool = False
    cleaning_enabled: bool = True
    single_route_config: Optional[Dict[str, Any]] = None
    single_route_config_path: Optional[str] = None
    single_route_name: Optional[str] = None
    process_time_map: Optional[Dict[str, int]] = None
    wait_duration: int = 5

    chambers: Optional[Dict[str, Dict[str, Any]]] = None
    cleaning_trigger_wafers_map: Optional[Dict[str, int]] = None
    cleaning_duration_map: Optional[Dict[str, int]] = None

    routes: Optional[Union[Dict[str, List[str]], List[List[str]]]] = None
    start_place_names: Optional[List[str]] = None
    end_place_name: str = "LP_done"
    no_residence_place_names: Optional[Set[str]] = None
    place_display_names: Optional[Dict[str, str]] = None

    @model_validator(mode="after")
    def _normalize_chamber_config(self) -> PetriEnvConfig:
        if self.single_route_config is None:
            raise ValueError("single_route_config must be provided")
        route_name = str(self.single_route_name or "").strip()
        if not route_name:
            raise ValueError("single_route_name must be provided")
        routes_cfg = dict((self.single_route_config or {}).get("routes") or {})
        if route_name not in routes_cfg:
            raise ValueError(f"single_route_name not found in single_route_config.routes: {route_name}")
        self.single_route_name = route_name
        self.process_time_map = {
            str(name): int(value) for name, value in dict(self.process_time_map or {}).items()
        }
        if self.chambers is not None:
            pt_map = dict(self.process_time_map)
            for name, spec in self.chambers.items():
                if not isinstance(spec, dict):
                    continue
                pt = spec.get("process_time")
                if pt is not None:
                    pt_map[name] = int(pt)
            self.process_time_map = pt_map
            self.cleaning_trigger_wafers_map = {
                name: max(0, int(spec.get("cleaning_trigger_wafers", 0)))
                for name, spec in (self.chambers or {}).items()
                if isinstance(spec, dict)
            }
            self.cleaning_duration_map = {
                name: max(0, int(spec.get("cleaning_duration", 0)))
                for name, spec in (self.chambers or {}).items()
                if isinstance(spec, dict)
            }
        else:
            trig_map = dict(self.cleaning_trigger_wafers_map or {})
            dur_map = dict(self.cleaning_duration_map or {})
            if not trig_map and not dur_map:
                route_chambers = dict((self.single_route_config or {}).get("chambers") or {})
                trig_map = {
                    str(name): max(0, int((spec or {}).get("cleaning_trigger_wafers", 0)))
                    for name, spec in route_chambers.items()
                }
                dur_map = {
                    str(name): max(0, int((spec or {}).get("cleaning_duration", 0)))
                    for name, spec in route_chambers.items()
                }
            elif not trig_map or not dur_map:
                raise ValueError("cleaning_trigger_wafers_map and cleaning_duration_map must be provided together")
            self.cleaning_trigger_wafers_map = {
                str(name): max(0, int(value)) for name, value in trig_map.items()
            }
            self.cleaning_duration_map = {
                str(name): max(0, int(value)) for name, value in dur_map.items()
            }
        return self

    def format(self, detailed: bool = False) -> str:
        if detailed:
            return self._format_detailed()
        return self._format_brief()

    def _format_brief(self) -> str:
        lines = ["PetriEnvConfig (简略模式):"]
        lines.append(f"  晶圆数: n_wafer1={self.n_wafer1}, n_wafer2={self.n_wafer2}")

        if self.routes is not None:
            if isinstance(self.routes, dict):
                route_info = f"{len(self.routes)} 条路线: {', '.join(self.routes.keys())}"
            else:
                route_info = f"{len(self.routes)} 条路线"
            lines.append(f"  路线配置: {route_info}")
        else:
            lines.append("  路线配置: 使用默认双路线")

        if self.single_route_config is not None:
            sel = self.single_route_name or "<auto>"
            lines.append(f"  单设备配置驱动路径: enabled (route={sel})")
        elif self.single_route_config_path:
            sel = self.single_route_name or "<auto>"
            lines.append(f"  单设备配置驱动路径文件: {self.single_route_config_path} (route={sel})")
        lines.append(f"  WAIT 时长: {self.wait_duration}s")

        if self.end_place_name != "LP_done":
            lines.append(f"  终点库所: {self.end_place_name}")

        if self.no_residence_place_names:
            lines.append(f"  无驻留腔室: {sorted(self.no_residence_place_names)}")

        lines.append("  奖励分项: 全开（固定，不在配置中切换）")
        return "\n".join(lines)

    def _format_detailed(self) -> str:
        lines = ["PetriEnvConfig (详细模式):"]
        lines.append("=" * 60)

        lines.append("\n【基础配置】")
        lines.append(f"  n_wafer1: {self.n_wafer1}")
        lines.append(f"  n_wafer2: {self.n_wafer2}")
        lines.append(f"  max_wafers1_in_system: {self.max_wafers1_in_system}")
        lines.append(f"  max_wafers2_in_system: {self.max_wafers2_in_system}")

        lines.append("\n【奖励参数】")
        lines.append(f"  done_event_reward: {self.done_event_reward}")
        lines.append(f"  finish_event_reward: {self.finish_event_reward}")
        lines.append(f"  scrap_event_penalty: {self.scrap_event_penalty}")
        lines.append(f"  time_coef: {self.time_coef_penalty}")

        lines.append("\n【路线配置】")
        if self.routes is not None:
            if isinstance(self.routes, dict):
                lines.append(f"  routes (字典, {len(self.routes)} 条):")
                for name, route in self.routes.items():
                    lines.append(f"    {name}: {route}")
            else:
                lines.append(f"  routes (列表, {len(self.routes)} 条):")
                for i, route in enumerate(self.routes, 1):
                    lines.append(f"    路线{i}: {route}")
        else:
            lines.append("  routes: None (使用默认双路线)")

        if self.start_place_names is not None:
            lines.append(f"  start_place_names: {self.start_place_names}")
        else:
            lines.append("  start_place_names: None (从 routes 自动推导)")

        lines.append(f"  end_place_name: {self.end_place_name}")

        if self.no_residence_place_names:
            lines.append(f"  no_residence_place_names: {sorted(self.no_residence_place_names)}")
        else:
            lines.append("  no_residence_place_names: None")

        if self.place_display_names:
            lines.append(f"  place_display_names: {self.place_display_names}")
        else:
            lines.append("  place_display_names: None")

        lines.append(f"  single_route_config: {'set' if self.single_route_config is not None else 'None'}")
        lines.append(f"  single_route_config_path: {self.single_route_config_path}")
        lines.append(f"  single_route_name: {self.single_route_name}")
        lines.append(f"  wait_duration: {self.wait_duration}")
        lines.append("\n【奖励分项】固定全开（无 reward_config）")

        lines.append("=" * 60)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format(detailed=False)

    def __repr__(self) -> str:
        return self.format(detailed=True)

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = self.model_dump(mode="json")
        if path.suffix.lower() in (".yaml", ".yml"):
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    payload,
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"配置已保存到: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> PetriEnvConfig:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"配置文件顶层必须是映射: {path}")
        if "no_residence_place_names" in data and data["no_residence_place_names"] is not None:
            data["no_residence_place_names"] = set(data["no_residence_place_names"])
        route_cfg_path = data.get("single_route_config_path")
        if route_cfg_path and not data.get("single_route_config"):
            route_path = Path(route_cfg_path)
            if not route_path.is_absolute():
                route_path = path.parent / route_path
            with open(route_path, "r", encoding="utf-8") as rf:
                data["single_route_config"] = json.load(rf)
        return cls.model_validate(data)
