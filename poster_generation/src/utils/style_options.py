"""Poster style and visual-density option resolution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


DEFAULT_POSTER_STYLE = "navy_serif"
DEFAULT_VISUAL_DENSITY = "balanced"
DEFAULT_BACKGROUND_STYLE = "auto"
DEFAULT_BACKGROUND_PALETTE = "auto"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def poster_style_presets(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return dict(config.get("poster_style_presets") or {})


def available_poster_styles(config: Dict[str, Any]) -> List[str]:
    presets = poster_style_presets(config)
    if not presets:
        return [DEFAULT_POSTER_STYLE]
    return list(presets.keys())


def normalize_poster_style(value: Any, config: Dict[str, Any]) -> str:
    presets = poster_style_presets(config)
    default = str(config.get("default_poster_style") or DEFAULT_POSTER_STYLE)
    requested = str(value or default).strip()
    if requested in presets:
        return requested
    if default in presets:
        return default
    return next(iter(presets), DEFAULT_POSTER_STYLE)


def _selected_style_preset(state: Dict[str, Any] | None, config: Dict[str, Any]) -> Dict[str, Any]:
    style_name = normalize_poster_style((state or {}).get("poster_style_preset"), config)
    return deepcopy((poster_style_presets(config).get(style_name) or {}))


def resolve_poster_visual_style(state: Dict[str, Any] | None, config: Dict[str, Any]) -> Dict[str, Any]:
    base = deepcopy(config.get("poster_visual_style") or {})
    style_name = normalize_poster_style((state or {}).get("poster_style_preset"), config)
    preset = poster_style_presets(config).get(style_name) or {}
    resolved = _deep_merge(base, preset.get("poster_visual_style") or {})
    resolved["selected_preset"] = style_name
    return resolved


def resolve_typography_config(state: Dict[str, Any] | None, config: Dict[str, Any]) -> Dict[str, Any]:
    typography = deepcopy(config.get("typography") or {})
    preset = _selected_style_preset(state, config)
    return _deep_merge(typography, preset.get("typography") or {})


def resolve_color_scheme_overrides(state: Dict[str, Any] | None, config: Dict[str, Any]) -> Dict[str, str]:
    preset = _selected_style_preset(state, config)
    return deepcopy(preset.get("color_scheme") or {})


def visual_density_presets(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return dict(config.get("visual_density_presets") or {})


def available_visual_densities(config: Dict[str, Any]) -> List[str]:
    presets = visual_density_presets(config)
    if not presets:
        return [DEFAULT_VISUAL_DENSITY]
    return list(presets.keys())


def normalize_visual_density(value: Any, config: Dict[str, Any]) -> str:
    presets = visual_density_presets(config)
    default = str(config.get("default_visual_density") or DEFAULT_VISUAL_DENSITY)
    requested = str(value or default).strip()
    if requested in presets:
        return requested
    if default in presets:
        return default
    return next(iter(presets), DEFAULT_VISUAL_DENSITY)


def resolve_visual_density_settings(state: Dict[str, Any] | None, config: Dict[str, Any]) -> Dict[str, Any]:
    density = normalize_visual_density((state or {}).get("visual_density"), config)
    settings = deepcopy(visual_density_presets(config).get(density) or {})
    settings["name"] = density
    return settings


def background_style_presets(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return dict((config.get("generated_background") or {}).get("styles") or {})


def available_background_styles(config: Dict[str, Any]) -> List[str]:
    presets = background_style_presets(config)
    values = [DEFAULT_BACKGROUND_STYLE]
    values.extend(name for name in presets.keys() if name != DEFAULT_BACKGROUND_STYLE)
    return values


def normalize_background_style(value: Any, config: Dict[str, Any]) -> str:
    presets = background_style_presets(config)
    default = str((config.get("generated_background") or {}).get("style") or DEFAULT_BACKGROUND_STYLE).strip()
    requested = str(value or default or DEFAULT_BACKGROUND_STYLE).strip()
    if requested == DEFAULT_BACKGROUND_STYLE:
        return DEFAULT_BACKGROUND_STYLE
    if requested in presets:
        return requested
    return DEFAULT_BACKGROUND_STYLE


def background_palette_presets(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return dict((config.get("generated_background") or {}).get("palettes") or {})


def available_background_palettes(config: Dict[str, Any]) -> List[str]:
    presets = background_palette_presets(config)
    values = [DEFAULT_BACKGROUND_PALETTE]
    values.extend(name for name in presets.keys() if name != DEFAULT_BACKGROUND_PALETTE)
    return values


def normalize_background_palette(value: Any, config: Dict[str, Any]) -> str:
    presets = background_palette_presets(config)
    default = str((config.get("generated_background") or {}).get("palette") or DEFAULT_BACKGROUND_PALETTE).strip()
    requested = str(value or default or DEFAULT_BACKGROUND_PALETTE).strip()
    if requested == DEFAULT_BACKGROUND_PALETTE:
        return DEFAULT_BACKGROUND_PALETTE
    if requested in presets:
        return requested
    return DEFAULT_BACKGROUND_PALETTE
