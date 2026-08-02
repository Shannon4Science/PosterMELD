from pathlib import Path

from src.utils.conference_logos import resolve_conference_logo


def test_resolve_conference_logo_prefers_year_specific_asset():
    path = resolve_conference_logo("CVPR 2025")

    assert path is not None
    assert Path(path).name == "cvpr_2025.png"


def test_resolve_conference_logo_keeps_generic_fallback():
    path = resolve_conference_logo("CVPR")

    assert path is not None
    assert Path(path).name == "cvpr.png"
