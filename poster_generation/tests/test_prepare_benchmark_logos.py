from pathlib import Path

from PIL import Image

from scripts.prepare_benchmark_logos import (
    canonical_institution,
    email_domains,
    image_size,
    normalize_name,
    normalize_raster_without_upscale,
    slugify,
    trim_raster_canvas,
    validate_raster,
)


def test_logo_name_normalization() -> None:
    assert normalize_name("The University of Hong Kong") == "university of hong kong"
    assert slugify("Universita degli Studi di Napoli Federico II") == "universita-degli-studi-di-napoli-federico-ii"
    assert canonical_institution("BDSI, ANU") == "Australian National University"
    assert canonical_institution("KAIST AI") == "Korea Advanced Institute of Science and Technology"


def test_email_domains_ignores_generic_mailboxes() -> None:
    text = "a@sjtu.edu.cn, b@gmail.com, c@cs.example.edu"
    assert email_domains(text) == ["sjtu.edu.cn", "cs.example.edu"]


def test_raster_quality_gate_never_upscales_small_logo(tmp_path: Path) -> None:
    source = tmp_path / "small.png"
    destination = tmp_path / "normalized.png"
    Image.new("RGBA", (64, 64), (20, 40, 80, 255)).save(source)

    assert not normalize_raster_without_upscale(source, destination)
    assert not destination.exists()


def test_raster_quality_gate_accepts_native_high_resolution_logo(tmp_path: Path) -> None:
    source = tmp_path / "large.png"
    destination = tmp_path / "normalized.png"
    Image.new("RGBA", (1200, 500), (20, 40, 80, 255)).save(source)

    assert normalize_raster_without_upscale(source, destination)
    assert image_size(destination) == (1200, 500)
    assert validate_raster(destination)


def test_raster_quality_gate_rejects_white_logo_on_transparent_canvas(tmp_path: Path) -> None:
    path = tmp_path / "white-logo.png"
    image = Image.new("RGBA", (1200, 500), (0, 0, 0, 0))
    for x in range(200, 1000):
        for y in range(180, 320):
            image.putpixel((x, y), (255, 255, 255, 255))
    image.save(path)

    assert not validate_raster(path)


def test_trim_raster_canvas_removes_empty_padding_without_upscaling(tmp_path: Path) -> None:
    path = tmp_path / "padded.png"
    image = Image.new("RGBA", (1200, 500), (0, 0, 0, 0))
    for x in range(400, 800):
        for y in range(150, 350):
            image.putpixel((x, y), (20, 40, 80, 255))
    image.save(path)

    assert trim_raster_canvas(path)
    width, height = image_size(path) or (0, 0)
    assert 400 <= width < 1200
    assert 200 <= height < 500
