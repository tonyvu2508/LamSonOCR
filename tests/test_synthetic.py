import pytest
from pathlib import Path


def test_generate_samples(tmp_path):
    from scripts.generate_synthetic import generate_synthetic_dataset
    from config.charset import Charset
    cs = Charset()
    generate_synthetic_dataset(
        output_dir=tmp_path,
        num_samples=10,
        charset=cs,
        fonts_dir=None,  # Use system defaults
    )
    assert (tmp_path / "labels.csv").exists()
    assert (tmp_path / "images").is_dir()

    import csv
    with open(tmp_path / "labels.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 10
    for row in rows:
        assert "image" in row
        assert "text" in row
        assert Path(row["image"]).exists()


def test_generated_images_valid(tmp_path):
    from scripts.generate_synthetic import generate_synthetic_dataset
    from config.charset import Charset
    from PIL import Image
    cs = Charset()
    generate_synthetic_dataset(output_dir=tmp_path, num_samples=5, charset=cs)

    import csv
    with open(tmp_path / "labels.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        img = Image.open(row["image"])
        assert img.size[0] > 0
        assert img.size[1] > 0
