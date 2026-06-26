"""Generate synthetic text line images for OCR training.

Uses PIL to render text with various Japanese fonts.
"""
import csv
import random
import string
from pathlib import Path
from typing import Optional, List

from PIL import Image, ImageDraw, ImageFont


# Domain-specific text generators
def _random_registration_date() -> str:
    """Generate a Japanese vehicle registration date."""
    era = random.choice(["H", "R", "S"])
    year = random.randint(1, 31)
    month = random.randint(1, 12)
    formats = [
        f"{era}{year:02d} {month:02d}月",
        f"{era}{year} {month}月",
        f"{era}{year:02d}/{month:02d}",
        f"{era}{year}年{month}月",
    ]
    return random.choice(formats)


def _random_exhibit_number() -> str:
    """Generate auction exhibit numbers."""
    prefix = random.choice(["", f"[{random.randint(1000,9999)}] "])
    number = f"{random.randint(1, 99999):05d}"
    return f"{prefix}{number}"


def _random_door_count() -> str:
    return str(random.choice([2, 3, 4, 5]))


def _random_seating() -> str:
    return str(random.choice([2, 4, 5, 6, 7, 8]))


def _random_interior_grade() -> str:
    return random.choice(["A", "B", "C", "D", "E", "S"])


def _random_number_string() -> str:
    length = random.randint(1, 6)
    return "".join(random.choices(string.digits, k=length))


def _random_mixed_text() -> str:
    """Generate various text patterns."""
    generators = [
        _random_registration_date,
        _random_exhibit_number,
        _random_door_count,
        _random_seating,
        _random_interior_grade,
        _random_number_string,
    ]
    return random.choice(generators)()


def _get_system_fonts() -> List[str]:
    """Find available Japanese-capable fonts on the system."""
    font_paths = []
    search_dirs = [
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
        Path.home() / "Library/Fonts",
    ]
    japanese_font_names = [
        "Hiragino", "YuGothic", "YuMincho", "Osaka",
        "HiraginoSans", "HiraginoSerif",
    ]

    for font_dir in search_dirs:
        if font_dir.exists():
            for f in font_dir.rglob("*.ttf"):
                font_paths.append(str(f))
            for f in font_dir.rglob("*.ttc"):
                font_paths.append(str(f))
            for f in font_dir.rglob("*.otf"):
                font_paths.append(str(f))

    # Filter for Japanese-capable fonts if possible
    ja_fonts = [f for f in font_paths if any(n.lower() in f.lower() for n in japanese_font_names)]
    if ja_fonts:
        return ja_fonts
    return font_paths[:10] if font_paths else []


def render_text_image(
    text: str,
    font_path: Optional[str] = None,
    font_size: int = 24,
    img_height: int = 32,
) -> Image.Image:
    """Render text to a grayscale image."""
    # Load font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Calculate text size
    dummy_img = Image.new("L", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0] + 10  # padding
    text_h = bbox[3] - bbox[1] + 6

    # Create image with white background
    bg_color = random.randint(220, 255)
    img = Image.new("L", (max(text_w, 10), max(text_h, img_height)), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw text
    text_color = random.randint(0, 50)
    x = random.randint(2, 5)
    y = max(0, (img.size[1] - text_h) // 2)
    draw.text((x, y), text, fill=text_color, font=font)

    # Resize to target height maintaining aspect ratio
    w, h = img.size
    new_w = max(1, int(w * img_height / h))
    img = img.resize((new_w, img_height), Image.BILINEAR)

    return img


def generate_synthetic_dataset(
    output_dir: Path,
    num_samples: int,
    charset=None,
    fonts_dir: Optional[str] = None,
    img_height: int = 32,
) -> None:
    """Generate a synthetic OCR dataset.

    Creates:
        output_dir/images/00001.png, ...
        output_dir/labels.csv
    """
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Find fonts
    if fonts_dir and Path(fonts_dir).exists():
        fonts = list(Path(fonts_dir).rglob("*.ttf")) + list(Path(fonts_dir).rglob("*.otf"))
        fonts = [str(f) for f in fonts]
    else:
        fonts = _get_system_fonts()

    if not fonts:
        fonts = [None]  # Use default font

    samples = []
    for i in range(num_samples):
        text = _random_mixed_text()
        font_path = random.choice(fonts)
        font_size = random.randint(18, 28)

        img = render_text_image(
            text=text,
            font_path=font_path,
            font_size=font_size,
            img_height=img_height,
        )

        img_path = img_dir / f"{i:06d}.png"
        img.save(img_path)

        samples.append({
            "image": str(img_path),
            "text": text,
        })

    # Write labels
    label_file = output_dir / "labels.csv"
    with open(label_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "text"])
        writer.writeheader()
        writer.writerows(samples)

    print(f"Generated {num_samples} samples in {output_dir}")


if __name__ == "__main__":
    import argparse
    from config.charset import Charset

    parser = argparse.ArgumentParser(description="Generate synthetic OCR data")
    parser.add_argument("--output", type=str, default="data/train")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--fonts-dir", type=str, default=None)
    args = parser.parse_args()

    cs = Charset()
    generate_synthetic_dataset(
        output_dir=Path(args.output),
        num_samples=args.num_samples,
        charset=cs,
        fonts_dir=args.fonts_dir,
    )
