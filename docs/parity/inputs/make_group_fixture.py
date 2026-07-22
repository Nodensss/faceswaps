"""Build the Stage-D assignment/compositing fixture from approved synthetic portraits."""
from pathlib import Path
import random
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "stage_d_group_target.png"
SEED = 20260722


def portrait(name: str, size: int) -> Image.Image:
    image = Image.open(ROOT / name).convert("RGB")
    # The portraits are square; retain the central face and make scaling deterministic.
    return image.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    random.seed(SEED)
    canvas = Image.new("RGB", (1600, 1100), (31, 42, 57))
    draw = ImageDraw.Draw(canvas)
    # Subtle deterministic background makes pasted identities easy to see.
    for y in range(canvas.height):
        draw.line((0, y, canvas.width, y), fill=(31 + y // 28, 42 + y // 36, 57 + y // 44))
    # First two overlap deliberately: their 128-pixel model paste ROIs overlap after scale.
    placements = [
        ("pair_01_target.png", 80, 130, 520),
        ("pair_02_target.png", 410, 170, 470),
        ("pair_03_target.png", 910, 90, 430),
        ("pair_01_source.png", 1010, 510, 390),
    ]
    for index, (filename, left, top, size) in enumerate(placements, 1):
        image = portrait(filename, size)
        canvas.paste(image, (left, top))
        draw.rectangle((left, top, left + size - 1, top + size - 1), outline=(235, 210, 80), width=5)
        draw.text((left + 14, top + 14), f"T{index}", fill=(255, 245, 180), stroke_width=2, stroke_fill=(0, 0, 0))
    canvas.save(OUT, "PNG", optimize=False)


if __name__ == "__main__":
    main()
