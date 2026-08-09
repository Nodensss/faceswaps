"""Build the Stage-D and Stage-E assignment/compositing fixtures.

Two fixtures are produced from the same approved synthetic portraits:

* ``stage_d_group_target.png`` — four faces, three of them assigned. Its bytes are
  frozen: several committed parity artefacts reference this exact image, so the code
  that builds it must not change.
* ``stage_e_dense_pair_target.png`` — two faces standing close enough that the paste
  ROI of the assigned face reaches across the unassigned one. It exists to prove that
  "do not change" is a structural guarantee of both passes rather than a property of
  the roomy Stage-D layout.
"""
from pathlib import Path
import random
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "stage_d_group_target.png"
DENSE_OUT = ROOT / "stage_e_dense_pair_target.png"
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
        ("pair_02_target.png", 330, 170, 470),
        ("pair_03_target.png", 910, 90, 430),
        ("pair_01_source.png", 1010, 510, 390),
    ]
    for index, (filename, left, top, size) in enumerate(placements, 1):
        image = portrait(filename, size)
        if index == 2:
            # Preserve T1 while bringing the actual faces close enough that their
            # affine paste ROIs overlap. The fade is deterministic and affects only
            # the left background edge of T2, not its face.
            mask = Image.new("L", (size, size), 255)
            mask_pixels = mask.load()
            for x in range(140):
                alpha = max(0, min(255, (x - 80) * 255 // 60))
                for y in range(size):
                    mask_pixels[x, y] = alpha
            canvas.paste(image, (left, top), mask)
        else:
            canvas.paste(image, (left, top))
        draw.rectangle((left, top, left + size - 1, top + size - 1), outline=(235, 210, 80), width=5)
        draw.text((left + 14, top + 14), f"T{index}", fill=(255, 245, 180), stroke_width=2, stroke_fill=(0, 0, 0))
    canvas.save(OUT, "PNG", optimize=False)


def dense_pair() -> None:
    """Two faces at the Stage-D T1/T2 spacing, but only the left one gets a source.

    The T1/T2 offsets are reused deliberately: Stage D already proves that pair's
    paste ROIs overlap, so the assigned face here is guaranteed to paste across its
    unassigned neighbour. A is assigned, B must stay bit-identical.
    """
    random.seed(SEED)
    canvas = Image.new("RGB", (860, 700), (31, 42, 57))
    draw = ImageDraw.Draw(canvas)
    for y in range(canvas.height):
        draw.line((0, y, canvas.width, y), fill=(31 + y // 28, 42 + y // 36, 57 + y // 44))
    placements = [
        ("A", "pair_01_target.png", 20, 70, 520),
        ("B", "pair_02_target.png", 270, 110, 470),
    ]
    for label, filename, left, top, size in placements:
        image = portrait(filename, size)
        if label == "B":
            # Same deterministic left-edge fade as Stage D: it keeps A's face visible
            # while the two faces stay close, and never touches B's own face.
            mask = Image.new("L", (size, size), 255)
            mask_pixels = mask.load()
            for x in range(140):
                alpha = max(0, min(255, (x - 80) * 255 // 60))
                for y in range(size):
                    mask_pixels[x, y] = alpha
            canvas.paste(image, (left, top), mask)
        else:
            canvas.paste(image, (left, top))
        draw.rectangle((left, top, left + size - 1, top + size - 1), outline=(235, 210, 80), width=5)
        draw.text((left + 14, top + 14), label, fill=(255, 245, 180), stroke_width=2, stroke_fill=(0, 0, 0))
    canvas.save(DENSE_OUT, "PNG", optimize=False)


if __name__ == "__main__":
    main()
    dense_pair()
