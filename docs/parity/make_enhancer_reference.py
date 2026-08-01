#!/usr/bin/env python3
"""Stage E1 parity reference: GFPGAN 1.4 and BiSeNet parser on a fixed 512 crop.

The input is a deterministic ffhq_512-aligned crop of the already-swapped pair_01
frame (Stage C final). It is committed as an asset so the mobile parity test can feed
the byte-identical crop to its own ONNX sessions without re-running the swap. This
isolates GFPGAN inference and BiSeNet parsing, exactly like the Stage B raw-output
parity isolated the swapper.

Outputs (written under reference/facefusion-3.7.1/enhancer/pair_01):
  - gfpgan_output_f32le.bin  : GFPGAN raw output, RGB CHW float in [-1, 1], [3,512,512]
  - gfpgan_output.png        : visualisation
  - region_argmax_512.bin    : BiSeNet argmax class id per pixel, uint8 [512,512]
  - region_mask_512.png      : binary Stage E1 region mask (skin/brows/eyes/nose/mouth)
  - metadata.json            : shapes, sha256, config

The committed input crop is written to inputs/pair_01_enhancer_input_512.png.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MODEL_SPECS = {
    "enhancer": (
        "gfpgan_1.4.onnx",
        "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385",
    ),
    "parser": (
        "bisenet_resnet_34.onnx",
        "4a0b8c958a3c938913bd06a8365dbb3c8761afba6ecbf0d14b3b1f77eb230c96",
    ),
}

# FaceFusion Stage E1 protected regions (face_mask_region_set).
REGION_IDS = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(model_dir: Path) -> dict[str, Path]:
    paths = {}
    for role, (name, expected) in MODEL_SPECS.items():
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role} model: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
        paths[role] = path
    return paths


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def gfpgan_forward(sess: ort.InferenceSession, crop_bgr: np.ndarray) -> np.ndarray:
    # FaceFusion prepare_crop_frame: BGR->RGB, /255, (x-0.5)/0.5, NCHW.
    frame = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0
    frame = (frame - 0.5) / 0.5
    frame = np.expand_dims(frame.transpose(2, 0, 1), axis=0).astype(np.float32)
    output = sess.run(None, {"input": frame})[0][0]
    return np.asarray(output, dtype=np.float32)  # [3,512,512], [-1,1]


def parser_forward(sess: ort.InferenceSession, crop_bgr: np.ndarray) -> np.ndarray:
    # FaceFusion create_region_mask preprocessing (BiSeNet, ImageNet stats).
    frame = cv2.resize(crop_bgr, (512, 512))
    frame = frame[:, :, ::-1].astype(np.float32) / 255.0
    frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
    frame = np.expand_dims(frame.transpose(2, 0, 1), axis=0).astype(np.float32)
    output = sess.run(None, {"input": frame})[0]
    logits = np.asarray(output[0], dtype=np.float32)  # [19,512,512]
    return logits.argmax(0).astype(np.uint8)  # [512,512] class ids


def render_gfpgan(raw: np.ndarray) -> np.ndarray:
    rgb = np.clip(raw.transpose(1, 2, 0), -1, 1)
    rgb = (rgb + 1) / 2
    return np.rint(rgb[:, :, ::-1] * 255.0).astype(np.uint8)  # BGR uint8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facefusion-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--parity-root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    sys.path.insert(0, str(args.facefusion_root.resolve()))
    from facefusion.face_helper import warp_face_by_face_landmark_5

    parity_root = args.parity_root
    pair_ref = parity_root / "reference" / "facefusion-3.7.1" / "pair_01"
    swapped_path = pair_ref / "inswapper_final_box_03.png"
    swapped = cv2.imread(str(swapped_path), cv2.IMREAD_COLOR)
    if swapped is None:
        raise RuntimeError(f"Could not read swapped frame: {swapped_path}")
    landmarks = np.array(json.loads((pair_ref / "metadata.json").read_text())["target"]["landmarks_5_xy"], dtype=np.float32)

    crop_512, _ = warp_face_by_face_landmark_5(swapped, landmarks, "ffhq_512", (512, 512))
    input_path = parity_root / "inputs" / "pair_01_enhancer_input_512.png"
    if not cv2.imwrite(str(input_path), crop_512):
        raise RuntimeError(f"Could not write enhancer input crop: {input_path}")

    model_paths = verify(args.model_dir)
    enhancer = session(model_paths["enhancer"])
    parser_sess = session(model_paths["parser"])

    gfpgan_raw = gfpgan_forward(enhancer, crop_512)
    region_argmax = parser_forward(parser_sess, crop_512)
    region_mask = np.isin(region_argmax, REGION_IDS).astype(np.uint8)

    out_dir = pair_ref.parent / "enhancer" / "pair_01"
    out_dir.mkdir(parents=True, exist_ok=True)
    gfpgan_raw.tofile(out_dir / "gfpgan_output_f32le.bin")
    region_argmax.tofile(out_dir / "region_argmax_512.bin")
    cv2.imwrite(str(out_dir / "gfpgan_output.png"), render_gfpgan(gfpgan_raw))
    cv2.imwrite(str(out_dir / "region_mask_512.png"), (region_mask * 255).astype(np.uint8))

    metadata = {
        "input_crop": {
            "file": "inputs/pair_01_enhancer_input_512.png",
            "size": [512, 512],
            "template": "ffhq_512",
            "sha256": sha256(input_path),
            "source": "ffhq_512 warp of pair_01 Stage C swapped frame (inswapper_final_box_03.png)",
        },
        "gfpgan": {
            "model": "gfpgan_1.4",
            "shape": list(gfpgan_raw.shape),
            "range": "[-1,1] RGB CHW",
            "min": float(gfpgan_raw.min()),
            "max": float(gfpgan_raw.max()),
            "mean": float(gfpgan_raw.mean()),
            "sha256_f32le": sha256(out_dir / "gfpgan_output_f32le.bin"),
        },
        "parser": {
            "model": "bisenet_resnet_34",
            "region_ids": REGION_IDS,
            "region_pixels": int(region_mask.sum()),
            "argmax_sha256": sha256(out_dir / "region_argmax_512.bin"),
        },
        "provider": "CPUExecutionProvider",
        "onnxruntime": ort.__version__,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
