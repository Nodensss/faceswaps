#!/usr/bin/env python3
"""Create the isolated Stage E1 GFPGAN/BiSeNet desktop reference.

The canonical 512x512 PNG is extracted only by the pinned desktop FaceFusion
checkout from the already committed Stage C desktop result. Reference inference
then reopens that PNG from disk. Android consumes the exact same committed bytes;
no Android crop, swapper, compositor, coordinator, or UI participates here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


FACEFUSION_COMMIT = "3f81a8a78454089d720b8f318a12ae1702c4633b"
CANONICAL_INPUT_SHA256 = "5987781f96010ceddbf7445b26bb5420b56e20138d9603a352a48a57f0fb2ec8"
CANONICAL_INPUT_SIZE = 350_560
CANONICAL_DECODED_BGR_SHA256 = "42239023bbb07502fd60efb589a5186c94b5207401be2b488976076b3ee84eea"
STAGE_C_FRAME_SHA256 = "eeada935b979fd02c34504a777681d618fd062ec3949117fa336c25d2b026afe"
CANONICAL_AFFINE_MATRIX = [
    [0.5064969784282285, 0.00663733526187763, -64.4590473770804],
    [-0.00663733526187763, 0.5064969784282285, -24.847903691598972],
]
MODEL_SPECS = {
    "gfpgan": {
        "file": "gfpgan_1.4.onnx",
        "size": 340_299_087,
        "sha256": "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385",
    },
    "bisenet": {
        "file": "bisenet_resnet_34.onnx",
        "size": 93_632_546,
        "sha256": "4a0b8c958a3c938913bd06a8365dbb3c8761afba6ecbf0d14b3b1f77eb230c96",
    },
}
REGION_NAMES = [
    "skin",
    "left-eyebrow",
    "right-eyebrow",
    "left-eye",
    "right-eye",
    "glasses",
    "nose",
    "mouth",
    "upper-lip",
    "lower-lip",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, expected_size: int | None, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {path}: {path.stat().st_size} != {expected_size}"
        )
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected_sha256}")


def verify_facefusion_checkout(facefusion_root: Path) -> None:
    actual = subprocess.check_output(
        ["git", "-C", str(facefusion_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != FACEFUSION_COMMIT:
        raise RuntimeError(
            f"FaceFusion checkout mismatch: {actual} != {FACEFUSION_COMMIT}"
        )


def extract_canonical_input(parity_root: Path, warp_face_by_face_landmark_5) -> dict[str, object]:
    pair_dir = parity_root / "reference" / "facefusion-3.7.1" / "pair_01"
    stage_c_frame = pair_dir / "inswapper_final_box_03.png"
    pair_metadata_path = pair_dir / "metadata.json"
    require_file(stage_c_frame, None, STAGE_C_FRAME_SHA256)
    pair_metadata = json.loads(pair_metadata_path.read_text(encoding="utf-8"))
    landmarks = np.asarray(
        pair_metadata["target"]["landmarks_5_xy"], dtype=np.float32
    )
    frame = cv2.imread(str(stage_c_frame), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not decode desktop Stage C frame: {stage_c_frame}")

    crop, affine_matrix = warp_face_by_face_landmark_5(
        frame, landmarks, "ffhq_512", (512, 512)
    )
    input_path = parity_root / "inputs" / "pair_01_face_quality_input_512.png"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(input_path), crop):
        raise RuntimeError(f"Could not write canonical crop: {input_path}")
    require_file(input_path, CANONICAL_INPUT_SIZE, CANONICAL_INPUT_SHA256)
    actual_matrix = np.asarray(affine_matrix, dtype=np.float64)
    if not np.allclose(actual_matrix, np.asarray(CANONICAL_AFFINE_MATRIX), atol=1e-12):
        raise RuntimeError(f"Canonical affine matrix changed: {actual_matrix.tolist()}")
    return {
        "source_frame": "reference/facefusion-3.7.1/pair_01/inswapper_final_box_03.png",
        "source_frame_sha256": STAGE_C_FRAME_SHA256,
        "source_landmarks": "reference/facefusion-3.7.1/pair_01/metadata.json#target.landmarks_5_xy",
        "template": "ffhq_512",
        "size": [512, 512],
        "affine_matrix": actual_matrix.tolist(),
    }


class CapturingSession:
    def __init__(
        self,
        session: ort.InferenceSession,
        requested_outputs: list[str] | None = None,
    ):
        self.session = session
        self.requested_outputs = requested_outputs
        self.last_outputs: list[np.ndarray] | None = None
        self.last_input_sha256: str | None = None

    def get_inputs(self):
        return self.session.get_inputs()

    def run(self, output_names, inputs):
        input_tensor = np.ascontiguousarray(inputs["input"], dtype="<f4")
        self.last_input_sha256 = hashlib.sha256(input_tensor.tobytes()).hexdigest()
        self.last_outputs = self.session.run(self.requested_outputs or output_names, inputs)
        return self.last_outputs


def create_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.inter_op_num_threads = 1
    # A single thread makes the committed float reference reproducible and matches
    # the Android parity session. Performance tuning belongs to later integration.
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def render_gfpgan(raw_chw: np.ndarray) -> np.ndarray:
    rgb = np.clip(raw_chw.transpose(1, 2, 0), -1.0, 1.0)
    rgb = (rgb + 1.0) / 2.0
    return np.rint(rgb[:, :, ::-1] * 255.0).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facefusion-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--parity-root", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--regenerate-canonical-input",
        action="store_true",
        help="Re-extract the committed 512 PNG from the desktop Stage C reference.",
    )
    args = parser.parse_args()

    verify_facefusion_checkout(args.facefusion_root)
    sys.path.insert(0, str(args.facefusion_root.resolve()))
    from facefusion import face_masker, state_manager
    from facefusion.face_helper import warp_face_by_face_landmark_5
    from facefusion.processors.modules.face_enhancer import core as face_enhancer

    provenance: dict[str, object] = {
        "source_frame": "reference/facefusion-3.7.1/pair_01/inswapper_final_box_03.png",
        "source_frame_sha256": STAGE_C_FRAME_SHA256,
        "source_landmarks": "reference/facefusion-3.7.1/pair_01/metadata.json#target.landmarks_5_xy",
        "template": "ffhq_512",
        "size": [512, 512],
        "affine_matrix": CANONICAL_AFFINE_MATRIX,
    }
    if args.regenerate_canonical_input:
        provenance = extract_canonical_input(
            args.parity_root, warp_face_by_face_landmark_5
        )

    input_path = args.parity_root / "inputs" / "pair_01_face_quality_input_512.png"
    require_file(input_path, CANONICAL_INPUT_SIZE, CANONICAL_INPUT_SHA256)
    # Reopen the committed PNG: this is the exact byte source Android decodes too.
    crop_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if crop_bgr is None or crop_bgr.shape != (512, 512, 3):
        raise RuntimeError(f"Unexpected canonical crop: {input_path} {crop_bgr.shape}")
    decoded_bgr_sha256 = hashlib.sha256(np.ascontiguousarray(crop_bgr).tobytes()).hexdigest()
    if decoded_bgr_sha256 != CANONICAL_DECODED_BGR_SHA256:
        raise RuntimeError(
            f"Decoded canonical crop changed: {decoded_bgr_sha256} != "
            f"{CANONICAL_DECODED_BGR_SHA256}"
        )

    model_paths: dict[str, Path] = {}
    for role, spec in MODEL_SPECS.items():
        model_path = args.model_dir / str(spec["file"])
        require_file(model_path, int(spec["size"]), str(spec["sha256"]))
        model_paths[role] = model_path

    # Static model metadata calls resolve_download_url even though inference uses the
    # injected, checksum-verified local sessions. Keep the reference strictly offline.
    face_masker.resolve_download_url = (
        lambda model_set, name: f"offline://{model_set}/{name}"
    )
    prepared_crop = face_enhancer.prepare_crop_frame(crop_bgr)
    gfpgan_session = CapturingSession(create_session(model_paths["gfpgan"]))
    face_enhancer.get_inference_pool = lambda: {"face_enhancer": gfpgan_session}
    started = time.perf_counter()
    gfpgan_raw = np.asarray(
        face_enhancer.forward(prepared_crop, np.asarray([0.5], dtype=np.float64)),
        dtype=np.float32,
    )
    gfpgan_ms = (time.perf_counter() - started) * 1000.0
    gfpgan_input_sha256 = gfpgan_session.last_input_sha256
    face_enhancer.get_inference_pool = lambda: {}
    del gfpgan_session
    gc.collect()

    # The production function asks for all outputs. The injected wrapper requests only
    # the main `output`, which is the only tensor create_region_mask reads.
    bisenet_session = CapturingSession(
        create_session(model_paths["bisenet"]), requested_outputs=["output"]
    )
    face_masker.get_inference_pool = lambda: {"bisenet_resnet_34": bisenet_session}
    state_manager.init_item("face_parser_model", "bisenet_resnet_34")
    started = time.perf_counter()
    region_mask = np.asarray(
        face_masker.create_region_mask(crop_bgr, REGION_NAMES), dtype=np.float32
    )
    bisenet_ms = (time.perf_counter() - started) * 1000.0
    if bisenet_session.last_outputs is None:
        raise RuntimeError("FaceFusion did not invoke the injected BiSeNet session")
    bisenet_input_sha256 = bisenet_session.last_input_sha256
    bisenet_logits = np.asarray(bisenet_session.last_outputs[0][0], dtype=np.float32)
    bisenet_argmax = bisenet_logits.argmax(axis=0).astype(np.uint8)

    if gfpgan_raw.shape != (3, 512, 512):
        raise RuntimeError(f"Unexpected GFPGAN output shape: {gfpgan_raw.shape}")
    if bisenet_argmax.shape != (512, 512):
        raise RuntimeError(f"Unexpected BiSeNet argmax shape: {bisenet_argmax.shape}")
    if region_mask.shape != (512, 512):
        raise RuntimeError(f"Unexpected FaceFusion region mask shape: {region_mask.shape}")

    out_dir = (
        args.parity_root
        / "reference"
        / "facefusion-3.7.1"
        / "face_quality"
        / "pair_01"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(gfpgan_raw, dtype="<f4").tofile(
        out_dir / "gfpgan_output_f32le.bin"
    )
    bisenet_argmax.tofile(out_dir / "bisenet_argmax_u8.bin")
    np.ascontiguousarray(region_mask, dtype="<f4").tofile(
        out_dir / "bisenet_region_mask_f32le.bin"
    )
    cv2.imwrite(str(out_dir / "gfpgan_output.png"), render_gfpgan(gfpgan_raw))
    cv2.imwrite(
        str(out_dir / "bisenet_region_mask.png"),
        np.rint(np.clip(region_mask, 0.0, 1.0) * 255.0).astype(np.uint8),
    )

    metadata = {
        "facefusion": {
            "version": "3.7.1",
            "commit": FACEFUSION_COMMIT,
            "instrumented_functions": [
                "face_helper.warp_face_by_face_landmark_5",
                "face_enhancer.prepare_crop_frame",
                "face_enhancer.forward",
                "face_masker.create_region_mask",
            ],
        },
        "canonical_input": {
            "file": "inputs/pair_01_face_quality_input_512.png",
            "size_bytes": input_path.stat().st_size,
            "sha256": sha256(input_path),
            "decoded_bgr_sha256": decoded_bgr_sha256,
            "provenance": provenance,
        },
        "gfpgan": {
            "model": MODEL_SPECS["gfpgan"],
            "shape": list(gfpgan_raw.shape),
            "range": "raw RGB CHW, nominal [-1,1]",
            "min": float(gfpgan_raw.min()),
            "max": float(gfpgan_raw.max()),
            "mean": float(gfpgan_raw.mean()),
            "input_sha256_f32le": gfpgan_input_sha256,
            "sha256_f32le": sha256(out_dir / "gfpgan_output_f32le.bin"),
            "inference_ms": gfpgan_ms,
        },
        "bisenet": {
            "model": MODEL_SPECS["bisenet"],
            "argmax_shape": list(bisenet_argmax.shape),
            "argmax_sha256": sha256(out_dir / "bisenet_argmax_u8.bin"),
            "input_sha256_f32le": bisenet_input_sha256,
            "region_names": REGION_NAMES,
            "region_mask_sha256_f32le": sha256(
                out_dir / "bisenet_region_mask_f32le.bin"
            ),
            "inference_ms": bisenet_ms,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "onnxruntime": ort.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "provider": "CPUExecutionProvider",
        },
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
