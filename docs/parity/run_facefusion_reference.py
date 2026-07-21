#!/usr/bin/env python3
"""Generate Stage B geometry and raw swapper references with FaceFusion 3.7.1.

The checked-out FaceFusion modules own detector preprocessing, recognition,
alignment, embedding conversion and swapper forwarding. Their inference pools are
instrumented with revision-pinned local ONNX Runtime sessions so no download is
attempted and the raw tensor can be captured before masks or paste-back (Stage C).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort


MODEL_SPECS = {
    "detector": (
        "yoloface_8n.onnx",
        "821cdbb1e65fbbabdde7dd0933f754797a343e56fd962729c61ffcefcd135929",
    ),
    "recognizer": (
        "arcface_w600k_r50.onnx",
        "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1",
    ),
    "swapper": (
        "hyperswap_1a_256.onnx",
        "c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add",
    ),
    "fallback_swapper": (
        "inswapper_128_fp16.onnx",
        "c4eccca86ad177586c85c28bf1a64a9d9ed237e283a15818d831f7facfd3f420",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_models(model_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for role, (name, expected) in MODEL_SPECS.items():
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role} model: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
        paths[role] = path
    return paths


def create_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = max(1, min(4, (os_cpu_count() or 1)))
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


class CapturingSession:
    """ORT-compatible proxy that exposes FaceFusion's last raw model outputs."""

    def __init__(self, session: ort.InferenceSession) -> None:
        self.session = session
        self.last_outputs: list[np.ndarray] | None = None

    def get_inputs(self):
        return self.session.get_inputs()

    def run(self, output_names, inputs):
        self.last_outputs = self.session.run(output_names, inputs)
        return self.last_outputs


def os_cpu_count() -> int | None:
    import os

    return os.cpu_count()


def detect_primary_face(
    image_bgr: np.ndarray,
    face_detector_module,
    apply_nms,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    boxes, scores, landmarks = face_detector_module.detect_with_yolo_face(
        image_bgr,
        "640x640",
    )
    if not scores:
        raise RuntimeError("yoloface_8n found no face above score 0.5")
    kept = np.asarray(apply_nms(boxes, scores, 0.5, 0.4)).reshape(-1)
    if kept.size == 0:
        raise RuntimeError("NMS removed every detected face")
    best = max((int(index) for index in kept), key=lambda index: scores[index])
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return scores[best], boxes[best], landmarks[best], elapsed_ms


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def as_list(array: np.ndarray) -> list:
    return np.asarray(array).astype(float).tolist()


def render_raw_output(raw_output: np.ndarray) -> np.ndarray:
    rgb = raw_output.transpose(1, 2, 0) * 0.5 + 0.5
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.rint(rgb[:, :, ::-1] * 255.0).astype(np.uint8)


def render_inswapper_output(raw_output: np.ndarray) -> np.ndarray:
    rgb = np.clip(raw_output.transpose(1, 2, 0), 0.0, 1.0)
    return np.rint(rgb[:, :, ::-1] * 255.0).astype(np.uint8)


def create_face(face_type, score, box, landmarks, embedding, embedding_norm):
    return face_type(
        origin="yoloface_8n",
        bounding_box=box,
        score_set={"detector": score},
        landmark_set={"5": landmarks, "5/68": landmarks},
        angle=0,
        embedding=embedding,
        embedding_norm=embedding_norm,
        age=None,
        gender=None,
        race=None,
    )


def run_pair(
    pair_name: str,
    source_path: Path,
    target_path: Path,
    output_root: Path,
    capturing_sessions: dict[str, CapturingSession],
    face_detector_module,
    face_recognizer_module,
    face_swapper_module,
    state_manager,
    face_type,
    apply_nms,
    warp_face_by_face_landmark_5,
) -> dict[str, object]:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    if source is None or target is None:
        raise RuntimeError(f"Could not decode inputs for {pair_name}")

    pair_dir = output_root / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    source_score, source_box, source_points, source_detect_ms = detect_primary_face(
        source, face_detector_module, apply_nms
    )
    target_score, target_box, target_points, target_detect_ms = detect_primary_face(
        target, face_detector_module, apply_nms
    )

    source_aligned, source_matrix = warp_face_by_face_landmark_5(
        source, source_points, "arcface_112_v2", (112, 112)
    )
    target_aligned, target_matrix = warp_face_by_face_landmark_5(
        target, target_points, "arcface_128", (256, 256)
    )
    target_aligned_128, target_matrix_128 = warp_face_by_face_landmark_5(
        target, target_points, "arcface_128", (128, 128)
    )
    cv2.imwrite(str(pair_dir / "source_aligned_112.png"), source_aligned)
    cv2.imwrite(str(pair_dir / "target_aligned_256.png"), target_aligned)
    cv2.imwrite(str(pair_dir / "target_aligned_128.png"), target_aligned_128)

    started = time.perf_counter()
    embedding, embedding_norm = face_recognizer_module.calculate_face_embedding(
        source,
        source_points,
    )
    recognizer_ms = (time.perf_counter() - started) * 1000.0
    target_embedding, target_embedding_norm = face_recognizer_module.calculate_face_embedding(
        target,
        target_points,
    )
    source_face = create_face(
        face_type,
        source_score,
        source_box,
        source_points,
        embedding,
        embedding_norm,
    )
    target_face = create_face(
        face_type,
        target_score,
        target_box,
        target_points,
        target_embedding,
        target_embedding_norm,
    )

    state_manager.set_item("face_swapper_model", "hyperswap_1a_256")
    prepared_target = face_swapper_module.prepare_crop_frame(target_aligned)
    started = time.perf_counter()
    raw_output = face_swapper_module.forward_swap_face(
        source_face,
        target_face,
        source,
        prepared_target,
    )
    swapper_ms = (time.perf_counter() - started) * 1000.0
    raw_output = np.asarray(raw_output, dtype=np.float32)
    hyperswap_outputs = capturing_sessions["swapper"].last_outputs
    if hyperswap_outputs is None or len(hyperswap_outputs) < 2:
        raise RuntimeError("FaceFusion HyperSwap forward did not expose the raw mask")
    raw_mask = np.asarray(hyperswap_outputs[1][0], dtype=np.float32)

    state_manager.set_item("face_swapper_model", "inswapper_128_fp16")
    fallback_source = face_swapper_module.prepare_source_embedding(source_face)
    fallback_source = face_swapper_module.balance_source_embedding(
        fallback_source,
        target_face.embedding,
    )
    prepared_fallback_target = face_swapper_module.prepare_crop_frame(target_aligned_128)
    started = time.perf_counter()
    fallback_raw_output = face_swapper_module.forward_swap_face(
        source_face,
        target_face,
        source,
        prepared_fallback_target,
    )
    fallback_swapper_ms = (time.perf_counter() - started) * 1000.0
    # forward_swap_face already strips batch dimension and returns CHW.
    fallback_raw_output = np.asarray(fallback_raw_output, dtype=np.float32)
    if fallback_raw_output.ndim != 3:
        # Preserve a clear failure if a future FaceFusion version changes its contract.
        raise RuntimeError(f"Unexpected InSwapper output shape: {fallback_raw_output.shape}")

    embedding.astype(np.float32).tofile(pair_dir / "source_embedding_f32le.bin")
    embedding_norm.tofile(pair_dir / "source_embedding_norm_f32le.bin")
    raw_output.tofile(pair_dir / "raw_output_f32le.bin")
    raw_mask.tofile(pair_dir / "raw_mask_f32le.bin")
    fallback_source.tofile(pair_dir / "inswapper_source_f32le.bin")
    fallback_raw_output.tofile(pair_dir / "inswapper_raw_output_f32le.bin")
    cv2.imwrite(str(pair_dir / "raw_output.png"), render_raw_output(raw_output))
    cv2.imwrite(
        str(pair_dir / "inswapper_raw_output.png"),
        render_inswapper_output(fallback_raw_output),
    )

    metadata = {
        "pair": pair_name,
        "source": {
            "file": source_path.name,
            "sha256": sha256(source_path),
            "detector_score": source_score,
            "bounding_box_xyxy": as_list(source_box),
            "landmarks_5_xy": as_list(source_points),
            "alignment_template": "arcface_112_v2",
            "alignment_size": [112, 112],
            "affine_source_to_crop": as_list(source_matrix),
        },
        "target": {
            "file": target_path.name,
            "sha256": sha256(target_path),
            "detector_score": target_score,
            "bounding_box_xyxy": as_list(target_box),
            "landmarks_5_xy": as_list(target_points),
            "alignment_template": "arcface_128",
            "alignment_size": [256, 256],
            "affine_source_to_crop": as_list(target_matrix),
            "fallback_alignment_size": [128, 128],
            "fallback_affine_source_to_crop": as_list(target_matrix_128),
        },
        "raw_output": {
            "shape": list(raw_output.shape),
            "dtype": str(raw_output.dtype),
            "minimum": float(raw_output.min()),
            "maximum": float(raw_output.max()),
            "mean": float(raw_output.mean()),
            "sha256_f32le": sha256(pair_dir / "raw_output_f32le.bin"),
            "visual_sha256": sha256(pair_dir / "raw_output.png"),
        },
        "fallback_raw_output": {
            "model": "inswapper_128_fp16",
            "embedding_conversion": "embedding @ emap / l2_norm(embedding)",
            "shape": list(fallback_raw_output.shape),
            "dtype": str(fallback_raw_output.dtype),
            "minimum": float(fallback_raw_output.min()),
            "maximum": float(fallback_raw_output.max()),
            "mean": float(fallback_raw_output.mean()),
            "sha256_f32le": sha256(pair_dir / "inswapper_raw_output_f32le.bin"),
            "visual_sha256": sha256(pair_dir / "inswapper_raw_output.png"),
        },
        "timings_ms": {
            "source_detection": source_detect_ms,
            "target_detection": target_detect_ms,
            "recognizer": recognizer_ms,
            "swapper": swapper_ms,
            "fallback_swapper": fallback_swapper_ms,
        },
    }
    save_json(pair_dir / "metadata.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facefusion-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input-root", type=Path, default=Path(__file__).parent / "inputs")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).parent / "reference" / "facefusion-3.7.1",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.facefusion_root.resolve()))
    from facefusion import face_detector, face_recognizer, state_manager
    from facefusion.face_helper import apply_nms, warp_face_by_face_landmark_5
    from facefusion.processors.modules.face_swapper import core as face_swapper
    from facefusion.types import Face

    model_paths = verify_models(args.model_dir)
    sessions = {role: create_session(path) for role, path in model_paths.items()}
    capturing_sessions = {
        "swapper": CapturingSession(sessions["swapper"]),
        "fallback_swapper": CapturingSession(sessions["fallback_swapper"]),
    }
    fallback_model = onnx.load(model_paths["fallback_swapper"])
    fallback_emap = onnx.numpy_helper.to_array(fallback_model.graph.initializer[-1])
    if fallback_emap.shape != (512, 512):
        raise RuntimeError(f"Unexpected InSwapper emap shape: {fallback_emap.shape}")

    # Keep all FaceFusion preprocessing/conversion functions intact, but inject the
    # already checksum-verified sessions to prevent downloads and capture raw output.
    state_manager.init_item("face_detector_score", 0.5)
    state_manager.init_item("face_swapper_weight", 0.5)
    state_manager.init_item("face_swapper_model", "hyperswap_1a_256")
    offline_url = lambda base, name: f"offline://{base}/{name}"
    face_recognizer.resolve_download_url = offline_url
    face_swapper.resolve_download_url = offline_url
    face_detector.forward_with_yolo_face = lambda frame: sessions["detector"].run(
        None,
        {"input": frame},
    )[0]
    face_recognizer.forward = lambda frame: sessions["recognizer"].run(
        None,
        {"input": frame},
    )[0]

    def injected_swapper_pool():
        model = state_manager.get_item("face_swapper_model")
        key = "fallback_swapper" if model == "inswapper_128_fp16" else "swapper"
        return {"face_swapper": capturing_sessions[key]}

    face_swapper.get_inference_pool = injected_swapper_pool
    face_swapper.get_static_model_initializer = lambda _path: fallback_emap
    args.output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    try:
        for index in range(1, 4):
            name = f"pair_{index:02d}"
            results.append(
                run_pair(
                    name,
                    args.input_root / f"{name}_source.png",
                    args.input_root / f"{name}_target.png",
                    args.output_root,
                    capturing_sessions,
                    face_detector,
                    face_recognizer,
                    face_swapper,
                    state_manager,
                    Face,
                    apply_nms,
                    warp_face_by_face_landmark_5,
                )
            )
    finally:
        sessions.clear()

    summary = {
        "reference": {
            "project": "FaceFusion",
            "version": "3.7.1",
            "git_commit": "3f81a8a78454089d720b8f318a12ae1702c4633b",
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "provider": "CPUExecutionProvider",
            "instrumented_functions": [
                "face_detector.detect_with_yolo_face",
                "face_recognizer.calculate_face_embedding",
                "face_helper.warp_face_by_face_landmark_5",
                "face_swapper.prepare_crop_frame",
                "face_swapper.prepare_source_embedding",
                "face_swapper.balance_source_embedding",
                "face_swapper.forward_swap_face",
            ],
        },
        "models": {
            role: {"file": path.name, "sha256": sha256(path), "size": path.stat().st_size}
            for role, path in model_paths.items()
        },
        "pairs": results,
    }
    save_json(args.output_root / "desktop_results.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
