from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import shutil
from typing import Callable

import numpy as np
import trimesh

from .repair import _component_count


TrainingProgress = Callable[[int, str], None]

MESH_SUFFIXES = {".stl", ".obj", ".ply"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_MESH_TIMEOUT_SECONDS = int(os.environ.get("MESHMEND_TRAIN_MESH_TIMEOUT", "120"))
LOCAL_VOXEL_SUMMARY_MAX_FACES = int(os.environ.get("MESHMEND_LOCAL_VOXEL_SUMMARY_MAX_FACES", "250000"))
LOCAL_COMPONENT_CHECK_MAX_FACES = int(os.environ.get("MESHMEND_LOCAL_COMPONENT_CHECK_MAX_FACES", "250000"))


@dataclass(slots=True)
class TrainingExample:
    mesh_path: str
    source_path: str
    image_path: str | None
    caption: str
    tags: list[str]
    vertices: int
    faces: int
    extents: list[float]
    watertight: bool
    voxel_resolution: int = 32
    voxel_pitch: float = 0.0
    voxel_filled: int = 0
    voxel_fill_ratio: float = 0.0
    quality_warnings: list[str] | None = None


@dataclass(slots=True)
class TrainingResult:
    checkpoint_path: str
    examples: int
    images: int
    message: str


class Local3DGenerativeModel:
    """Small local trainable 3D generator foundation.

    This is intentionally lightweight: it learns a curated exemplar/prototype
    library from STL/OBJ/PLY files and uses prompt tag matching during inference.
    It gives MeshMend a real train/save/load loop now, while leaving room to
    replace the checkpoint format with a PyTorch 3D diffusion model later.
    """

    version = 1

    def __init__(self, examples: list[TrainingExample], checkpoint_path: Path):
        self.examples = examples
        self.checkpoint_path = checkpoint_path

    @classmethod
    def train_from_directory(
        cls,
        source_dir: str | Path,
        checkpoint_path: str | Path | None = None,
        progress: TrainingProgress | None = None,
    ) -> TrainingResult:
        source_dir = Path(source_dir).expanduser().resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Training directory does not exist: {source_dir}")

        checkpoint_path = Path(checkpoint_path) if checkpoint_path else default_checkpoint_path()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        processed_dir = checkpoint_path.parent.parent / "processed_meshes"
        processed_dir.mkdir(parents=True, exist_ok=True)
        progress = progress or (lambda percent, message: None)

        mesh_files = sorted(path for path in source_dir.rglob("*") if path.suffix.lower() in MESH_SUFFIXES)
        image_roots = [source_dir]
        sibling_images = source_dir.parent / "raw_images"
        if sibling_images.exists() and sibling_images not in image_roots:
            image_roots.append(sibling_images)
        image_files = sorted(
            path
            for image_root in image_roots
            for path in image_root.rglob("*")
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not mesh_files:
            raise ValueError(f"No STL/OBJ/PLY files found under {source_dir}")

        image_by_stem = {path.stem.lower(): path for path in image_files}
        examples: list[TrainingExample] = []
        total = len(mesh_files)
        progress(2, f"Found {total} mesh files and {len(image_files)} image files")

        for index, mesh_path in enumerate(mesh_files, start=1):
            percent = 5 + int((index - 1) / total * 82)
            progress(percent, f"Processing {mesh_path.name}")
            output_mesh = processed_dir / f"{mesh_path.stem}_trained.stl"
            image_path = image_by_stem.get(mesh_path.stem.lower())
            example, warning = _process_training_mesh_with_timeout(mesh_path, output_mesh, image_path)
            if warning:
                progress(percent, warning)
            if example is not None:
                examples.append(example)

        if not examples:
            raise ValueError("Training did not produce any usable mesh examples")

        progress(92, "Saving local 3D generative checkpoint")
        payload = {
            "version": cls.version,
            "source_dir": str(source_dir),
            "examples": [asdict(example) for example in examples],
        }
        checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        latest_path = checkpoint_path.parent / "latest_model.json"
        if latest_path != checkpoint_path:
            shutil.copyfile(checkpoint_path, latest_path)
        progress(100, f"Training complete: {len(examples)} examples")
        return TrainingResult(
            checkpoint_path=str(checkpoint_path),
            examples=len(examples),
            images=len(image_files),
            message="Local 3D generative checkpoint trained successfully.",
        )

    @classmethod
    def load(cls, checkpoint_path: str | Path) -> Local3DGenerativeModel:
        checkpoint_path = Path(checkpoint_path)
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        examples = [TrainingExample(**_upgrade_example_payload(example)) for example in payload.get("examples", [])]
        return cls(examples=examples, checkpoint_path=checkpoint_path)

    @classmethod
    def load_latest(cls) -> Local3DGenerativeModel | None:
        path = default_checkpoint_path().parent / "latest_model.json"
        if not path.exists():
            return None
        model = cls.load(path)
        return model if model.examples else None

    def generate(self, prompt: str, max_faces: int | None = None) -> trimesh.Trimesh | None:
        if not self.examples:
            return None
        prompt_tags = set(_tags_from_name(prompt))
        candidates = [example for example in self.examples if max_faces is None or example.faces <= max_faces]
        if not candidates:
            return None
        best = max(candidates, key=lambda example: _example_score(example, prompt_tags))
        mesh_path = Path(best.mesh_path)
        if not mesh_path.exists():
            return None
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.geometry.values())
        if not isinstance(mesh, trimesh.Trimesh):
            return None
        return mesh.copy()


def default_training_data_dir() -> Path:
    root = Path(__file__).resolve().parent / "training_data"
    for child in (root / "raw_stl", root / "raw_images", root / "processed_meshes", root / "checkpoints"):
        child.mkdir(parents=True, exist_ok=True)
    return root


def default_checkpoint_path() -> Path:
    return default_training_data_dir() / "checkpoints" / "meshmend_local_3d_model.json"


def _tags_from_name(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    words = [word for word in cleaned.split() if len(word) > 2]
    known = {
        "knight", "soldier", "marine", "robot", "mech", "tank", "dragon", "demon", "chaos",
        "elf", "elven", "archer", "dwarf", "wizard", "mage", "sword", "shield", "rifle",
        "gun", "armor", "armored", "wing", "wings", "cloak", "beard", "hammer", "miniature",
        "orc", "ork", "axe", "head", "torso", "body", "arm", "arms", "leg", "legs", "hands",
        "halberd", "falcion", "drone",
    }
    tags = [word for word in words if word in known]
    return sorted(set(tags or words[:8] or ["model"]))


def _caption_from_tags(tags: list[str], fallback: str) -> str:
    if tags:
        return "tabletop miniature: " + ", ".join(tags)
    return fallback.replace("_", " ").replace("-", " ")


def _example_score(example: TrainingExample, prompt_tags: set[str]) -> int:
    example_tags = set(example.tags)
    overlap = len(example_tags & prompt_tags)
    caption_overlap = sum(1 for tag in prompt_tags if tag in example.caption.lower())
    quality_penalty = len(example.quality_warnings or []) * 2
    return (overlap * 10) + caption_overlap + min(example.faces // 1000, 5) - quality_penalty


def _voxel_summary(mesh: trimesh.Trimesh, resolution: int = 32) -> dict[str, int | float | bool]:
    extents = np.asarray(mesh.extents, dtype=float)
    max_extent = float(np.max(extents)) if len(extents) else 0.0
    if max_extent <= 1e-9:
        return {"resolution": resolution, "pitch": 0.0, "filled": 0, "fill_ratio": 0.0, "skipped": False}

    pitch = max_extent / float(resolution)
    if len(mesh.faces) > LOCAL_VOXEL_SUMMARY_MAX_FACES:
        return {
            "resolution": resolution,
            "pitch": round(float(pitch), 6),
            "filled": 0,
            "fill_ratio": 0.0,
            "skipped": True,
        }
    try:
        voxels = mesh.voxelized(pitch)
        filled = int(len(voxels.points))
    except Exception:
        filled = 0
    return {
        "resolution": resolution,
        "pitch": round(float(pitch), 6),
        "filled": filled,
        "fill_ratio": round(float(filled) / float(resolution**3), 6),
        "skipped": False,
    }


def _quality_warnings(mesh: trimesh.Trimesh, voxel_summary: dict[str, int | float | bool]) -> list[str]:
    warnings: list[str] = []
    extents = np.asarray(mesh.extents, dtype=float)
    max_extent = float(np.max(extents)) if len(extents) else 0.0
    min_extent = float(np.min(extents)) if len(extents) else 0.0
    if max_extent <= 1e-9:
        warnings.append("empty_or_zero_size")
    elif min_extent / max_extent < 0.04:
        warnings.append("very_flat_mesh")
    try:
        if not mesh.is_watertight:
            warnings.append("not_watertight_training_source")
    except Exception:
        warnings.append("watertight_check_failed")
    try:
        if len(mesh.faces) > LOCAL_COMPONENT_CHECK_MAX_FACES:
            warnings.append("component_check_skipped_large_mesh")
        else:
            components = _component_count(mesh)
            if components > 8:
                warnings.append(f"many_components:{components}")
    except Exception:
        warnings.append("component_check_failed")
    fill_ratio = float(voxel_summary.get("fill_ratio", 0.0))
    if bool(voxel_summary.get("skipped", False)):
        warnings.append("voxel_summary_skipped_large_mesh")
    elif fill_ratio <= 0.0:
        warnings.append("voxelization_failed")
    elif fill_ratio > 0.55:
        warnings.append("voxel_grid_too_solid")
    return warnings


def _upgrade_example_payload(example: dict) -> dict:
    upgraded = dict(example)
    upgraded.setdefault("voxel_resolution", 32)
    upgraded.setdefault("voxel_pitch", 0.0)
    upgraded.setdefault("voxel_filled", 0)
    upgraded.setdefault("voxel_fill_ratio", 0.0)
    upgraded.setdefault("quality_warnings", [])
    return upgraded


def _process_training_mesh_with_timeout(
    mesh_path: Path,
    output_mesh: Path,
    image_path: Path | None,
    timeout_seconds: int = DEFAULT_MESH_TIMEOUT_SECONDS,
) -> tuple[TrainingExample | None, str | None]:
    context = mp.get_context("spawn")
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=_training_mesh_worker,
        args=(str(mesh_path), str(output_mesh), str(image_path) if image_path else None, child_conn),
    )
    process.start()
    child_conn.close()
    try:
        ready = wait([parent_conn, process.sentinel], timeout_seconds)
        if parent_conn in ready:
            payload = parent_conn.recv()
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
        elif process.sentinel in ready:
            process.join(5)
            return None, f"Skipped {mesh_path.name}: preprocessing returned no result"
        else:
            process.terminate()
            process.join(5)
            return None, f"Skipped {mesh_path.name}: preprocessing timed out after {timeout_seconds}s"
    except Exception as exc:
        process.terminate()
        process.join(5)
        return None, f"Skipped {mesh_path.name}: preprocessing failed ({exc})"
    finally:
        parent_conn.close()

    if payload.get("ok"):
        return TrainingExample(**payload["example"]), None
    return None, f"Skipped {mesh_path.name}: {payload.get('error', 'unknown preprocessing error')}"


def _training_mesh_worker(mesh_path: str, output_mesh: str, image_path: str | None, conn) -> None:
    try:
        mesh_path_obj = Path(mesh_path)
        mesh = _load_training_mesh_fast(mesh_path_obj)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            conn.send({"ok": False, "error": "empty or unsupported training mesh"})
            return
        voxel_summary = _voxel_summary(mesh)
        warnings = _quality_warnings(mesh, voxel_summary)
        tags = _tags_from_name(mesh_path_obj.stem)
        caption = _caption_from_tags(tags, mesh_path_obj.stem)
        conn.send(
            {
                "ok": True,
                "example": asdict(
                    TrainingExample(
                        mesh_path=str(mesh_path_obj),
                        source_path=str(mesh_path_obj),
                        image_path=image_path,
                        caption=caption,
                        tags=tags,
                        vertices=len(mesh.vertices),
                        faces=len(mesh.faces),
                        extents=np.asarray(mesh.extents, dtype=float).round(5).tolist(),
                        watertight=bool(mesh.is_watertight),
                        voxel_resolution=voxel_summary["resolution"],
                        voxel_pitch=voxel_summary["pitch"],
                        voxel_filled=voxel_summary["filled"],
                        voxel_fill_ratio=voxel_summary["fill_ratio"],
                        quality_warnings=warnings,
                    )
                ),
            }
        )
    except Exception as exc:
        conn.send({"ok": False, "error": str(exc)})
    finally:
        conn.close()


def _load_training_mesh_fast(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("scene contains no mesh geometry")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("unsupported mesh format")
    loaded.remove_unreferenced_vertices()
    return loaded
