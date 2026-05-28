from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

from .generative_model import MESH_SUFFIXES, default_training_data_dir


NeuralProgress = Callable[[int, str], None]
DEFAULT_NEURAL_MESH_TIMEOUT_SECONDS = int(os.environ.get("MESHMEND_TRAIN_MESH_TIMEOUT", "120"))
NEURAL_EXACT_VOXEL_MAX_FACES = int(os.environ.get("MESHMEND_NEURAL_EXACT_VOXEL_MAX_FACES", "250000"))
NEURAL_SURFACE_SAMPLE_MAX_FACES = int(os.environ.get("MESHMEND_NEURAL_SURFACE_SAMPLE_MAX_FACES", "300000"))


@dataclass(slots=True)
class NeuralTrainingConfig:
    resolution: int = 32
    latent_channels: int = 8
    autoencoder_epochs: int = 20
    diffusion_epochs: int = 40
    batch_size: int = 4
    learning_rate: float = 1e-3
    diffusion_steps: int = 24
    device: str = "auto"


@dataclass(slots=True)
class NeuralTrainingResult:
    checkpoint_path: str
    examples: int
    resolution: int
    message: str
    manifest_path: str | None = None
    checkpoint_size_bytes: int = 0
    data_signature: str = ""


TAG_VOCAB = [
    "knight", "soldier", "marine", "robot", "mech", "tank", "dragon", "demon", "chaos",
    "elf", "elven", "archer", "dwarf", "wizard", "mage", "sword", "shield", "rifle",
    "gun", "armor", "armored", "wing", "wings", "cloak", "beard", "hammer", "miniature",
]


class Neural3DDiffusionModel:
    """PyTorch voxel autoencoder + latent diffusion generator.

    This is the first true neural backend in MeshMend: it learns a compressed 3D
    latent distribution from voxelized meshes, trains a denoising diffusion model
    over that latent space, and decodes newly sampled latents into STL meshes.
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path

    @classmethod
    def train_from_directory(
        cls,
        source_dir: str | Path,
        config: NeuralTrainingConfig | None = None,
        checkpoint_path: str | Path | None = None,
        progress: NeuralProgress | None = None,
    ) -> NeuralTrainingResult:
        torch, nn, functional, data = _torch_modules()
        config = config or NeuralTrainingConfig()
        progress = progress or (lambda percent, message: None)
        source_dir = Path(source_dir).expanduser().resolve()
        checkpoint_path = Path(checkpoint_path) if checkpoint_path else default_neural_checkpoint_path()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if config.resolution % 8 != 0:
            raise ValueError("NeuralTrainingConfig.resolution must be divisible by 8")

        mesh_files = sorted(path for path in source_dir.rglob("*") if path.suffix.lower() in MESH_SUFFIXES)
        if not mesh_files:
            raise ValueError(f"No STL/OBJ/PLY files found under {source_dir}")

        progress(2, f"Voxelizing {len(mesh_files)} meshes at {config.resolution}³")
        voxels: list[np.ndarray] = []
        conditions: list[np.ndarray] = []
        used_meshes: list[dict[str, object]] = []
        skipped_meshes: list[str] = []
        for index, mesh_path in enumerate(mesh_files, start=1):
            progress(2 + int(index / len(mesh_files) * 18), f"Voxelizing {mesh_path.name}")
            voxel, warning = _voxelize_mesh_with_timeout(mesh_path, config.resolution)
            if warning:
                progress(2 + int(index / len(mesh_files) * 18), warning)
                skipped_meshes.append(warning)
            if voxel is not None:
                voxels.append(voxel)
                conditions.append(prompt_condition(mesh_path.stem))
                used_meshes.append(_mesh_training_record(mesh_path, voxel))

        if not voxels:
            raise ValueError("No usable meshes could be voxelized for neural training")

        data_signature = _dataset_signature(used_meshes, config)
        progress(22, f"Prepared {len(voxels)} neural examples | dataset signature {data_signature[:12]}")

        x = torch.tensor(np.stack(voxels), dtype=torch.float32).unsqueeze(1)
        c = torch.tensor(np.stack(conditions), dtype=torch.float32)
        dataset = data.TensorDataset(x, c)
        loader = data.DataLoader(dataset, batch_size=max(1, config.batch_size), shuffle=True)
        device = _resolve_device(torch, config.device)
        latent_shape = (config.latent_channels, config.resolution // 8, config.resolution // 8, config.resolution // 8)

        autoencoder = VoxelAutoencoder(nn, config.latent_channels).to(device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config.learning_rate)
        progress(24, "Training 3D voxel autoencoder")
        autoencoder_loss_history: list[float] = []
        for epoch in range(config.autoencoder_epochs):
            losses = []
            for batch, _cond in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = autoencoder(batch)
                loss = functional.binary_cross_entropy_with_logits(logits, batch)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            mean_loss = float(np.mean(losses)) if losses else 0.0
            autoencoder_loss_history.append(mean_loss)
            progress(24 + int((epoch + 1) / max(1, config.autoencoder_epochs) * 30), f"Autoencoder epoch {epoch + 1}/{config.autoencoder_epochs} loss={mean_loss:.4f}")

        progress(56, "Encoding training meshes into latent space")
        latents: list[object] = []
        conds: list[object] = []
        autoencoder.eval()
        with torch.no_grad():
            for batch, cond in data.DataLoader(dataset, batch_size=max(1, config.batch_size)):
                latents.append(autoencoder.encode(batch.to(device)).detach().cpu())
                conds.append(cond)
        latent_tensor = torch.cat(latents, dim=0)
        cond_tensor = torch.cat(conds, dim=0)
        latent_dataset = data.TensorDataset(latent_tensor, cond_tensor)
        latent_loader = data.DataLoader(latent_dataset, batch_size=max(1, config.batch_size), shuffle=True)

        denoiser = LatentDenoiser(nn, config.latent_channels, len(TAG_VOCAB)).to(device)
        diffusion_optimizer = torch.optim.Adam(denoiser.parameters(), lr=config.learning_rate)
        progress(60, "Training latent denoising diffusion model")
        diffusion_loss_history: list[float] = []
        for epoch in range(config.diffusion_epochs):
            losses = []
            for latent_batch, cond_batch in latent_loader:
                latent_batch = latent_batch.to(device)
                cond_batch = cond_batch.to(device)
                noise = torch.randn_like(latent_batch)
                t = torch.rand((latent_batch.shape[0], 1), device=device)
                noisy = latent_batch + noise * t.view(-1, 1, 1, 1, 1)
                diffusion_optimizer.zero_grad(set_to_none=True)
                predicted = denoiser(noisy, t, cond_batch)
                loss = functional.mse_loss(predicted, noise)
                loss.backward()
                diffusion_optimizer.step()
                losses.append(float(loss.detach().cpu()))
            mean_loss = float(np.mean(losses)) if losses else 0.0
            diffusion_loss_history.append(mean_loss)
            progress(60 + int((epoch + 1) / max(1, config.diffusion_epochs) * 34), f"Diffusion epoch {epoch + 1}/{config.diffusion_epochs} loss={mean_loss:.4f}")

        manifest = {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(source_dir),
            "checkpoint_path": str(checkpoint_path),
            "config": asdict(config),
            "mesh_files_found": len(mesh_files),
            "examples": len(voxels),
            "skipped": skipped_meshes,
            "used_meshes": used_meshes,
            "data_signature": data_signature,
            "autoencoder_loss_history": autoencoder_loss_history,
            "diffusion_loss_history": diffusion_loss_history,
            "autoencoder_initial_loss": autoencoder_loss_history[0] if autoencoder_loss_history else None,
            "autoencoder_final_loss": autoencoder_loss_history[-1] if autoencoder_loss_history else None,
            "diffusion_initial_loss": diffusion_loss_history[0] if diffusion_loss_history else None,
            "diffusion_final_loss": diffusion_loss_history[-1] if diffusion_loss_history else None,
        }

        payload = {
            "config": asdict(config),
            "tag_vocab": TAG_VOCAB,
            "latent_shape": latent_shape,
            "examples": len(voxels),
            "training_manifest": manifest,
            "autoencoder": autoencoder.state_dict(),
            "denoiser": denoiser.state_dict(),
        }
        torch.save(payload, checkpoint_path)
        latest = checkpoint_path.parent / "latest_neural_model.pt"
        torch.save(payload, latest)
        manifest["checkpoint_path"] = str(latest)
        manifest["checkpoint_size_bytes"] = latest.stat().st_size
        manifest_path = latest.with_name(f"{latest.stem}_manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        specific_manifest_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_manifest.json")
        if specific_manifest_path != manifest_path:
            specific_manifest_path.write_text(json.dumps({**manifest, "checkpoint_path": str(checkpoint_path)}, indent=2), encoding="utf-8")
        progress(100, f"Neural checkpoint saved: {latest.name} | examples={len(voxels)} | signature={data_signature[:12]}")
        return NeuralTrainingResult(
            str(latest),
            len(voxels),
            config.resolution,
            "Neural 3D diffusion model trained successfully.",
            str(manifest_path),
            latest.stat().st_size,
            data_signature,
        )

    @classmethod
    def load_latest(cls) -> Neural3DDiffusionModel | None:
        path = default_neural_checkpoint_path().parent / "latest_neural_model.pt"
        return cls(path) if path.exists() else None

    def checkpoint_config(self) -> NeuralTrainingConfig:
        torch, _nn, _functional, _data = _torch_modules()
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        return NeuralTrainingConfig(**payload["config"])

    def training_examples(self) -> int:
        torch, _nn, _functional, _data = _torch_modules()
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        return int(payload.get("examples", 0))

    def generate(self, prompt: str, steps: int | None = None) -> trimesh.Trimesh | None:
        torch, nn, _functional, _data = _torch_modules()
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        config = NeuralTrainingConfig(**payload["config"])
        steps = steps or config.diffusion_steps
        device = _resolve_device(torch, config.device)
        autoencoder = VoxelAutoencoder(nn, config.latent_channels).to(device)
        denoiser = LatentDenoiser(nn, config.latent_channels, len(TAG_VOCAB)).to(device)
        autoencoder.load_state_dict(payload["autoencoder"])
        denoiser.load_state_dict(payload["denoiser"])
        autoencoder.eval()
        denoiser.eval()

        cond = torch.tensor(prompt_condition(prompt), dtype=torch.float32, device=device).unsqueeze(0)
        latent_shape = tuple(payload["latent_shape"])
        latent = torch.randn((1, *latent_shape), device=device)
        with torch.no_grad():
            for index in reversed(range(steps)):
                t_value = (index + 1) / max(1, steps)
                t = torch.full((1, 1), t_value, device=device)
                predicted_noise = denoiser(latent, t, cond)
                latent = latent - predicted_noise * (1.0 / max(1, steps))
            logits = autoencoder.decode(latent)
            occupancy = (torch.sigmoid(logits)[0, 0].detach().cpu().numpy() > 0.42)
        return occupancy_to_mesh(occupancy)


def default_neural_checkpoint_path() -> Path:
    return default_training_data_dir() / "checkpoints" / "meshmend_neural_3d_diffusion.pt"


def prompt_condition(prompt: str) -> np.ndarray:
    text = prompt.lower()
    return np.array([1.0 if tag in text else 0.0 for tag in TAG_VOCAB], dtype=np.float32)


def _mesh_training_record(mesh_path: Path, voxel: np.ndarray) -> dict[str, object]:
    stat = mesh_path.stat()
    filled = int(np.count_nonzero(voxel))
    return {
        "path": str(mesh_path),
        "name": mesh_path.name,
        "stem": mesh_path.stem,
        "size_bytes": int(stat.st_size),
        "modified_ns": int(stat.st_mtime_ns),
        "voxel_filled": filled,
        "voxel_fill_ratio": round(float(filled) / float(voxel.size), 6),
    }


def _dataset_signature(mesh_records: list[dict[str, object]], config: NeuralTrainingConfig) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(config), sort_keys=True).encode("utf-8"))
    for record in mesh_records:
        digest.update(json.dumps(record, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def mesh_to_occupancy(mesh: trimesh.Trimesh, resolution: int) -> np.ndarray:
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)
    max_extent = float(np.max(mesh.extents))
    if max_extent <= 1e-9:
        raise ValueError("Cannot voxelize zero-size mesh")
    mesh.apply_scale(0.92 / max_extent)
    if len(mesh.faces) > NEURAL_EXACT_VOXEL_MAX_FACES:
        return _surface_occupancy(mesh, resolution)
    pitch = 1.0 / float(resolution)
    try:
        voxels = mesh.voxelized(pitch).fill()
        matrix = np.asarray(voxels.matrix, dtype=bool)
        occupancy = _center_crop_or_pad(matrix, resolution)
    except Exception:
        occupancy = _surface_occupancy(mesh, resolution).astype(bool)
    if not np.any(occupancy):
        occupancy = _surface_occupancy(mesh, resolution).astype(bool)
    return occupancy.astype(np.float32)


def _surface_occupancy(mesh: trimesh.Trimesh, resolution: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0:
        raise ValueError("Cannot voxelize mesh with no vertices")

    points = [vertices]
    if len(faces) > 0:
        if len(faces) > NEURAL_SURFACE_SAMPLE_MAX_FACES:
            stride = max(1, len(faces) // NEURAL_SURFACE_SAMPLE_MAX_FACES)
            faces_for_sampling = faces[::stride]
        else:
            faces_for_sampling = faces
        triangles = vertices[faces_for_sampling]
        points.extend(
            [
                triangles.mean(axis=1),
                (triangles[:, 0] + triangles[:, 1]) * 0.5,
                (triangles[:, 1] + triangles[:, 2]) * 0.5,
                (triangles[:, 2] + triangles[:, 0]) * 0.5,
            ]
        )

    cloud = np.concatenate(points, axis=0)
    grid = np.clip(((cloud + 0.5) * (resolution - 1)).round().astype(np.int64), 0, resolution - 1)
    occupancy = np.zeros((resolution, resolution, resolution), dtype=bool)
    occupancy[grid[:, 0], grid[:, 1], grid[:, 2]] = True
    return _dilate_occupancy(occupancy, iterations=1)


def _dilate_occupancy(occupancy: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = np.asarray(occupancy, dtype=bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        expanded = result.copy()
        expanded |= padded[0:-2, 1:-1, 1:-1]
        expanded |= padded[2:, 1:-1, 1:-1]
        expanded |= padded[1:-1, 0:-2, 1:-1]
        expanded |= padded[1:-1, 2:, 1:-1]
        expanded |= padded[1:-1, 1:-1, 0:-2]
        expanded |= padded[1:-1, 1:-1, 2:]
        result = expanded
    return result


def occupancy_to_mesh(occupancy: np.ndarray) -> trimesh.Trimesh | None:
    occupancy = np.asarray(occupancy, dtype=bool)
    if not np.any(occupancy):
        return None
    try:
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(occupancy, pitch=1.0)
    except Exception:
        return None
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh


def _center_crop_or_pad(matrix: np.ndarray, resolution: int) -> np.ndarray:
    result = np.zeros((resolution, resolution, resolution), dtype=bool)
    src_slices = []
    dst_slices = []
    for axis_size in matrix.shape[:3]:
        if axis_size >= resolution:
            src_start = (axis_size - resolution) // 2
            src_slices.append(slice(src_start, src_start + resolution))
            dst_slices.append(slice(0, resolution))
        else:
            dst_start = (resolution - axis_size) // 2
            src_slices.append(slice(0, axis_size))
            dst_slices.append(slice(dst_start, dst_start + axis_size))
    result[tuple(dst_slices)] = matrix[tuple(src_slices)]
    return result


def _resolve_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _torch_modules():
    try:
        import torch
        from torch import nn
        from torch.nn import functional
        from torch.utils import data
    except Exception as exc:
        raise RuntimeError("Neural 3D diffusion requires PyTorch. Install with: python -m pip install -e .[neural]") from exc
    return torch, nn, functional, data


def _voxelize_mesh_with_timeout(
    mesh_path: Path,
    resolution: int,
    timeout_seconds: int = DEFAULT_NEURAL_MESH_TIMEOUT_SECONDS,
) -> tuple[np.ndarray | None, str | None]:
    context = mp.get_context("spawn")
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(target=_voxelize_mesh_worker, args=(str(mesh_path), resolution, child_conn))
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
            return None, f"Skipped {mesh_path.name}: neural voxelization returned no result"
        else:
            process.terminate()
            process.join(5)
            return None, f"Skipped {mesh_path.name}: neural voxelization timed out after {timeout_seconds}s"
    except Exception as exc:
        process.terminate()
        process.join(5)
        return None, f"Skipped {mesh_path.name}: neural voxelization failed ({exc})"
    finally:
        parent_conn.close()

    if payload.get("ok"):
        return payload["voxel"], None
    return None, f"Skipped {mesh_path.name}: {payload.get('error', 'unknown voxelization error')}"


def _voxelize_mesh_worker(mesh_path: str, resolution: int, conn) -> None:
    try:
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.geometry.values())
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            conn.send({"ok": False, "error": "empty or unsupported mesh"})
            return
        voxel = mesh_to_occupancy(mesh, resolution)
        if not np.any(voxel):
            conn.send({"ok": False, "error": "empty voxel grid"})
            return
        conn.send({"ok": True, "voxel": voxel})
    except Exception as exc:
        conn.send({"ok": False, "error": str(exc)})
    finally:
        conn.close()


class VoxelAutoencoder:
    def __new__(cls, nn, latent_channels: int):
        class _VoxelAutoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv3d(1, 16, 4, 2, 1), nn.ReLU(inplace=True),
                    nn.Conv3d(16, 32, 4, 2, 1), nn.ReLU(inplace=True),
                    nn.Conv3d(32, latent_channels, 4, 2, 1),
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose3d(latent_channels, 32, 4, 2, 1), nn.ReLU(inplace=True),
                    nn.ConvTranspose3d(32, 16, 4, 2, 1), nn.ReLU(inplace=True),
                    nn.ConvTranspose3d(16, 1, 4, 2, 1),
                )

            def encode(self, x):
                return self.encoder(x)

            def decode(self, z):
                return self.decoder(z)

            def forward(self, x):
                return self.decode(self.encode(x))

        return _VoxelAutoencoder()


class LatentDenoiser:
    def __new__(cls, nn, latent_channels: int, condition_dim: int):
        class _LatentDenoiser(nn.Module):
            def __init__(self):
                super().__init__()
                self.condition = nn.Sequential(nn.Linear(condition_dim + 1, latent_channels), nn.SiLU())
                self.net = nn.Sequential(
                    nn.Conv3d(latent_channels, 32, 3, padding=1), nn.SiLU(),
                    nn.Conv3d(32, 32, 3, padding=1), nn.SiLU(),
                    nn.Conv3d(32, latent_channels, 3, padding=1),
                )

            def forward(self, latent, t, cond):
                import torch

                embedding = self.condition(torch.cat([cond, t], dim=1)).view(latent.shape[0], latent.shape[1], 1, 1, 1)
                return self.net(latent + embedding)

        return _LatentDenoiser()
