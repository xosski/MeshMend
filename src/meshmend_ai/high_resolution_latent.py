from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import re
from typing import Callable

import numpy as np
import trimesh

from .generative_model import Local3DGenerativeModel, MESH_SUFFIXES, TrainingExample, default_training_data_dir


LatentProgress = Callable[[int, str], None]


ROLE_KEYWORDS = {
    "head": {"head", "helmet", "helm", "mask", "face", "skull"},
    "torso": {"torso", "body", "chest", "cloak"},
    "legs": {"leg", "legs", "greave", "feet", "boot", "boots"},
    "left_arm": {"left", "arm", "arms", "hand", "hands"},
    "right_arm": {"right", "arm", "arms", "hand", "hands"},
    "weapon": {"weapon", "rifle", "gun", "bolter", "flamer", "launcher", "sword", "axe", "hammer", "mace", "bow"},
    "shoulder": {"shoulder", "pad", "pauldron"},
    "backpack": {"backpack", "pack", "powerpack", "jump"},
    "full": {"miniature", "warlord", "warboss", "lieutenant", "assassin", "bearkin", "juggernaut", "eradicator"},
}


STYLE_ALIASES = {
    "marine": {"marine", "prime", "infantry", "power", "armor", "armour"},
    "orc": {"orc", "ork", "gob", "gobby", "warboss", "warlord", "guk"},
    "chaos": {"chaos", "spike", "spikey", "reaver", "doom", "claw", "maw"},
    "robot": {"robot", "mech", "walker", "machine", "tank", "dreadnaught"},
    "fantasy": {"knight", "wizard", "mage", "demon", "dragon", "beast", "bear"},
}

COHERENT_FAMILY_BONUS = 55.0
SIDE_MATCH_BONUS = 35.0
SIDE_MISMATCH_PENALTY = 45.0
CHARACTER_TAGS = {"miniature", "wargaming", "soldier", "marine", "warrior", "knight", "orc", "ork", "warboss", "wizard", "robot", "mech", "undead"}


@dataclass(slots=True)
class MeshLatentAsset:
    path: str
    name: str
    tags: list[str]
    roles: list[str]
    vertices: int
    faces: int
    extents: list[float]
    watertight: bool
    file_size_bytes: int
    latent: list[float]


@dataclass(slots=True)
class MeshLatentTrainingResult:
    checkpoint_path: str
    assets: int
    message: str


class LocalMeshLatentGenerator:
    """High-resolution local mesh/latent generator.

    This replaces the old voxel-diffusion bottleneck for creation. Instead of
    compressing miniatures into a 32-96³ occupancy grid, it indexes real STL/OBJ
    assets as semantic high-resolution latent parts and composes new printable
    sculpts from prompt-matched head/torso/limb/weapon/gear geometry.
    """

    version = 1

    def __init__(self, assets: list[MeshLatentAsset], checkpoint_path: Path):
        self.assets = assets
        self.checkpoint_path = Path(checkpoint_path)

    @classmethod
    def train_from_directory(
        cls,
        source_dir: str | Path,
        checkpoint_path: str | Path | None = None,
        progress: LatentProgress | None = None,
    ) -> MeshLatentTrainingResult:
        source_dir = Path(source_dir).expanduser().resolve()
        checkpoint_path = Path(checkpoint_path) if checkpoint_path else default_mesh_latent_checkpoint_path()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        progress = progress or (lambda _percent, _message: None)

        mesh_files = sorted(path for path in source_dir.rglob("*") if path.suffix.lower() in MESH_SUFFIXES)
        if not mesh_files:
            raise ValueError(f"No STL/OBJ/PLY files found under {source_dir}")

        assets: list[MeshLatentAsset] = []
        for index, mesh_path in enumerate(mesh_files, start=1):
            progress(5 + int(index / max(1, len(mesh_files)) * 85), f"Indexing high-res mesh latent {mesh_path.name}")
            asset = _asset_from_mesh_path(mesh_path)
            if asset is not None:
                assets.append(asset)
        if not assets:
            raise ValueError("No usable high-resolution mesh latent assets could be indexed")

        payload = {
            "version": cls.version,
            "source_dir": str(source_dir),
            "assets": [asdict(asset) for asset in assets],
        }
        checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        latest = checkpoint_path.parent / "latest_mesh_latent_index.json"
        latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        progress(100, f"High-resolution mesh latent index saved: {len(assets)} assets")
        return MeshLatentTrainingResult(str(latest), len(assets), "High-resolution mesh latent index trained successfully.")

    @classmethod
    def load_latest(cls) -> LocalMeshLatentGenerator | None:
        path = default_mesh_latent_checkpoint_path().parent / "latest_mesh_latent_index.json"
        raw_dir = default_training_data_dir() / "raw_stl"
        if not path.exists():
            model = Local3DGenerativeModel.load_latest()
            assets: list[MeshLatentAsset] = []
            if model is not None and model.examples:
                assets.extend(asset for example in model.examples if (asset := _asset_from_training_example(example)) is not None)
            # Downloadable builds should work immediately with dropped-in STLs.
            # If there is no trained metadata yet, create a lightweight filename
            # index instead of loading every large mesh during the first request.
            assets.extend(asset for mesh_path in sorted(raw_dir.rglob("*")) if mesh_path.suffix.lower() in MESH_SUFFIXES if (asset := _asset_from_file_metadata(mesh_path)) is not None)
            assets = _dedupe_assets(assets)
            if assets:
                return cls(assets, path)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        assets = [MeshLatentAsset(**asset) for asset in payload.get("assets", [])]
        known_paths = {asset.path for asset in assets}
        assets.extend(
            asset
            for mesh_path in sorted(raw_dir.rglob("*"))
            if mesh_path.suffix.lower() in MESH_SUFFIXES
            if str(mesh_path) not in known_paths
            if (asset := _asset_from_file_metadata(mesh_path)) is not None
        )
        assets = _dedupe_assets(assets)
        return cls(assets, path) if assets else None

    def generate(self, prompt: str, scale_mm: float = 32.0, max_asset_faces: int = 220_000) -> trimesh.Trimesh | None:
        if not self.assets:
            return None
        prompt_tags = _tags_from_text(prompt)
        selected = self._select_assets(prompt_tags, max_asset_faces=max_asset_faces)
        if not selected:
            return None
        meshes: list[trimesh.Trimesh] = [self._base(scale_mm, prompt_tags)]
        transforms = _role_transforms(scale_mm)
        for role, asset in selected.items():
            mesh = _load_asset_mesh(asset.path, target_faces=max_asset_faces)
            if mesh is None:
                continue
            mesh = _orient_role_mesh(mesh, role, asset)
            center, target_size, axis = transforms.get(role, transforms["torso"])
            mesh = _fit_mesh(mesh, center=center, target_size=target_size, axis=axis)
            if role == "shoulder":
                mirror = mesh.copy()
                mirror.apply_scale([-1.0, 1.0, 1.0])
                mirror.invert()
                mesh = trimesh.util.concatenate([mesh, mirror])
            meshes.append(mesh)

        if len(meshes) <= 1:
            return None
        combined = trimesh.util.concatenate(meshes)
        combined.merge_vertices()
        combined.remove_unreferenced_vertices()
        try:
            combined.fix_normals()
        except Exception:
            pass
        return combined

    def _select_assets(self, prompt_tags: set[str], max_asset_faces: int) -> dict[str, MeshLatentAsset]:
        roles = _requested_roles(prompt_tags)
        selected: dict[str, MeshLatentAsset] = {}
        primary_family = self._choose_primary_family(prompt_tags, roles, max_asset_faces)
        for role in roles:
            candidates = [asset for asset in self.assets if role in asset.roles]
            if not candidates and role in {"left_arm", "right_arm"}:
                candidates = [asset for asset in self.assets if "left_arm" in asset.roles or "right_arm" in asset.roles]
            if not candidates:
                continue
            requested_non_marine_styles = prompt_tags & {"orc", "ork", "robot", "mech", "undead", "wizard", "mage", "knight", "demon", "dragon"}
            if requested_non_marine_styles:
                style_candidates = [asset for asset in candidates if set(asset.tags) & requested_non_marine_styles]
                if not style_candidates:
                    continue
                candidates = style_candidates
            preferred = [asset for asset in candidates if asset.faces <= max_asset_faces]
            if preferred:
                candidates = preferred
            if role == "weapon":
                weapon_category = _requested_weapon_category(prompt_tags)
                if weapon_category is not None:
                    category_candidates = [asset for asset in candidates if _asset_weapon_category(asset) == weapon_category]
                    if not category_candidates:
                        continue
                    candidates = category_candidates
            if len(roles) == 1:
                meaningful = [asset for asset in candidates if _has_meaningful_prompt_overlap(asset, prompt_tags)]
                if not meaningful:
                    continue
                candidates = meaningful
            if role in {"left_arm", "right_arm"}:
                side = "left" if role == "left_arm" else "right"
                side_candidates = [asset for asset in candidates if _asset_side(asset) in {None, side}]
                if side_candidates:
                    candidates = side_candidates
            selected[role] = max(candidates, key=lambda asset: _asset_score(asset, prompt_tags, role, primary_family=primary_family))
        if len(roles) > 1 and not any(role in selected for role in ("torso", "legs", "head")):
            full_candidates = [asset for asset in self.assets if "full" in asset.roles and _has_meaningful_prompt_overlap(asset, prompt_tags)]
            if full_candidates:
                selected["torso"] = max(full_candidates, key=lambda asset: _asset_score(asset, prompt_tags, "full", primary_family=primary_family))
        return selected

    def _choose_primary_family(self, prompt_tags: set[str], roles: list[str], max_asset_faces: int) -> str | None:
        if not roles or len(roles) <= 1:
            return None
        family_scores: dict[str, float] = {}
        for asset in self.assets:
            family = _asset_family(asset)
            if family is None:
                continue
            usable_bonus = 1.0 if asset.faces <= max_asset_faces else 0.35
            role_overlap = len(set(asset.roles) & set(roles))
            if role_overlap <= 0:
                continue
            family_scores[family] = family_scores.get(family, 0.0) + role_overlap * usable_bonus * (1.0 + len(set(asset.tags) & prompt_tags))
        if not family_scores:
            return None
        return max(family_scores, key=family_scores.get)

    @staticmethod
    def _base(scale_mm: float, prompt_tags: set[str]) -> trimesh.Trimesh:
        radius = max(10.0, min(18.0, float(scale_mm) * (0.38 if "vehicle" not in prompt_tags else 0.65)))
        base = trimesh.creation.cylinder(radius=radius, height=2.2, sections=96)
        base.apply_translation([0, 0, 1.1])
        return base


def default_mesh_latent_checkpoint_path() -> Path:
    return default_training_data_dir() / "checkpoints" / "meshmend_mesh_latent_index.json"


def _asset_from_mesh_path(mesh_path: Path) -> MeshLatentAsset | None:
    try:
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.geometry.values())
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            return None
        tags = sorted(_tags_from_text(mesh_path.stem))
        roles = _roles_from_tags(tags)
        extents = np.asarray(mesh.extents, dtype=float).round(5).tolist()
        return MeshLatentAsset(
            path=str(mesh_path),
            name=mesh_path.name,
            tags=tags,
            roles=roles,
            vertices=int(len(mesh.vertices)),
            faces=int(len(mesh.faces)),
            extents=extents,
            watertight=bool(mesh.is_watertight),
            file_size_bytes=int(mesh_path.stat().st_size),
            latent=_latent_vector(tags, roles, extents, len(mesh.faces)),
        )
    except Exception:
        return None


def _asset_from_training_example(example: TrainingExample) -> MeshLatentAsset | None:
    mesh_path = Path(example.mesh_path)
    if not mesh_path.exists():
        return None
    tags = sorted(set(example.tags) | _tags_from_text(mesh_path.stem))
    roles = _roles_from_tags(tags)
    extents = [float(value) for value in (example.extents or [1.0, 1.0, 1.0])]
    return MeshLatentAsset(
        path=str(mesh_path),
        name=mesh_path.name,
        tags=tags,
        roles=roles,
        vertices=int(example.vertices),
        faces=int(example.faces),
        extents=extents,
        watertight=bool(example.watertight),
        file_size_bytes=int(mesh_path.stat().st_size),
        latent=_latent_vector(tags, roles, extents, int(example.faces)),
    )


def _asset_from_file_metadata(mesh_path: Path) -> MeshLatentAsset | None:
    try:
        tags = sorted(_tags_from_text(mesh_path.stem))
        roles = _roles_from_tags(tags)
        file_size = int(mesh_path.stat().st_size)
        estimated_faces = max(1000, min(400_000, file_size // 50))
        extents = [1.0, 1.0, 1.0]
        return MeshLatentAsset(
            path=str(mesh_path),
            name=mesh_path.name,
            tags=tags,
            roles=roles,
            vertices=max(500, estimated_faces // 2),
            faces=estimated_faces,
            extents=extents,
            watertight=False,
            file_size_bytes=file_size,
            latent=_latent_vector(tags, roles, extents, estimated_faces),
        )
    except Exception:
        return None


def _tags_from_text(text: str) -> set[str]:
    split_camel = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text or "")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", split_camel).lower()
    words = {word for word in cleaned.split() if len(word) > 1 and word not in {"repaired", "trained", "generate", "mesh", "stl"}}
    aliases = {
        "auto": "rifle", "bolter": "rifle", "bolt": "rifle", "flammer": "flamer", "prime": "marine",
        "primeinfantry": "marine", "infantry": "soldier", "da": "marine", "pack": "backpack", "powerpack": "backpack",
        "shoylder": "shoulder", "pad": "shoulder", "pauldron": "shoulder", "body": "torso",
    }
    expanded = set(words)
    for word in list(words):
        if word in aliases:
            expanded.add(aliases[word])
    for style, style_words in STYLE_ALIASES.items():
        if expanded & style_words:
            expanded.add(style)
    return expanded or {"miniature"}


def _dedupe_assets(assets: list[MeshLatentAsset]) -> list[MeshLatentAsset]:
    deduped: dict[str, MeshLatentAsset] = {}
    for asset in assets:
        deduped[asset.path] = asset
    return list(deduped.values())


def _roles_from_tags(tags: list[str]) -> list[str]:
    tag_set = set(tags)
    roles = [role for role, keywords in ROLE_KEYWORDS.items() if tag_set & keywords]
    if not roles:
        roles = ["full"]
    if "left" in tag_set and "arm" in tag_set and "left_arm" not in roles:
        roles.append("left_arm")
    if "right" in tag_set and "arm" in tag_set and "right_arm" not in roles:
        roles.append("right_arm")
    if "arm" in tag_set and "left_arm" not in roles and "right_arm" not in roles:
        roles.extend(["left_arm", "right_arm"])
    return sorted(set(roles))


def _requested_roles(prompt_tags: set[str]) -> list[str]:
    object_tags = {"mask", "helmet", "helm", "face", "skull"}
    weapon_tags = {"rifle", "gun", "weapon", "sword", "axe", "hammer", "mace", "bow", "launcher", "flamer"}
    if prompt_tags & weapon_tags and prompt_tags & {"prop", "object", "accessory"}:
        return ["weapon"]
    if prompt_tags & object_tags and not (prompt_tags & CHARACTER_TAGS):
        return ["head"]
    if prompt_tags & weapon_tags and not (prompt_tags & CHARACTER_TAGS):
        return ["weapon"]
    if "bust" in prompt_tags:
        return ["head", "torso", "shoulder", "backpack"]
    return ["head", "torso", "legs", "left_arm", "right_arm", "weapon", "shoulder", "backpack"]


def _latent_vector(tags: list[str], roles: list[str], extents: list[float], faces: int) -> list[float]:
    tag_seed = sum((index + 1) * sum(ord(ch) for ch in tag) for index, tag in enumerate(tags))
    ext = np.asarray(extents or [1, 1, 1], dtype=float)
    max_ext = float(np.max(ext)) + 1e-8
    return [
        float(ext[0] / max_ext),
        float(ext[1] / max_ext),
        float(ext[2] / max_ext),
        min(1.0, float(faces) / 1_000_000.0),
        (tag_seed % 997) / 997.0,
        len(roles) / 8.0,
    ]


def _asset_score(asset: MeshLatentAsset, prompt_tags: set[str], role: str, primary_family: str | None = None) -> float:
    tags = set(asset.tags)
    score = 0.0
    score += len(tags & prompt_tags) * 40.0
    if role in asset.roles:
        score += 35.0
    if primary_family is not None and _asset_family(asset) == primary_family:
        score += COHERENT_FAMILY_BONUS
    if role in {"left_arm", "right_arm"}:
        wanted_side = "left" if role == "left_arm" else "right"
        actual_side = _asset_side(asset)
        if actual_side == wanted_side:
            score += SIDE_MATCH_BONUS
        elif actual_side is not None:
            score -= SIDE_MISMATCH_PENALTY
    for style, style_tags in STYLE_ALIASES.items():
        if style in prompt_tags and tags & style_tags:
            score += 25.0
    score += min(asset.faces / 50_000.0, 18.0)
    if not asset.watertight:
        score -= 5.0
    return score


def _asset_family(asset: MeshLatentAsset) -> str | None:
    name = asset.name.lower()
    tags = set(asset.tags)
    if name.startswith("primeranged_") or name.startswith("primeinfantry_") or name.startswith("prime") or "prime" in tags:
        return "prime"
    if name.startswith("da_") or "da" in tags:
        return "da"
    if "orc" in tags or "ork" in tags:
        return "orc"
    if "assassin" in tags:
        return "assassin"
    if "meshy" in tags:
        return "meshy"
    return None


def _asset_side(asset: MeshLatentAsset) -> str | None:
    name = asset.name.lower()
    if re.search(r"(?:^|[_\-\s])l(?:[_\-\s\.]|$)", name) or "left" in asset.tags:
        return "left"
    if re.search(r"(?:^|[_\-\s])r(?:[_\-\s\.]|$)", name) or "right" in asset.tags:
        return "right"
    return None


def _requested_weapon_category(prompt_tags: set[str]) -> str | None:
    if prompt_tags & {"axe", "sword", "hammer", "mace"}:
        return "melee"
    if prompt_tags & {"rifle", "gun", "bolter", "launcher", "flamer", "bow"}:
        return "ranged"
    return None


def _asset_weapon_category(asset: MeshLatentAsset) -> str | None:
    tags = set(asset.tags)
    if tags & {"axe", "sword", "hammer", "mace"}:
        return "melee"
    if tags & {"rifle", "gun", "bolter", "launcher", "flamer", "bow", "pistol"}:
        return "ranged"
    return None


def _has_meaningful_prompt_overlap(asset: MeshLatentAsset, prompt_tags: set[str]) -> bool:
    generic = {
        "detailed", "highly", "intricate", "ornate", "printable", "studio", "production",
        "mask", "helmet", "helm", "head", "face", "skull", "weapon", "prop", "object",
        "miniature", "wargaming", "tabletop",
    }
    meaningful_prompt = prompt_tags - generic
    if not meaningful_prompt:
        return True
    return bool(set(asset.tags) & meaningful_prompt)


def _orient_role_mesh(mesh: trimesh.Trimesh, role: str, asset: MeshLatentAsset) -> trimesh.Trimesh:
    mesh = mesh.copy()
    side = _asset_side(asset)
    if role == "right_arm" and side == "left":
        mesh.apply_scale([-1.0, 1.0, 1.0])
        mesh.invert()
    elif role == "left_arm" and side == "right":
        mesh.apply_scale([-1.0, 1.0, 1.0])
        mesh.invert()
    return mesh


def _load_asset_mesh(path: str, target_faces: int) -> trimesh.Trimesh | None:
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.geometry.values())
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            return None
        mesh = mesh.copy()
        mesh.remove_unreferenced_vertices()
        if len(mesh.faces) > target_faces:
            mesh = _simplify(mesh, target_faces)
        return mesh
    except Exception:
        return None


def _simplify(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
        if isinstance(simplified, trimesh.Trimesh) and len(simplified.faces) > 0:
            simplified.remove_unreferenced_vertices()
            return simplified
    except TypeError:
        try:
            simplified = mesh.simplify_quadric_decimation(target_faces)
            if isinstance(simplified, trimesh.Trimesh) and len(simplified.faces) > 0:
                simplified.remove_unreferenced_vertices()
                return simplified
        except Exception:
            pass
    except Exception:
        pass
    # Never fall back to arbitrary face sampling: it creates isolated triangle
    # fragments that render as transparent dots instead of filled surfaces. The
    # selector prefers assets below target_faces, so if quadric decimation is not
    # installed we skip over-large parts rather than exporting point-cloud-like STLs.
    raise RuntimeError("mesh latent simplification unavailable")


def _fit_mesh(mesh: trimesh.Trimesh, center: tuple[float, float, float], target_size: float, axis: str) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    extents = np.asarray(mesh.extents, dtype=float)
    reference = float(extents[2] if axis == "z" else np.max(extents))
    if reference <= 1e-8:
        return mesh
    mesh.apply_scale(float(target_size) / reference)
    mesh.apply_translation(np.asarray(center, dtype=float))
    return mesh


def _role_transforms(scale_mm: float) -> dict[str, tuple[tuple[float, float, float], float, str]]:
    height = float(scale_mm)
    return {
        "head": ((0.0, -0.45, height * 0.88), height * 0.18, "max"),
        "torso": ((0.0, 0.0, height * 0.62), height * 0.42, "z"),
        "legs": ((0.0, 0.0, height * 0.30), height * 0.42, "z"),
        "left_arm": ((-height * 0.18, -0.45, height * 0.58), height * 0.35, "z"),
        "right_arm": ((height * 0.18, -0.45, height * 0.58), height * 0.35, "z"),
        "weapon": ((height * 0.17, -height * 0.10, height * 0.55), height * 0.46, "max"),
        "shoulder": ((-height * 0.18, -0.18, height * 0.73), height * 0.18, "max"),
        "backpack": ((0.0, height * 0.10, height * 0.64), height * 0.24, "max"),
    }
