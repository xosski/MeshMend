"""
AI-powered model generation using Diffusers and 3D volumetric conversion
"""

from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np
from pathlib import Path
import re
import sys
import trimesh
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
import warnings
warnings.filterwarnings("ignore")

from backend.pipelines.image_to_mesh import reconstruct_mesh_from_image, reconstruct_mesh_from_image_volumetric
from backend.processing.mesh_simplifier import MeshSimplifier
from app.mesh_from_volume import volume_to_mesh_trimesh
from app.llm_part_planner import LLMPartPlanner
from app.part_mesh_builder import PartMeshBuilder
from app.stl_quality_reviewer import STLQualityReviewer


class PromptEnhancer:
    """Enhance user prompts with style modifiers and return proper positive/negative pair"""

    STYLE_MODIFIERS = {
        "Sci-Fi Soldier": "futuristic, sci-fi, military armor, high tech, professional render",
        "Fantasy Knight": "medieval, fantasy armor, sword, heroic, professional render",
        "Chaos Warrior": "dark, corrupted, spiky armor, demonic, professional render",
        "Elven Archer": "elvish, graceful, bow and arrow, ethereal, professional render",
        "Dwarf Engineer": "dwarf, stocky, mechanical gadgets, beard, professional render",
        "Ork Brute": "ork, muscular, brutish, tusks, crude armor, professional render",
        "Undead Warrior": "skeleton, undead, dark, grim, bonewraith, professional render",
        "Generic Humanoid": "human warrior, armor, weapon, heroic, professional render"
    }

    @staticmethod
    def enhance_prompt(user_prompt: str, style: str = "Generic Humanoid") -> tuple[str, str]:
        """Enhance prompt with style modifiers and return positive/negative prompt pair."""
        modifier = PromptEnhancer.STYLE_MODIFIERS.get(style, "")
        scale_match = re.search(r"\b(15|20|25|28|30|32|35|40|48|54|75)\s*mm\b", user_prompt.lower())
        scale_text = f"{scale_match.group(1)}mm" if scale_match else "32mm"
        intent = _generation_intent(user_prompt)

        positive = f"{user_prompt}, {modifier}, "
        if intent == "character_miniature":
            positive += f"tabletop miniature figure for STL generation, {scale_text} heroic scale, resin-printable sculpt, "
            positive += "full body subject, clear silhouette, distinctive pose and equipment matching the description, "
        else:
            positive += f"standalone {scale_text} resin-printable STL subject, no forced humanoid body, topology matching the requested object, "
            positive += "clear silhouette and distinctive shape matching the description, "
        positive += "highly detailed, centered, isometric view, museum quality, professional sculpt, "
        positive += "clean watertight geometry, sharp features, deep recesses, dramatic lighting"

        negative = (
            "blurry, low quality, distorted, ugly, bad anatomy, noisy, duplicate, "
            "muted colors, flat, dull, abstract, deformed, oversized features, "
            "cartoon, sketch, painting, watermark, text, signature, out of focus, "
            "smooth blob, melted lump, rock, plain cylinder, base only, generic pedestal, "
            "featureless silhouette, unrelated shape"
        )

        return positive, negative


def _generation_intent(prompt: str) -> str:
    lower = (prompt or "").lower()
    tokens = set(re.findall(r"[a-z0-9']+", lower))
    if any(term in lower for term in ("full body", "full-body", "whole character", "entire character")):
        return "character_miniature"
    if tokens & {"mask", "helmet", "helm", "headpiece", "faceplate"}:
        return "wearable_object"
    if tokens & {"rifle", "gun", "pistol", "sword", "axe", "hammer", "shield", "banner", "weapon", "prop", "accessory"}:
        return "prop_object"
    if tokens & {"bust", "portrait"} or "head bust" in lower:
        return "bust"
    if tokens & {"terrain", "scenery", "building", "ruin", "dungeon", "objective"}:
        return "terrain_object"
    if tokens & {"vehicle", "tank", "ship", "walker", "turret"}:
        return "vehicle_object"
    character_terms = {
        "miniature", "figure", "character", "warrior", "soldier", "knight", "wizard", "mage",
        "orc", "ork", "elf", "dwarf", "demon", "daemon", "undead", "skeleton", "zombie",
        "ranger", "archer", "marine", "humanoid", "creature", "monster", "beast",
    }
    if tokens & character_terms:
        return "character_miniature"
    return "printable_subject"


class AIGeneratorWorker(QObject):
    """Worker thread for AI model generation"""

    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, prompt: str, style: str, detail_level: int, scale: float, source_image_path: str | None = None):
        super().__init__()
        self.prompt = prompt
        self.style = style
        self.detail_level = detail_level
        self.scale = scale
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source_image_path = source_image_path

    def run(self):
        """Run the generation process."""
        try:
            self.progress.emit(5)
            generation_prompt = self._miniature_generation_prompt()

            mesh = self._generate_with_meshmend_sculptor(generation_prompt)
            if mesh is not None:
                self.progress.emit(100)
                self.finished.emit(mesh)
                return

            planner = LLMPartPlanner()
            plan = planner.create_plan(
                prompt=generation_prompt,
                style=self.style,
                scale_mm=self.scale,
            )

            print("\n=== MINIATURE PART PLAN ===")
            print(plan.to_json())

            # Prompt-only mode: build primitive 3D scaffold.
            if self.prompt and not self.source_image_path:
                self.progress.emit(30)

                builder = PartMeshBuilder(use_llm_lookup=True)
                mesh = builder.build_from_plan(plan)
                mesh = self._apply_meshmend_trained_exemplar_layer(mesh, generation_prompt)

                self.progress.emit(75)

                mesh = self._validate_mesh(mesh)

                mesh = self._review_mesh_or_fail(mesh, generation_prompt)

                self.progress.emit(100)
                self.finished.emit(mesh)
                return

            self.progress.emit(10)

            if self.source_image_path:
                image = Image.open(self.source_image_path).convert("RGB")
            else:
                image = self._generate_image()

            self.progress.emit(40)

            mesh = reconstruct_mesh_from_image(
                image,
                num_points=self._num_points_for_detail(),
                method="poisson",
                smooth_iterations=0,
            )

            if self._is_blob_like_mesh(mesh):
                mesh = reconstruct_mesh_from_image(
                    image,
                    num_points=int(self._num_points_for_detail() * 1.2),
                    method="ball_pivot",
                    smooth_iterations=0,
                )

            if self._is_collapsed_mesh(mesh):
                self.progress.emit(65)
                mesh = reconstruct_mesh_from_image_volumetric(
                    image,
                    vol_size=128,
                    voxel_size_mm=0.22,
                )

            if self.source_image_path and self._is_base_only_or_blob_mesh(mesh):
                print("Image reconstruction looked base/blob-like; using silhouette-preserving fallback")
                mesh = self._fallback_mesh_from_image(image)

            if self.source_image_path and self._is_underfleshed_mesh(mesh):
                print("Primary mesh under-fleshed; attempting fuller fallback")
                try:
                    fuller = self._fallback_mesh_from_image(image)
                    if self._mesh_fullness_score(fuller) > self._mesh_fullness_score(mesh):
                        mesh = fuller
                    else:
                        mesh = self._inflate_mesh_volume(mesh)
                except Exception:
                    mesh = self._inflate_mesh_volume(mesh)

            mesh = MeshSimplifier.polish_mesh(
                mesh,
                quality=self._quality_from_detail_level(),
            )
            mesh = MeshSimplifier.simplify_for_printing(
                mesh,
                quality=self._quality_from_detail_level(),
            )

            self.progress.emit(85)

            mesh = self._validate_mesh(mesh)

            scale_factor = self.scale / 28.0
            mesh.apply_scale(scale_factor)

            mesh = self._review_mesh_or_fail(mesh, generation_prompt)

            self.progress.emit(100)
            self.finished.emit(mesh)

        except Exception as e:
            self.error.emit(str(e))

    def _generate_with_meshmend_sculptor(self, generation_prompt: str) -> trimesh.Trimesh | None:
        """Use the MeshMend image/text-aware sculptor as the primary generator.

        The older standalone path reconstructs a mesh directly from pixels, which
        often collapses unrelated images into the same relief/blob. MeshMend's
        sculptor converts image/text inputs into a miniature part plan first, so
        prompt and reference features can affect actual geometry.
        """
        try:
            meshmend_src = Path(__file__).resolve().parents[3]
            meshmend_src_text = str(meshmend_src)
            if meshmend_src_text not in sys.path:
                sys.path.insert(0, meshmend_src_text)
            from meshmend_ai.sculptor import get_sculptor_foundation

            self.progress.emit(20)
            output_path = get_sculptor_foundation().create_model(
                self.prompt.strip() if isinstance(self.prompt, str) and self.prompt.strip() else generation_prompt,
                image_path=Path(self.source_image_path) if self.source_image_path else None,
                progress=lambda percent, _message: self.progress.emit(max(20, min(95, int(percent)))),
                scale_mm=self.scale,
            )
            mesh = trimesh.load(output_path, force="mesh")
            if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) >= 1000:
                return mesh
        except Exception as exc:
            print(f"MeshMend sculptor primary generation unavailable; falling back: {exc}")
        return None

    def _apply_meshmend_trained_exemplar_layer(self, mesh: trimesh.Trimesh, generation_prompt: str) -> trimesh.Trimesh:
        """Use MeshMend's trained local model library for prompt-specific detail.

        The standalone 3D Sculptor builder is procedural. This bridge lets it use
        the same trained-exemplar kitbash layer as MeshMend's create workflow so
        orc/weapon prompts can inherit actual trained STL detail.
        """
        if not any(term in generation_prompt.lower() for term in ("orc", "ork", "warboss")):
            return mesh
        try:
            meshmend_src = Path(__file__).resolve().parents[3]
            meshmend_src_text = str(meshmend_src)
            if meshmend_src_text not in sys.path:
                sys.path.insert(0, meshmend_src_text)
            from meshmend_ai.sculptor import get_sculptor_foundation

            return get_sculptor_foundation()._augment_with_trained_kitbash(generation_prompt, mesh)
        except Exception as exc:
            print(f"Trained exemplar detail layer unavailable: {exc}")
            return mesh

    def _miniature_generation_prompt(self) -> str:
        """Return the prompt every downstream AI/reviewer sees for this job."""
        user_text = self.prompt.strip() if isinstance(self.prompt, str) else ""
        if user_text:
            subject = user_text
        elif self.source_image_path:
            filename_cues = " ".join(re.split(r"[_\-.\s]+", Path(self.source_image_path).stem)).strip()
            subject = (
                "the imported source image subject"
                + (f" ({filename_cues})" if filename_cues else "")
                + ", preserving its silhouette, pose, clothing, gear, and distinctive visual features"
            )
        else:
            subject = "a detailed character"

        scale_text = f"{self.scale:g}mm" if self.scale else "28mm"
        intent = _generation_intent(subject)
        if intent == "character_miniature":
            return (
                f"Create a {scale_text} tabletop miniature STL of {subject}. "
                "This is a small resin-printable miniature, not a life-size object, not a plain rock, "
                "and not a cylinder or base-only model. Preserve the specific subject, pose, silhouette, "
                "equipment, facial/animal features, clothing, armor, and texture cues from the input. "
                "Use exaggerated miniature details, deep recesses, thick printable parts, and a clear full-body silhouette."
            )

        return (
            f"Create a {scale_text} standalone resin-printable STL of {subject}. "
            "Do not force this into a humanoid or full-body miniature unless the user explicitly requested a character. "
            "Preserve the requested object topology, silhouette, openings, rims, panels, straps, mechanical detail, texture, "
            "and other visual cues from the input. Use crisp raised details, deep recesses, printable wall thickness, and watertight geometry."
        )

    def _review_mesh_or_fail(self, mesh: trimesh.Trimesh, prompt: str | None = None) -> trimesh.Trimesh:
        """Run final STL quality review before returning the mesh."""
        reviewer = STLQualityReviewer()
        review = reviewer.review(mesh, prompt or self._miniature_generation_prompt())

        print("\n=== STL QUALITY REVIEW ===")
        print(review)

        if not review.get("passed", False):
            raise Exception(
                "Generated STL failed quality review: "
                + "; ".join(review.get("issues", []))
            )

        return mesh

    def _generate_image(self) -> Image.Image:
        """Generate 2D image using Stable Diffusion with proper prompt handling"""
        try:
            # Enhance prompt and get separate positive/negative
            positive_prompt, negative_prompt = PromptEnhancer.enhance_prompt(
                self._miniature_generation_prompt(),
                self.style
            )
            
            print(f"Positive prompt: {positive_prompt}")
            print(f"Negative prompt: {negative_prompt}")
            
            # Load model
            model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            )
            pipe = pipe.to(self.device)
            pipe.enable_attention_slicing()
            
            # Generate with proper positive AND negative prompts
            result = pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=512,
                width=512
            )
            
            return result.images[0].convert("RGB")
            
        except Exception as e:
            raise Exception(f"Image generation failed: {str(e)}")
    
    def _num_points_for_detail(self) -> int:
        """Map UI detail level to reconstruction point budget."""
        mapping = {
            1: 14000,
            2: 22000,
            3: 32000,
            4: 44000,
            5: 64000,
        }
        return mapping.get(int(self.detail_level), 32000)

    def _quality_from_detail_level(self) -> str:
        level = int(self.detail_level)
        if level >= 4:
            return "high"
        if level <= 2:
            return "low"
        return "standard"

    def _is_blob_like_mesh(self, mesh: trimesh.Trimesh) -> bool:
        try:
            if mesh is None or len(mesh.vertices) < 400 or len(mesh.faces) < 600:
                return True
            extents = np.asarray(mesh.extents, dtype=np.float32)
            max_extent = float(np.max(extents))
            min_extent = float(np.min(extents))
            if max_extent <= 1e-6:
                return True
            flat_ratio = min_extent / max_extent
            bbox_vol = float(np.prod(np.maximum(extents, 1e-6)))
            mesh_vol = float(abs(mesh.volume))
            fullness = mesh_vol / (bbox_vol + 1e-8)
            return flat_ratio < 0.08 or fullness > 0.72
        except Exception:
            return True

    def _is_base_only_or_blob_mesh(self, mesh: trimesh.Trimesh) -> bool:
        """Detect generic rock/cylinder/base outputs before they reach STL export."""
        try:
            if mesh is None or len(mesh.vertices) < 500 or len(mesh.faces) < 800:
                return True

            ext = np.asarray(mesh.extents, dtype=np.float32)
            max_ext = float(np.max(ext)) + 1e-8
            min_ext = float(np.min(ext))
            flat_ratio = min_ext / max_ext
            bbox_vol = float(np.prod(np.maximum(ext, 1e-6)))
            fullness = float(abs(mesh.volume)) / (bbox_vol + 1e-8)

            # Base/rock failures tend to be short, wide, and overly filled.
            height = float(ext[2])
            width = float(max(ext[0], ext[1])) + 1e-8
            squat_ratio = height / width
            if squat_ratio < 0.32 and fullness > 0.18:
                return True

            # Smooth convex-ish blobs/cylinders occupy too much of their bbox and lack relief.
            if fullness > 0.68 or flat_ratio < 0.075:
                return True

            return False
        except Exception:
            return True

    def _is_detail_poor_mesh(self, mesh: trimesh.Trimesh) -> bool:
        try:
            if mesh is None:
                return True
            v_count = int(len(mesh.vertices))
            f_count = int(len(mesh.faces))
            if v_count < 1800 or f_count < 3000:
                return True
            ext = np.asarray(mesh.extents, dtype=np.float32)
            max_ext = float(np.max(ext)) + 1e-8
            diagonal = float(np.linalg.norm(ext)) + 1e-8
            density = f_count / (diagonal * max_ext + 1e-8)
            return density < 550.0
        except Exception:
            return True

    def _is_collapsed_mesh(self, mesh: trimesh.Trimesh) -> bool:
        try:
            if mesh is None or len(mesh.vertices) < 900 or len(mesh.faces) < 1400:
                return True
            ext = np.asarray(mesh.extents, dtype=np.float32)
            max_ext = float(np.max(ext)) + 1e-8
            min_ext = float(np.min(ext))
            flat_ratio = min_ext / max_ext
            if flat_ratio < 0.045:
                return True
            bbox_vol = float(np.prod(np.maximum(ext, 1e-6)))
            fullness = float(abs(mesh.volume)) / (bbox_vol + 1e-8)
            return fullness < 0.018 or fullness > 0.86
        except Exception:
            return True

    def _is_trivial_mesh(self, mesh: trimesh.Trimesh) -> bool:
        """Detect collapsed outputs such as flat polygons or near-empty meshes."""
        try:
            if mesh is None or len(mesh.vertices) < 80 or len(mesh.faces) < 120:
                return True

            extents = np.asarray(mesh.extents, dtype=np.float32)
            max_extent = float(np.max(extents))
            min_extent = float(np.min(extents))
            if max_extent <= 1e-6:
                return True

            flat_ratio = min_extent / max_extent
            if flat_ratio < 0.025:
                return True

            return False
        except Exception:
            return True

    def _mesh_fullness_score(self, mesh: trimesh.Trimesh) -> float:
        """Estimate how volumetrically filled a mesh is relative to its bbox."""
        try:
            ext = np.asarray(mesh.extents, dtype=np.float32)
            bbox_vol = float(np.prod(np.maximum(ext, 1e-6)))
            mesh_vol = float(abs(mesh.volume))
            return mesh_vol / (bbox_vol + 1e-8)
        except Exception:
            return 0.0

    def _is_underfleshed_mesh(self, mesh: trimesh.Trimesh) -> bool:
        """Detect thin, fish-like meshes that need volumetric thickening."""
        try:
            ext = np.asarray(mesh.extents, dtype=np.float32)
            max_ext = float(np.max(ext)) + 1e-8
            min_ext = float(np.min(ext))
            thickness_ratio = min_ext / max_ext
            fullness = self._mesh_fullness_score(mesh)
            return thickness_ratio < 0.14 or fullness < 0.04
        except Exception:
            return True

    def _inflate_mesh_volume(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Thicken thinnest axis and apply mild isotropic inflate."""
        try:
            ext = np.asarray(mesh.extents, dtype=np.float32)
            max_ext = float(np.max(ext)) + 1e-8
            thin_axis = int(np.argmin(ext))
            min_target = max_ext * 0.18
            axis_scale = max(1.0, min_target / (float(ext[thin_axis]) + 1e-8))

            # Scale around center mass.
            center = np.asarray(mesh.center_mass, dtype=np.float32)
            verts = np.asarray(mesh.vertices, dtype=np.float32) - center[None, :]
            scale_vec = np.ones(3, dtype=np.float32) * 1.06
            scale_vec[thin_axis] *= min(axis_scale, 2.2)
            verts *= scale_vec[None, :]
            mesh.vertices = verts + center[None, :]

            mesh = self._clean_mesh(mesh)
            return mesh
        except Exception:
            return mesh

    def _fallback_mesh_from_image(self, image: Image.Image) -> trimesh.Trimesh:
        """Build a volumetric mesh from image silhouette/depth cues as robust fallback."""
        from scipy.ndimage import (
            binary_closing,
            binary_fill_holes,
            binary_opening,
            binary_erosion,
            distance_transform_edt,
            gaussian_filter,
            label,
            zoom,
        )
        from collections import deque

        img = image.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        gray = np.mean(arr, axis=2)
        sat = np.max(arr, axis=2) - np.min(arr, axis=2)

        # Prefer GrabCut when available; it's far more robust than raw threshold masks.
        fg_grabcut = None
        try:
            import cv2

            arr_u8 = (arr * 255.0).astype(np.uint8)
            h_gc, w_gc = arr_u8.shape[:2]
            gc_mask = np.zeros((h_gc, w_gc), np.uint8)
            bg_model = np.zeros((1, 65), np.float64)
            fg_model = np.zeros((1, 65), np.float64)
            rect = (max(1, int(w_gc * 0.06)), max(1, int(h_gc * 0.06)), int(w_gc * 0.88), int(h_gc * 0.88))
            cv2.grabCut(arr_u8, gc_mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
            fg_grabcut = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)

            fg_ratio_gc = float(np.mean(fg_grabcut))
            if fg_ratio_gc < 0.01 or fg_ratio_gc > 0.75:
                fg_grabcut = None
        except Exception:
            fg_grabcut = None

        # Estimate background from image border colors and segment foreground by color distance.
        border = np.concatenate(
            [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]],
            axis=0,
        )
        bg_color = np.median(border, axis=0)
        color_dist = np.linalg.norm(arr - bg_color[None, None, :], axis=2)

        edge_x = np.gradient(gray, axis=1)
        edge_y = np.gradient(gray, axis=0)
        edge_mag = np.sqrt(edge_x**2 + edge_y**2)

        gray_q = np.quantile(gray, 0.72)
        sat_q = np.quantile(sat, 0.60)
        dist_q = np.quantile(color_dist, 0.73)
        edge_q = np.quantile(edge_mag, 0.84)

        fg = (color_dist > dist_q) | (sat > sat_q) | (gray < gray_q) | (edge_mag > edge_q)
        if fg_grabcut is not None:
            fg = fg & fg_grabcut

        # Remove anything connected to border to avoid square/background frame capture.
        border_seed = np.zeros_like(fg, dtype=bool)
        border_seed[0, :] = True
        border_seed[-1, :] = True
        border_seed[:, 0] = True
        border_seed[:, -1] = True

        bg_connected = np.zeros_like(fg, dtype=bool)
        queue = deque(zip(*np.where(border_seed & ~fg)))
        while queue:
            y_idx, x_idx = queue.popleft()
            if bg_connected[y_idx, x_idx]:
                continue
            bg_connected[y_idx, x_idx] = True
            for ny, nx in ((y_idx - 1, x_idx), (y_idx + 1, x_idx), (y_idx, x_idx - 1), (y_idx, x_idx + 1)):
                if 0 <= ny < fg.shape[0] and 0 <= nx < fg.shape[1] and (not fg[ny, nx]) and (not bg_connected[ny, nx]):
                    queue.append((ny, nx))

        fg = ~bg_connected

        # Keep best central component to avoid selecting border/background blobs.
        labeled, num = label(fg)
        if num > 0:
            h, w = gray.shape
            yy, xx = np.indices((h, w))
            cx, cy = w * 0.5, h * 0.5
            best_score = -1.0
            best_mask = fg

            for comp_idx in range(1, num + 1):
                comp = labeled == comp_idx
                area = float(np.sum(comp))
                if area < (h * w * 0.002):
                    continue

                mean_x = float(np.mean(xx[comp]))
                mean_y = float(np.mean(yy[comp]))
                center_penalty = np.sqrt(((mean_x - cx) / w) ** 2 + ((mean_y - cy) / h) ** 2)
                score = area * (1.0 - min(0.95, center_penalty))
                if score > best_score:
                    best_score = score
                    best_mask = comp

            fg = best_mask

        # If mask is too large, tighten it aggressively; this prevents canvas-wide squares/cylinders.
        fg_ratio = float(np.mean(fg))
        if fg_ratio > 0.52:
            tighter = (
                (color_dist > np.quantile(color_dist, 0.82))
                | (sat > np.quantile(sat, 0.72))
                | (edge_mag > np.quantile(edge_mag, 0.88))
            )
            fg = tighter

        # If the image still collapses to a generic mask, keep only strong detail cues
        # around the center instead of meshing a same-looking rock/cylinder every time.
        fg_ratio = float(np.mean(fg))
        if fg_ratio < 0.012 or fg_ratio > 0.62:
            h, w = gray.shape
            yy, xx = np.indices((h, w))
            center_prior = (((xx - w * 0.5) / max(w * 0.45, 1)) ** 2 + ((yy - h * 0.52) / max(h * 0.45, 1)) ** 2) < 1.0
            fg = center_prior & (
                (color_dist > np.quantile(color_dist, 0.78))
                | (sat > np.quantile(sat, 0.68))
                | (edge_mag > np.quantile(edge_mag, 0.82))
            )

        fg = binary_opening(fg, iterations=1)
        fg = binary_closing(fg, iterations=2)
        fg = binary_fill_holes(fg)
        fg = binary_erosion(fg, iterations=1)

        depth = 1.0 - gray
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        vol_size = 96
        fg_resized = zoom(fg.astype(np.float32), (vol_size / fg.shape[0], vol_size / fg.shape[1]), order=0) > 0.5
        depth_resized = zoom(depth.astype(np.float32), (vol_size / depth.shape[0], vol_size / depth.shape[1]), order=1)
        depth_resized = np.clip(depth_resized, 0.0, 1.0)

        fg_resized = binary_fill_holes(binary_closing(fg_resized, iterations=2))
        fg_resized = binary_erosion(fg_resized, iterations=1)
        dist_inside = distance_transform_edt(fg_resized.astype(np.uint8)).astype(np.float32)
        dist_inside = dist_inside / (dist_inside.max() + 1e-8)

        volume = np.zeros((vol_size, vol_size, vol_size), dtype=np.float32)

        # Cross-sectional volumetric build (ellipse per row) to flesh out X/Y/Z coherently.
        min_half_thick = int(vol_size * 0.06)
        max_half_thick = int(vol_size * 0.26)
        center_z = int(vol_size * 0.50)

        for y_idx in range(vol_size):
            row_mask = fg_resized[y_idx]
            row_x = np.where(row_mask)[0]
            if len(row_x) < 3:
                continue

            x_left = int(row_x.min())
            x_right = int(row_x.max())
            x_center = 0.5 * (x_left + x_right)
            rx_row = max(1.5, 0.5 * (x_right - x_left))
            row_depth = float(np.mean(depth_resized[y_idx, row_mask]))

            for x_idx in row_x:
                dx_norm = (x_idx - x_center) / (rx_row + 1e-8)
                ellipse_scale = np.sqrt(max(0.0, 1.0 - dx_norm * dx_norm))
                if ellipse_scale <= 0.0:
                    continue

                center_weight = float(dist_inside[y_idx, x_idx])
                local_depth = float(depth_resized[y_idx, x_idx])

                half_thickness = min_half_thick + int(
                    (max_half_thick - min_half_thick)
                    * (0.28 + 0.72 * center_weight)
                    * (0.42 + 0.58 * local_depth)
                    * ellipse_scale
                )
                half_thickness = max(1, half_thickness)

                # Depth shifts center of mass slightly without collapsing to a relief plate.
                z_shift = int((local_depth - 0.5) * vol_size * 0.07)
                row_shift = int((row_depth - 0.5) * vol_size * 0.03)
                z_center_local = int(np.clip(center_z + z_shift + row_shift, 0, vol_size - 1))

                z0 = max(0, z_center_local - half_thickness)
                z1 = min(vol_size, z_center_local + half_thickness)
                volume[z0:z1, y_idx, x_idx] = 1.0

        # Rounded taper on both sides to avoid hard backplane artifacts.
        for z_idx in range(vol_size):
            z_norm = abs((z_idx - center_z) / max(1, center_z))
            taper = max(0.0, 1.0 - z_norm**1.35)
            volume[z_idx] *= max(0.0, taper)

        # Mild anisotropic blur, then threshold to keep form crisp.
        volume = gaussian_filter(volume, sigma=(0.65, 0.35, 0.65))
        volume = (volume > 0.33).astype(np.float32)

        mesh = volume_to_mesh_trimesh(volume, kind="occupancy", iso=0.58, voxel_size_mm=0.24)
        mesh = self._clean_mesh(mesh)

        # Anti-relief guard: if still too flat, inflate depth axis and clean again.
        try:
            extents = np.asarray(mesh.extents, dtype=np.float32)
            flat_ratio = float(np.min(extents) / (np.max(extents) + 1e-8))
            if flat_ratio < 0.11:
                z_axis = int(np.argmin(extents))
                scale_vec = np.ones(3, dtype=np.float32)
                scale_vec[z_axis] = 1.8
                mesh.vertices = mesh.vertices * scale_vec[None, :]
                mesh = self._clean_mesh(mesh)
        except Exception:
            pass

        if self._is_trivial_mesh(mesh):
            raise Exception("Fallback reconstruction still produced a trivial mesh")

        return mesh
    
    def _clean_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Clean and simplify noisy meshes into a single printable body."""
        try:
            # Split into connected components and keep only the dominant body.
            components = [m for m in mesh.split(only_watertight=False) if len(m.vertices) > 30 and len(m.faces) > 40]
            if not components:
                return mesh

            components = sorted(components, key=lambda m: float(m.volume) if m.volume > 0 else float(m.area), reverse=True)
            main = components[0]

            # Force single-body output to avoid polygon masses from noisy satellites.
            mesh = main

            # Core cleanup ops.
            if hasattr(mesh, "remove_unreferenced_vertices"):
                mesh.remove_unreferenced_vertices()
            if hasattr(mesh, "merge_vertices"):
                mesh.merge_vertices()
            if hasattr(mesh, "remove_degenerate_faces"):
                mesh.remove_degenerate_faces()
            if hasattr(mesh, "remove_duplicate_faces"):
                mesh.remove_duplicate_faces()
            
            # Keep enough geometry for miniature details; only decimate truly noisy outputs.
            target_faces = {1: 8000, 2: 14000, 3: 24000, 4: 42000, 5: 70000}.get(int(self.detail_level), 24000)
            if len(mesh.faces) > target_faces and hasattr(mesh, "simplify_quadratic_decimation"):
                try:
                    mesh = mesh.simplify_quadratic_decimation(target_faces)
                except Exception:
                    pass

            # Very light smoothing only after simplification.
            smooth_iters = 0 if self.detail_level <= 2 else 1
            if smooth_iters > 0:
                mesh = self._smooth_mesh(mesh, iterations=smooth_iters)

            return mesh

        except Exception as e:
            print(f"Mesh cleaning warning: {e}")
            return mesh
    
    def _smooth_mesh(self, mesh: trimesh.Trimesh, iterations: int = 1) -> trimesh.Trimesh:
        """Apply smoothing to mesh vertices"""
        try:
            for _ in range(iterations):
                # Get adjacency information
                neighbors = [[] for _ in range(len(mesh.vertices))]
                
                for face in mesh.faces:
                    for i in range(3):
                        v1 = face[i]
                        v2 = face[(i + 1) % 3]
                        if v2 not in neighbors[v1]:
                            neighbors[v1].append(v2)
                        if v1 not in neighbors[v2]:
                            neighbors[v2].append(v1)
                
                # Smooth vertices
                new_vertices = mesh.vertices.copy()
                for i, neighbors_list in enumerate(neighbors):
                    if neighbors_list:
                        # Average with neighbors
                        avg = mesh.vertices[neighbors_list].mean(axis=0)
                        # Blend: keep 70% original, 30% average
                        new_vertices[i] = 0.7 * mesh.vertices[i] + 0.3 * avg
                
                mesh.vertices = new_vertices
            
            return mesh
            
        except Exception as e:
            print(f"Smoothing error: {e}")
            return mesh
    
    def _validate_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Validate and normalize mesh"""
        try:
            # Ensure mesh is valid
            mesh.process(validate=True)
            
            # Get mesh bounds
            bounds = mesh.bounds
            sizes = bounds[1] - bounds[0]
            
            # If mesh is extremely small, it likely failed
            if np.max(sizes) < 0.1:
                raise Exception("Generated mesh is too small - likely generation failed")
            
            # Center mesh at origin
            mesh.vertices -= mesh.center_mass
            
            return mesh
            
        except Exception as e:
            print(f"Mesh validation warning: {e}")
            return mesh
