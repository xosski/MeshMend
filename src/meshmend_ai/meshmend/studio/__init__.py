from __future__ import annotations

from .assets import ModularAssetProvider, ModularMiniaturePart, PartCategory
from .pipeline import StudioMiniaturePipeline, StudioMiniatureSpec, generate_studio_miniature
from .quality import MiniatureArtifactDetector, MiniatureQualityCritic, MiniatureSculptQualityGate, StudioQualityGate, StudioQualityReport
from .staged_pipeline import ArchetypeCandidate, CharacterArchetypeGenerator, CharacterComponentLibraryProvider, ConceptGenerator, GenerationFailed, MannequinDetector, MiniatureBlueprintGenerator, MiniatureConceptDesign, MiniatureDirector, PreSculptRecognizabilityGate, ProceduralMiniaturePartProvider, ResinMiniatureCritic, ShapeLanguageEngine, ShapeLanguageProfile, SilhouetteCritic, StagedMiniaturePipeline, StudioShapeAIPlanner, VisionCritic, silhouette_similarity_ratio, silhouette_similarity_signature, write_black_silhouette_previews
from meshmend.sculpt import DetailCritic, DetailMapSet, SculptEngine, SculptEngineReport

__all__ = [
    "ModularAssetProvider",
    "DetailCritic",
    "DetailMapSet",
    "ModularMiniaturePart",
    "MiniatureSculptQualityGate",
    "MiniatureArtifactDetector",
    "MiniatureQualityCritic",
    "MannequinDetector",
    "MiniatureBlueprintGenerator",
    "ConceptGenerator",
    "GenerationFailed",
    "ArchetypeCandidate",
    "CharacterArchetypeGenerator",
    "CharacterComponentLibraryProvider",
    "MiniatureConceptDesign",
    "MiniatureDirector",
    "PreSculptRecognizabilityGate",
    "PartCategory",
    "ProceduralMiniaturePartProvider",
    "ResinMiniatureCritic",
    "ShapeLanguageEngine",
    "ShapeLanguageProfile",
    "SculptEngine",
    "SculptEngineReport",
    "SilhouetteCritic",
    "StagedMiniaturePipeline",
    "StudioShapeAIPlanner",
    "VisionCritic",
    "silhouette_similarity_ratio",
    "silhouette_similarity_signature",
    "write_black_silhouette_previews",
    "StudioMiniaturePipeline",
    "StudioMiniatureSpec",
    "StudioQualityGate",
    "StudioQualityReport",
    "generate_studio_miniature",
]
