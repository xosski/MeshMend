from __future__ import annotations

from .assets import ModularAssetProvider, ModularMiniaturePart, PartCategory
from .pipeline import StudioMiniaturePipeline, StudioMiniatureSpec, generate_studio_miniature
from .quality import MiniatureArtifactDetector, MiniatureQualityCritic, MiniatureSculptQualityGate, StudioQualityGate, StudioQualityReport
from .staged_pipeline import ArchetypeCandidate, CharacterArchetypeGenerator, ConceptGenerator, MannequinDetector, MiniatureBlueprintGenerator, MiniatureConceptDesign, MiniatureDirector, PreSculptRecognizabilityGate, ProceduralMiniaturePartProvider, ResinMiniatureCritic, ShapeLanguageEngine, ShapeLanguageProfile, SilhouetteCritic, StagedMiniaturePipeline, StudioShapeAIPlanner, silhouette_similarity_signature, write_black_silhouette_previews
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
    "ArchetypeCandidate",
    "CharacterArchetypeGenerator",
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
    "silhouette_similarity_signature",
    "write_black_silhouette_previews",
    "StudioMiniaturePipeline",
    "StudioMiniatureSpec",
    "StudioQualityGate",
    "StudioQualityReport",
    "generate_studio_miniature",
]
