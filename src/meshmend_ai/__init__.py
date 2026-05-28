"""MeshMend AI STL repair utilities."""

from .assistant import AssistantPlan, AssistantResult, MeshMendAssistant
from .detail_quality import Detail8KReport, assess_8k_detail, ensure_8k_detail, ensure_high_resolution_detail
from .generative_model import Local3DGenerativeModel, TrainingResult, default_training_data_dir
from .neural_diffusion import Neural3DDiffusionModel, NeuralTrainingConfig, NeuralTrainingResult
from .repair import RepairOptions, RepairReport, repair_stl
from .sculptor import SculptorFoundation, get_sculptor_foundation

__all__ = [
    "AssistantPlan",
    "AssistantResult",
    "Detail8KReport",
    "Local3DGenerativeModel",
    "MeshMendAssistant",
    "Neural3DDiffusionModel",
    "NeuralTrainingConfig",
    "NeuralTrainingResult",
    "RepairOptions",
    "RepairReport",
    "SculptorFoundation",
    "TrainingResult",
    "assess_8k_detail",
    "default_training_data_dir",
    "ensure_8k_detail",
    "ensure_high_resolution_detail",
    "get_sculptor_foundation",
    "repair_stl",
]
