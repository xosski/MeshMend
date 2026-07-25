from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from meshmend_ai.assistant import MeshMendAssistant
    from meshmend_ai.generative_model import Local3DGenerativeModel
    from meshmend_ai.high_resolution_latent import LocalMeshLatentGenerator
    from meshmend_ai.neural_diffusion import Neural3DDiffusionModel, NeuralTrainingConfig
    from meshmend_ai.repair import RepairOptions, repair_stl
    from meshmend_ai.sculptor import get_sculptor_foundation
else:
    from .assistant import MeshMendAssistant
    from .generative_model import Local3DGenerativeModel
    from .high_resolution_latent import LocalMeshLatentGenerator
    from .neural_diffusion import Neural3DDiffusionModel, NeuralTrainingConfig
    from .repair import RepairOptions, repair_stl
    from .sculptor import get_sculptor_foundation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meshmend",
        description="Repair STL files for 3D printing and optionally bridge detached mesh islands.",
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input STL/mesh file. Omit input/output to open the GUI.")
    parser.add_argument("output", nargs="?", type=Path, help="Output STL file. Omit input/output to open the GUI.")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the MeshMend AI repair GUI for importing, repairing, and saving a model.",
    )
    parser.add_argument(
        "--bridge-disconnected",
        action="store_true",
        help="Add printable cylindrical connectors between disconnected components.",
    )
    parser.add_argument(
        "--fix-overhangs",
        action="store_true",
        help="Alias for --bridge-disconnected for detached/floating pieces that should attach to the model.",
    )
    parser.add_argument(
        "--assistant",
        action="store_true",
        help="Use the MeshMend assistant to inspect the model, choose connector settings, and explain the repair.",
    )
    parser.add_argument(
        "--perseus",
        action="store_true",
        help="When used with --assistant, run the explanation through the local Perseus memory orchestrator if available.",
    )
    parser.add_argument(
        "--sculptor",
        action="store_true",
        help="Launch the bundled 3D Sculptor app used as MeshMend's model-creation foundation.",
    )
    parser.add_argument(
        "--model-service",
        action="store_true",
        help="Run MeshMend's local model service in this terminal.",
    )
    parser.add_argument(
        "--stop-model-service",
        action="store_true",
        help="Stop any process listening on MeshMend's model-service port and exit.",
    )
    parser.add_argument(
        "--restart-model-service",
        action="store_true",
        help="If a model service is already running on the target port, stop it before starting a new one with the current CLI settings.",
    )
    parser.add_argument(
        "--model-service-port",
        type=int,
        default=8090,
        help="Port for MeshMend's local model service. Default: 8090",
    )
    parser.add_argument(
        "--check-hunyuan",
        action="store_true",
        help="Check which Python MeshMend will use for Hunyuan3D and whether it can import hy3dgen.",
    )
    parser.add_argument(
        "--free-local-hunyuan",
        action="store_true",
        help="Explicitly configure generation to use the free local Hunyuan3D backend with no API key. MeshMend native is the default for GUI/model-service launches.",
    )
    parser.add_argument(
        "--no-free-local-hunyuan",
        action="store_true",
        help="Do not auto-configure the free local Hunyuan3D backend for creation workflows.",
    )
    parser.add_argument(
        "--native-sculpt-backend",
        action="store_true",
        help="Use the experimental native MiniatureSpec -> rig -> sculpt backend. Not certified for store/studio-quality output yet.",
    )
    parser.add_argument(
        "--sculpt-planner-command",
        default=None,
        help="Optional AI/vision planner command for native sculpt. Receives {image_path}, {prompt_path}, {schema_path}, {plan_path}, {output_dir}, {local_plan_path}.",
    )
    parser.add_argument(
        "--require-ai-sculpt-planner",
        action="store_true",
        help="Require the configured AI/vision sculpt planner for high-detail native sculpt jobs instead of falling back to heuristics.",
    )
    parser.add_argument(
        "--hunyuan3d-path",
        type=Path,
        default=None,
        help="Path to a local Hunyuan3D-2/2.1 checkout. Sets MESHMEND_HUNYUAN3D_PATH.",
    )
    parser.add_argument(
        "--hunyuan3d-model",
        default="tencent/Hunyuan3D-2",
        help="Hunyuan3D Hugging Face model id/local path for --free-local-hunyuan. Default: tencent/Hunyuan3D-2",
    )
    parser.add_argument(
        "--hunyuan3d-subfolder",
        default="hunyuan3d-dit-v2-0",
        help="Hunyuan3D model subfolder for --free-local-hunyuan. Default: hunyuan3d-dit-v2-0",
    )
    parser.add_argument(
        "--hunyuan3d-output-format",
        default="stl",
        choices=("glb", "obj", "stl", "ply", "3mf", "fbx", "usdz"),
        help="Model format emitted by the local Hunyuan3D runner. Default: stl",
    )
    parser.add_argument(
        "--hunyuan3d-python",
        type=Path,
        default=None,
        help="Python executable for the Hunyuan3D worker. Auto-detected from the Hunyuan3D venv when possible.",
    )
    parser.add_argument(
        "--store-quality-text-command",
        default=None,
        help="Certified external text-to-3D command for store-quality generation. Receives {input_json}, {prompt_path}, {output_dir}, {quality}, {target_polycount}.",
    )
    parser.add_argument(
        "--store-quality-image-command",
        default=None,
        help="Certified external image-to-3D command for store-quality generation. Receives {input_json}, {prompt_path}, {image_path}, {output_dir}, {quality}, {target_polycount}.",
    )
    parser.add_argument(
        "--certify-store-quality-backend",
        action="store_true",
        help="Mark the configured external production command as certified for store/studio-quality 8K miniature generation.",
    )
    parser.add_argument(
        "--store-quality-config",
        type=Path,
        default=None,
        help="JSON config file for a certified store-quality backend. See store_quality_backend.template.json.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Train MeshMend's local 3D generative model from a directory of STL/OBJ/PLY files and optional images.",
    )
    parser.add_argument(
        "--train-neural-dir",
        type=Path,
        default=None,
        help="Train the PyTorch voxel autoencoder + latent 3D diffusion model from STL/OBJ/PLY files.",
    )
    parser.add_argument("--neural-resolution", type=int, default=96, help="Voxel resolution for --train-neural-dir. Default: 96; this model is coarse and not final tabletop detail")
    parser.add_argument("--neural-ae-epochs", type=int, default=50, help="Autoencoder epochs for --train-neural-dir. Default: 50")
    parser.add_argument("--neural-diffusion-epochs", type=int, default=90, help="Diffusion epochs for --train-neural-dir. Default: 90")
    parser.add_argument(
        "--connector-radius",
        type=float,
        default=0.75,
        help="Radius, in model units, for generated component bridges. Default: 0.75",
    )
    parser.add_argument(
        "--connector-sections",
        type=int,
        default=16,
        help="Number of radial sections for generated bridges. Default: 16",
    )
    parser.add_argument(
        "--max-bridge-distance",
        type=float,
        default=None,
        help="Skip bridge creation for components farther apart than this distance.",
    )
    parser.add_argument(
        "--merge-digits",
        type=int,
        default=6,
        help="Vertex welding precision passed to trimesh.merge_vertices. Default: 6",
    )
    parser.add_argument(
        "--max-hole-edges",
        type=int,
        default=80,
        help="Largest boundary loop MeshMend will cap automatically. Default: 80",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a JSON repair report after writing the output file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_generation_backend(args, _should_auto_configure_hunyuan(args))
    if args.restart_model_service:
        os.environ["MESHMEND_RESTART_MODEL_SERVICE"] = "1"
    if args.stop_model_service:
        return _stop_model_service_command(args.model_service_port)
    if args.check_hunyuan:
        return _check_hunyuan_install()
    if args.model_service:
        return _run_model_service()
    if args.sculptor:
        get_sculptor_foundation().launch()
        print("Launched bundled 3D Sculptor model creator.")
        return 0
    if args.train_dir is not None:
        result = Local3DGenerativeModel.train_from_directory(
            args.train_dir,
            progress=lambda percent, message: print(f"{percent}% - {message}"),
        )
        mesh_latent = LocalMeshLatentGenerator.train_from_directory(
            args.train_dir,
            progress=lambda percent, message: print(f"{percent}% - {message}"),
        )
        print(json.dumps({
            "checkpoint_path": result.checkpoint_path,
            "examples": result.examples,
            "images": result.images,
            "mesh_latent_checkpoint_path": mesh_latent.checkpoint_path,
            "mesh_latent_assets": mesh_latent.assets,
        }, indent=2))
        return 0
    if args.train_neural_dir is not None:
        config = NeuralTrainingConfig(
            resolution=args.neural_resolution,
            autoencoder_epochs=args.neural_ae_epochs,
            diffusion_epochs=args.neural_diffusion_epochs,
        )
        result = Neural3DDiffusionModel.train_from_directory(
            args.train_neural_dir,
            config=config,
            progress=lambda percent, message: print(f"{percent}% - {message}"),
        )
        print(json.dumps({
            "checkpoint_path": result.checkpoint_path,
            "manifest_path": result.manifest_path,
            "checkpoint_size_bytes": result.checkpoint_size_bytes,
            "data_signature": result.data_signature,
            "examples": result.examples,
            "resolution": result.resolution,
        }, indent=2))
        return 0
    if args.gui or (args.input is None and args.output is None):
        if __package__ in {None, ""}:
            from meshmend_ai.gui import launch_gui
        else:
            from .gui import launch_gui

        return launch_gui()
    if args.input is None or args.output is None:
        parser.error("input and output are both required for command-line repair; omit both or use --gui to open the GUI")

    if args.assistant:
        assistant = MeshMendAssistant(enable_perseus=args.perseus)
        result = assistant.repair(
            args.input,
            args.output,
            connector_radius=None if args.connector_radius == 0.75 else args.connector_radius,
            connector_sections=args.connector_sections,
            max_bridge_distance=args.max_bridge_distance,
            merge_digits=args.merge_digits,
            max_hole_edges=args.max_hole_edges,
            force_bridge=args.bridge_disconnected or args.fix_overhangs,
        )
        if args.report:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(result.explanation)
        return 0

    options = RepairOptions(
        bridge_disconnected=args.bridge_disconnected or args.fix_overhangs,
        connector_radius=args.connector_radius,
        connector_sections=args.connector_sections,
        max_bridge_distance=args.max_bridge_distance,
        merge_digits=args.merge_digits,
        max_hole_edges=args.max_hole_edges,
    )
    report = repair_stl(args.input, args.output, options)

    if args.report:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(
            f"Wrote {args.output} | watertight: {report.watertight_before} -> "
            f"{report.watertight_after} | components: {report.components_before} -> "
            f"{report.components_after} | bridges added: {report.bridges_added}"
        )
    return 0


def _should_auto_configure_hunyuan(args: argparse.Namespace) -> bool:
    if args.native_sculpt_backend:
        return True
    if args.no_free_local_hunyuan:
        return False
    if args.free_local_hunyuan:
        return True
    configured_engine = os.environ.get("MESHMEND_PRODUCTION_ENGINE", "").strip().lower()
    has_external_command = bool(
        os.environ.get("MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND", "").strip()
        or os.environ.get("MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND", "").strip()
    )
    if configured_engine in {"external", "command"} or (configured_engine and configured_engine not in {"free_local", "free_local_hunyuan", "hunyuan", "hunyuan3d"}):
        return False
    if has_external_command and os.environ.get("MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    # Creation workflows default to learned Hunyuan geometry. The procedural
    # MeshMend generators are useful for blockouts and repair tests, but cannot
    # create a studio sculpt merely by subdividing and tagging primitive parts.
    # Plain command-line repair should not start any model-generation stack.
    return bool(args.model_service or args.sculptor or args.gui or (args.input is None and args.output is None))


def _configure_generation_backend(args: argparse.Namespace, use_hunyuan: bool) -> None:
    if args.store_quality_config is not None:
        _configure_store_quality_backend_from_file(args.store_quality_config, args.model_service_port)
        return
    if args.store_quality_text_command or args.store_quality_image_command:
        os.environ["MESHMEND_PRODUCTION_ENGINE"] = "external"
        os.environ.setdefault("MESHMEND_USE_HOSTED_3D", "1")
        os.environ.setdefault("MESHMEND_HOSTED_3D_PROVIDER", "meshmend")
        os.environ.setdefault("MESHMEND_MODEL_SERVICE_URL", f"http://127.0.0.1:{args.model_service_port}")
        os.environ.setdefault("MESHMEND_MODEL_SERVICE_PORT", str(args.model_service_port))
        os.environ["MESHMEND_MODEL_WORKER_PYTHON"] = sys.executable
        os.environ["MESHMEND_MODEL_SERVICE_PYTHON"] = sys.executable
        if args.store_quality_text_command:
            os.environ["MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND"] = str(args.store_quality_text_command)
        if args.store_quality_image_command:
            os.environ["MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND"] = str(args.store_quality_image_command)
        if args.certify_store_quality_backend:
            os.environ["MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED"] = "1"
        return
    if _configure_bundled_no_api_external_backend(args):
        return
    if not use_hunyuan:
        return
    if args.native_sculpt_backend:
        os.environ["MESHMEND_PRODUCTION_ENGINE"] = "meshmend_sculpt"
        os.environ["MESHMEND_ALLOW_EXPERIMENTAL_SCULPT_HIGH_DETAIL"] = "1"
        os.environ["MESHMEND_ALLOW_UNCERTIFIED_STORE_QUALITY_OUTPUT"] = "1"
        if args.sculpt_planner_command:
            os.environ["MESHMEND_SCULPT_PLANNER_PROVIDER"] = "command"
            os.environ["MESHMEND_SCULPT_PLANNER_COMMAND"] = str(args.sculpt_planner_command)
        if args.require_ai_sculpt_planner:
            os.environ["MESHMEND_REQUIRE_AI_SCULPT_PLANNER"] = "1"
    else:
        os.environ["MESHMEND_PRODUCTION_ENGINE"] = "free_local_hunyuan"
    os.environ.setdefault("MESHMEND_USE_HOSTED_3D", "1")
    os.environ.setdefault("MESHMEND_HOSTED_3D_PROVIDER", "meshmend")
    os.environ.setdefault("MESHMEND_MODEL_SERVICE_URL", f"http://127.0.0.1:{args.model_service_port}")
    os.environ.setdefault("MESHMEND_MODEL_SERVICE_PORT", str(args.model_service_port))
    if args.native_sculpt_backend:
        os.environ["MESHMEND_MODEL_WORKER_PYTHON"] = sys.executable
        os.environ["MESHMEND_MODEL_SERVICE_PYTHON"] = sys.executable
        os.environ.pop("MESHMEND_HUNYUAN3D_PATH", None)
        os.environ.pop("MESHMEND_HUNYUAN3D_MODEL", None)
        os.environ.pop("MESHMEND_HUNYUAN3D_SUBFOLDER", None)
        return
    os.environ["MESHMEND_HUNYUAN3D_MODEL"] = str(args.hunyuan3d_model)
    os.environ["MESHMEND_HUNYUAN3D_SUBFOLDER"] = str(args.hunyuan3d_subfolder)
    os.environ["MESHMEND_HUNYUAN3D_OUTPUT_FORMAT"] = str(args.hunyuan3d_output_format)
    hunyuan_path = _resolve_hunyuan3d_path(args.hunyuan3d_path)
    if hunyuan_path is not None:
        os.environ["MESHMEND_HUNYUAN3D_PATH"] = str(hunyuan_path)
    worker_python = _resolve_hunyuan_worker_python(args.hunyuan3d_python, hunyuan_path)
    if worker_python is not None:
        os.environ["MESHMEND_MODEL_WORKER_PYTHON"] = str(worker_python)
    # Keep the FastAPI service on the current MeshMend interpreter by default;
    # only the heavy Hunyuan worker needs the Hunyuan environment.
    os.environ.setdefault("MESHMEND_MODEL_SERVICE_PYTHON", sys.executable)


def _configure_bundled_no_api_external_backend(args: argparse.Namespace) -> bool:
    # This runner is a procedural modular sculpt pipeline. It used to be
    # auto-selected and advertised as a certified external backend, allowing
    # high-face-count primitive blockouts to masquerade as studio output. Keep
    # it available only as an explicit development opt-in; normal creation uses
    # the learned Hunyuan path and fails closed if that backend is unavailable.
    if os.environ.get("MESHMEND_USE_BUNDLED_PROCEDURAL_BACKEND", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    if os.environ.get("MESHMEND_FORCE_NATIVE_SCULPT_BACKEND", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if args.free_local_hunyuan:
        return False
    if not (args.model_service or args.sculptor or args.gui or args.native_sculpt_backend):
        return False
    runner = Path(__file__).resolve().parent / "external_local_store_quality_generator.py"
    if not runner.exists():
        return False
    os.environ["MESHMEND_PRODUCTION_ENGINE"] = "external"
    os.environ["MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED"] = "0"
    os.environ.setdefault("MESHMEND_USE_HOSTED_3D", "1")
    os.environ.setdefault("MESHMEND_HOSTED_3D_PROVIDER", "meshmend")
    os.environ.setdefault("MESHMEND_MODEL_SERVICE_URL", f"http://127.0.0.1:{args.model_service_port}")
    os.environ.setdefault("MESHMEND_MODEL_SERVICE_PORT", str(args.model_service_port))
    os.environ["MESHMEND_MODEL_WORKER_PYTHON"] = sys.executable
    os.environ["MESHMEND_MODEL_SERVICE_PYTHON"] = sys.executable
    os.environ.setdefault("MESHMEND_MODEL_COMMAND_TIMEOUT_SECONDS", "10800")
    os.environ.setdefault("MESHMEND_MODEL_STALLED_TIMEOUT_SECONDS", "3600")
    os.environ.setdefault("MESHMEND_ALLOW_LOCAL_QUALITY_SCORE_ESTIMATES", "0")
    os.environ["MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND"] = (
        f'"{sys.executable}" "{runner}" --input {{input_json}} --prompt {{prompt_path}} '
        "--output-dir {output_dir} --quality {quality} --target-polycount {target_polycount}"
    )
    os.environ["MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND"] = (
        f'"{sys.executable}" "{runner}" --input {{input_json}} --prompt {{prompt_path}} --image {{image_path}} '
        "--output-dir {output_dir} --quality {quality} --target-polycount {target_polycount}"
    )
    return True


def _configure_store_quality_backend_from_file(config_path: Path, model_service_port: int) -> None:
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Store-quality backend config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_command = str(config.get("text_to_3d_command") or "").strip()
    image_command = str(config.get("image_to_3d_command") or "").strip()
    if not text_command and not image_command:
        raise ValueError("Store-quality backend config must define text_to_3d_command and/or image_to_3d_command")
    os.environ["MESHMEND_PRODUCTION_ENGINE"] = "external"
    os.environ["MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED"] = "1" if bool(config.get("certified", True)) else "0"
    os.environ.setdefault("MESHMEND_USE_HOSTED_3D", "1")
    os.environ.setdefault("MESHMEND_HOSTED_3D_PROVIDER", "meshmend")
    os.environ.setdefault("MESHMEND_MODEL_SERVICE_URL", f"http://127.0.0.1:{model_service_port}")
    os.environ.setdefault("MESHMEND_MODEL_SERVICE_PORT", str(model_service_port))
    os.environ["MESHMEND_MODEL_WORKER_PYTHON"] = str(config.get("worker_python") or sys.executable)
    os.environ["MESHMEND_MODEL_SERVICE_PYTHON"] = sys.executable
    if text_command:
        os.environ["MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND"] = text_command
    if image_command:
        os.environ["MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND"] = image_command
    for env_block_name in ("no_api_local_required_env", "http_generator_required_env", "environment"):
        env_block = config.get(env_block_name) or {}
        if isinstance(env_block, dict):
            for key, value in env_block.items():
                value_text = str(value or "").strip()
                if not key or not value_text or value_text.lower().startswith("optional ") or "your-" in value_text.lower():
                    continue
                os.environ[str(key)] = value_text
    validation = config.get("validation") or {}
    if isinstance(validation, dict):
        for key, env_name in (
            ("min_face_ratio", "MESHMEND_CERTIFIED_MIN_FACE_RATIO"),
            ("max_components", "MESHMEND_CERTIFIED_MAX_COMPONENTS"),
            ("min_depth_ratio", "MESHMEND_CERTIFIED_MIN_DEPTH_RATIO"),
        ):
            if key in validation:
                os.environ[env_name] = str(validation[key])


def _resolve_hunyuan3d_path(explicit_path: Path | None) -> Path | None:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    env_path = os.environ.get("MESHMEND_HUNYUAN3D_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("D:/models/Hunyuan3D-2"),
            Path("D:/models/Hunyuan3D-2.1"),
            Path.home() / "models" / "Hunyuan3D-2",
            Path.home() / "models" / "Hunyuan3D-2.1",
            Path.home() / "Hunyuan3D-2",
            Path.home() / "Hunyuan3D-2.1",
            Path.cwd() / "Hunyuan3D-2",
            Path.cwd() / "Hunyuan3D-2.1",
        ]
    )
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        if (resolved / "hy3dgen").exists() or (resolved / "setup.py").exists() or (resolved / "pyproject.toml").exists():
            return resolved
    return None


def _resolve_hunyuan_worker_python(explicit_python: Path | None, hunyuan_path: Path | None) -> Path | None:
    candidates: list[Path] = []
    if explicit_python is not None:
        candidates.append(explicit_python)
    env_python = os.environ.get("MESHMEND_MODEL_WORKER_PYTHON", "").strip()
    if env_python:
        candidates.append(Path(env_python))
    if hunyuan_path is not None:
        candidates.extend(
            [
                hunyuan_path / ".venv" / "Scripts" / "python.exe",
                hunyuan_path / "venv" / "Scripts" / "python.exe",
                hunyuan_path / "env" / "Scripts" / "python.exe",
                hunyuan_path / ".venv" / "bin" / "python",
                hunyuan_path / "venv" / "bin" / "python",
            ]
        )
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _check_hunyuan_install() -> int:
    worker_python = os.environ.get("MESHMEND_MODEL_WORKER_PYTHON", "").strip() or sys.executable
    hunyuan_path = os.environ.get("MESHMEND_HUNYUAN3D_PATH", "").strip()
    print(f"MeshMend Python: {sys.executable}")
    print(f"Hunyuan worker Python: {worker_python}")
    print(f"Hunyuan path: {hunyuan_path or '(not detected/set)'}")
    print(f"Production engine: {os.environ.get('MESHMEND_PRODUCTION_ENGINE', '(not set)')}")

    script = (
        "import sys\n"
        "print('python=' + sys.executable)\n"
        "try:\n"
        "    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline\n"
        "    print('hy3dgen=ok')\n"
        "except Exception as exc:\n"
        "    print('hy3dgen=failed: ' + repr(exc))\n"
        "    raise SystemExit(1)\n"
    )
    env = os.environ.copy()
    if hunyuan_path:
        env["PYTHONPATH"] = hunyuan_path + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run([worker_python, "-c", script], text=True, env=env)
    if completed.returncode == 0:
        print("Hunyuan3D import check passed.")
        return 0
    print(
        "Hunyuan3D import check failed. Install the official Hunyuan3D repo into the worker Python, "
        "or pass --hunyuan3d-python path\\to\\python.exe and --hunyuan3d-path path\\to\\Hunyuan3D-2."
    )
    return int(completed.returncode)


def _run_model_service() -> int:
    service_script = Path(__file__).resolve().parent / "3dsculpter" / "model_service" / "main.py"
    if not service_script.exists():
        raise FileNotFoundError(f"MeshMend model service not found: {service_script}")
    env = os.environ.copy()
    service_url = env.get("MESHMEND_MODEL_SERVICE_URL", "http://127.0.0.1:8090").rstrip("/")
    if env.get("MESHMEND_RESTART_MODEL_SERVICE") == "1":
        port = int(env.get("MESHMEND_MODEL_SERVICE_PORT", "8090"))
        stopped = _stop_processes_on_port(port)
        if stopped:
            print(f"Stopped existing MeshMend model service process(es) on port {port}: {', '.join(map(str, stopped))}")
            _wait_for_port_to_close(port)
        else:
            print(f"No existing process was listening on port {port}; starting a new service.")
    existing_health = _read_model_service_health(service_url)
    if existing_health is not None:
        desired_engine = env.get("MESHMEND_PRODUCTION_ENGINE", "meshmend_native")
        running_engine = str(existing_health.get("production_engine") or "")
        if desired_engine and running_engine and desired_engine.lower() != running_engine.lower():
            print("MeshMend model service is already running, but with a different production engine.")
            print(f"Running engine: {running_engine}")
            print(f"Requested engine: {desired_engine}")
            print("Restart it with --restart-model-service so the new engine is used.")
            return 1
        desired_path = env.get("MESHMEND_HUNYUAN3D_PATH", "")
        running_path = str(existing_health.get("hunyuan3d_path") or "")
        desired_worker = env.get("MESHMEND_MODEL_WORKER_PYTHON", "")
        running_worker = str(existing_health.get("model_worker_python") or "")
        if desired_path and running_path and desired_path.lower() != running_path.lower():
            print("MeshMend model service is already running, but with a different Hunyuan3D path.")
            print(f"Running path: {running_path}")
            print(f"Requested path: {desired_path}")
            print("Restart it with --restart-model-service so the new path is used.")
            return 1
        if desired_worker and running_worker and desired_worker.lower() != running_worker.lower():
            print("MeshMend model service is already running, but with a different Hunyuan worker Python.")
            print(f"Running worker: {running_worker}")
            print(f"Requested worker: {desired_worker}")
            print("Restart it with --restart-model-service so the new worker Python is used.")
            return 1
        print(f"MeshMend model service is already running at {service_url}")
        print(json.dumps(existing_health, indent=2))
        return 0
    package_parent = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = package_parent + os.pathsep + env.get("PYTHONPATH", "")
    print(f"Starting MeshMend model service: {service_script}")
    print(f"Production engine: {env.get('MESHMEND_PRODUCTION_ENGINE', 'meshmend_native')}")
    if env.get("MESHMEND_HUNYUAN3D_PATH"):
        print(f"Hunyuan3D path: {env['MESHMEND_HUNYUAN3D_PATH']}")
    if env.get("MESHMEND_MODEL_WORKER_PYTHON"):
        print(f"Hunyuan worker Python: {env['MESHMEND_MODEL_WORKER_PYTHON']}")
    completed = subprocess.run([sys.executable, str(service_script)], cwd=str(service_script.parent), env=env)
    return int(completed.returncode)


def _stop_model_service_command(port: int) -> int:
    stopped = _stop_processes_on_port(port)
    if not stopped:
        print(f"No process was listening on port {port}.")
        return 0
    _wait_for_port_to_close(port)
    print(f"Stopped process(es) on port {port}: {', '.join(map(str, stopped))}")
    return 0


def _read_model_service_health(service_url: str) -> dict | None:
    try:
        with urllib.request.urlopen(service_url.rstrip("/") + "/health", timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _stop_processes_on_port(port: int) -> list[int]:
    if os.name != "nt":
        return []
    try:
        completed = subprocess.run(
            ["netstat", "-ano"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return []
    pids: set[int] = set()
    marker = f":{port}"
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        local_address = parts[1]
        state = parts[3] if len(parts) >= 5 else ""
        pid_text = parts[-1]
        if marker in local_address and state.upper() == "LISTENING":
            try:
                pids.add(int(pid_text))
            except ValueError:
                pass
    stopped: list[int] = []
    for pid in sorted(pids):
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            if result.returncode == 0:
                stopped.append(pid)
        except Exception:
            pass
    return stopped


def _wait_for_port_to_close(port: int, timeout_seconds: float = 10.0) -> bool:
    import time

    deadline = time.time() + timeout_seconds
    service_url = f"http://127.0.0.1:{port}"
    while time.time() < deadline:
        if _read_model_service_health(service_url) is None:
            return True
        time.sleep(0.25)
    return False


if __name__ == "__main__":
    raise SystemExit(main())
