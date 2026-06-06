from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


SUPPORTED_MODEL_SUFFIXES = {".stl", ".glb", ".obj", ".ply", ".3mf", ".fbx", ".usdz"}


def main() -> int:
    parser = argparse.ArgumentParser(description="MeshMend certified store-quality runner adapter")
    parser.add_argument("--input", required=True, help="MeshMend request JSON path")
    parser.add_argument("--prompt", required=True, help="Prompt text file path")
    parser.add_argument("--image", default="", help="Optional decoded input image path")
    parser.add_argument("--output-dir", required=True, help="Directory where final model/result.json must be written")
    parser.add_argument("--quality", default="high")
    parser.add_argument("--target-polycount", default="2000000")
    parser.add_argument(
        "--generator-command",
        default=os.environ.get("MESHMEND_STORE_QUALITY_GENERATOR_COMMAND", ""),
        help=(
            "Underlying certified generator command. May contain placeholders: "
            "{input_json}, {prompt_path}, {image_path}, {output_dir}, {quality}, {target_polycount}. "
            "If omitted, MESHMEND_STORE_QUALITY_GENERATOR_COMMAND is used."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator_command = str(args.generator_command or "").strip()
    if not generator_command:
        return fail(
            output_dir,
            "No certified generator command configured. Set MESHMEND_STORE_QUALITY_GENERATOR_COMMAND or pass --generator-command.",
            exit_code=2,
        )

    command = generator_command.format(
        input_json=str(Path(args.input)),
        prompt_path=str(Path(args.prompt)),
        image_path=str(Path(args.image)) if args.image else "",
        output_dir=str(output_dir),
        quality=str(args.quality),
        target_polycount=str(args.target_polycount),
    )
    completed = subprocess.run(
        shlex.split(command, posix=os.name != "nt"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=float(os.environ.get("MESHMEND_STORE_QUALITY_RUNNER_TIMEOUT_SECONDS", "7200")),
    )
    if completed.returncode != 0:
        return fail(output_dir, completed.stderr.strip() or completed.stdout.strip() or f"generator exited {completed.returncode}")

    try:
        result = load_generator_result(output_dir, completed.stdout)
    except Exception as exc:
        return fail(output_dir, f"Could not read generator result: {exc}")
    if not result.get("model_file"):
        return fail(output_dir, "Generator did not report a local model_file")
    model_path = output_dir / Path(str(result["model_file"])).name
    if not model_path.exists():
        return fail(output_dir, f"Generator reported missing model_file: {result['model_file']}")
    if model_path.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
        return fail(output_dir, f"Unsupported model format: {model_path.suffix}")

    result.update(
        {
            "model_file": model_path.name,
            "model_format": model_path.suffix.lower().lstrip("."),
            "provider": result.get("provider") or "certified_external_store_quality_runner",
            "capability_tier": "certified_store_quality_external",
            "geometry_source": result.get("geometry_source") or "certified_external_3d_generator",
            "store_quality_certified": True,
        }
    )
    result.setdefault("mesh_info", {})
    if isinstance(result["mesh_info"], dict):
        result["mesh_info"].setdefault("detail_source", "certified_external_sculpt_geometry")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0


def load_generator_result(output_dir: Path, stdout: str) -> dict:
    result_json = output_dir / "result.json"
    if result_json.exists():
        return json.loads(result_json.read_text(encoding="utf-8"))
    stripped = (stdout or "").strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    for path in output_dir.iterdir():
        if path.suffix.lower() in SUPPORTED_MODEL_SUFFIXES:
            return {"model_file": path.name, "model_format": path.suffix.lower().lstrip(".")}
    return {}


def fail(output_dir: Path, message: str, *, exit_code: int = 1) -> int:
    payload = {"error": message, "store_quality_certified": False}
    try:
        (output_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
    print(json.dumps(payload), file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
