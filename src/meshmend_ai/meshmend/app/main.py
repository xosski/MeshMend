from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from meshmend.ai import GenerationRequest, get_adapter
from meshmend.core import add_circular_base, auto_scale_to_height, build_printability_report, decimate_mesh, load_mesh, remesh_subdivide
from meshmend.export import export_slicer_ready
from meshmend.repair import repair_mesh
from meshmend.studio import StagedMiniaturePipeline, StudioMiniatureSpec


SCALE_PRESETS = {"28mm": 28.0, "32mm": 32.0, "75mm": 75.0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MeshMend local desktop miniature repair/generation MVP")
    parser.add_argument("--cli", action="store_true", help="Run command-line workflow instead of desktop UI")
    parser.add_argument("--input", type=Path, help="Input STL/OBJ/GLB/PLY")
    parser.add_argument("--output", type=Path, help="Output mesh path")
    parser.add_argument("--repair", action="store_true", help="Repair input mesh")
    parser.add_argument("--scale", choices=sorted(SCALE_PRESETS), help="Auto-scale to miniature height")
    parser.add_argument("--add-base", action="store_true", help="Add circular base")
    parser.add_argument("--decimate", type=int, help="Target face count")
    parser.add_argument("--remesh", type=int, help="Subdivision target face count")
    parser.add_argument("--generate", type=str, help="Generate from a structured prompt with a local adapter")
    parser.add_argument("--studio-generate", type=str, help="Generate a gated offline studio/store-style miniature from a prompt")
    parser.add_argument("--studio-config", type=Path, help="Generate a gated offline studio/store-style miniature from a JSON spec")
    parser.add_argument("--detail-demo", action="store_true", help="Generate a procedural studio-detail demo armor plate scene")
    parser.add_argument("--target-faces", type=int, default=250_000, help="Target faces for draft high-detail studio generation/detail subdivision")
    parser.add_argument("--studio-candidates-dir", type=Path, help="Write per-category candidate part bundles for review/selection")
    parser.add_argument("--studio-candidates-per-category", type=int, default=3, help="Generate this many validated candidates per part category")
    parser.add_argument(
        "--adapter",
        default="existing",
        choices=("existing", "placeholder"),
        help="Local generation adapter. Default reuses the existing MeshMend SculptorFoundation path; placeholder is procedural test geometry.",
    )
    args = parser.parse_args(argv)

    if args.cli or args.input or args.generate or args.detail_demo:
        return run_cli(args)
    return run_desktop()


def run_cli(args: argparse.Namespace) -> int:
    if args.detail_demo:
        from meshmend.app.demo_scene import build_studio_detail_demo

        output = args.output or Path("meshmend_studio_detail_demo.stl")
        result = build_studio_detail_demo(output)
        print(json.dumps(result.summary, indent=2))
        return 0

    if args.studio_config or args.studio_generate:
        if args.studio_config:
            spec = StudioMiniatureSpec.from_json(args.studio_config)
        else:
            spec = StudioMiniatureSpec.from_prompt(
                args.studio_generate,
                scale_mm=SCALE_PRESETS.get(args.scale or "32mm", 32.0),
                target_faces=args.target_faces,
            )
        pipeline = StagedMiniaturePipeline()
        result = pipeline.generate(
            spec,
            candidates_per_category=args.studio_candidates_per_category,
            candidate_output_dir=args.studio_candidates_dir,
        )
        print(
            json.dumps(
                {
                    "studio_quality": result.quality_report.to_dict(),
                    "stages": [stage.to_dict() for stage in result.stage_results],
                    "selected_parts": {category.value: part.part_id for category, part in result.selected_parts.items()},
                    "spec": spec.to_dict(),
                },
                indent=2,
            )
        )
        if args.output:
            pipeline.export(
                spec,
                args.output,
                candidates_per_category=args.studio_candidates_per_category,
                candidate_output_dir=args.studio_candidates_dir,
            )
        return 0

    if args.generate:
        mesh = get_adapter(args.adapter).generate(
            GenerationRequest(prompt=args.generate, height_mm=SCALE_PRESETS.get(args.scale or "32mm", 32.0))
        )
    elif args.input:
        mesh = load_mesh(args.input)
    else:
        raise SystemExit("--input or --generate is required in --cli mode")

    if args.repair:
        mesh = repair_mesh(mesh).mesh
    if args.scale:
        mesh = auto_scale_to_height(mesh, SCALE_PRESETS[args.scale])
    if args.add_base:
        mesh = add_circular_base(mesh)
    if args.decimate:
        mesh = decimate_mesh(mesh, args.decimate)
    if args.remesh:
        mesh = remesh_subdivide(mesh, args.remesh)

    report = build_printability_report(mesh)
    print(json.dumps(report.to_dict(), indent=2))
    if args.output:
        export_slicer_ready(mesh, args.output)
    return 0


def run_desktop() -> int:
    try:
        from PySide6 import QtWidgets
    except Exception as exc:
        print("PySide6 is not installed. Install PySide6, or use `python -m meshmend --cli`.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2

    from meshmend.app.ui_main import MeshMendWindow

    app = QtWidgets.QApplication(sys.argv)
    window = MeshMendWindow()
    window.resize(1280, 820)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
