from pathlib import Path

import trimesh

from meshmend.ai import GenerationRequest, get_adapter
from meshmend.compliance.filters import sanitize_prompt
from meshmend.core import add_circular_base, auto_scale_to_height, build_printability_report, load_mesh
from meshmend.export import export_slicer_ready
from meshmend.repair import repair_mesh
from meshmend.studio import DetailCritic, MannequinDetector, PartCategory, MiniatureSculptQualityGate, SculptEngine, StagedMiniaturePipeline, StudioMiniaturePipeline, StudioMiniatureSpec, StudioQualityGate, silhouette_similarity_ratio, silhouette_similarity_signature, write_black_silhouette_previews


def test_repair_fills_simple_box_hole() -> None:
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    mesh.update_faces([i for i in range(len(mesh.faces)) if i != 0])
    before = build_printability_report(mesh)
    repaired = repair_mesh(mesh)
    assert before.holes > 0
    assert repaired.after.holes <= before.holes


def test_scale_and_base_change_dimensions() -> None:
    mesh = trimesh.creation.uv_sphere(radius=1.0)
    scaled = auto_scale_to_height(mesh, 32.0)
    based = add_circular_base(scaled, radius_mm=15.0, height_mm=2.0)
    assert 31.9 <= scaled.extents[2] <= 32.1
    assert based.extents[0] >= 29.9
    assert based.extents[2] > scaled.extents[2]


def test_export_and_reload_stl(tmp_path: Path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    output = export_slicer_ready(mesh, tmp_path / "mini.stl")
    assert output.exists()
    assert output.with_suffix(".stl.printability.json").exists()
    loaded = load_mesh(output)
    assert len(loaded.faces) > 0


def test_placeholder_generation_is_local_mesh() -> None:
    mesh = get_adapter("placeholder").generate(GenerationRequest(prompt="original armored knight with rifle", target_faces=5000))
    report = build_printability_report(mesh)
    assert len(mesh.faces) >= 1000
    assert report.dimensions_mm[2] > 30.0


def test_default_adapter_reuses_existing_meshmend_sculptor() -> None:
    assert get_adapter().name == "existing_meshmend_local_sculptor"


def test_compliance_sanitizes_direct_copy_terms() -> None:
    sanitized, warnings = sanitize_prompt("Warhammer space marine with aquila")
    assert "warhammer" not in sanitized.lower()
    assert "space marine" not in sanitized.lower()
    assert warnings


def test_studio_pipeline_generates_gated_heavy_infantry() -> None:
    spec = StudioMiniatureSpec.from_prompt(
        "original sci-fi heavy infantry with rifle backpack vents helmet lenses rivets panel lines",
        target_faces=250_000,
    )
    mesh, report = StudioMiniaturePipeline().generate(spec)
    assert report.passed
    assert len(mesh.faces) >= 250_000
    assert "weapon" in report.required_components_present
    assert report.sheet_artifacts == 0


def test_studio_quality_gate_rejects_large_sheet() -> None:
    sheet = trimesh.creation.box(extents=[40.0, 40.0, 0.08])
    sheet.metadata["studio_components"] = ["body", "head", "left_arm", "right_arm", "left_leg", "right_leg", "weapon"]
    report = StudioQualityGate(min_faces=1, min_vertices=1, max_boundary_edges=1000).evaluate(sheet)
    assert not report.passed
    assert any("sheet_artifacts" in issue or "too_flat" in issue for issue in report.issues)


def test_studio_pipeline_exports_quality_sidecars(tmp_path: Path) -> None:
    spec = StudioMiniatureSpec.from_prompt("original sci-fi heavy infantry rifle backpack vents", target_faces=250_000)
    output, report = StudioMiniaturePipeline().export(spec, tmp_path / "studio_mini.stl")
    assert report.passed
    assert output.exists()
    assert output.with_suffix(".stl.printability.json").exists()
    assert output.with_suffix(".stl.studio_quality.json").exists()
    assert output.with_suffix(".stl.studio_spec.json").exists()


def test_staged_pipeline_writes_selectable_part_candidates(tmp_path: Path) -> None:
    spec = StudioMiniatureSpec.from_prompt("original sci-fi heavy infantry rifle backpack vents", target_faces=250_000)
    candidates, stages = StagedMiniaturePipeline().generate_candidates(spec, candidates_per_category=2, output_dir=tmp_path / "candidates")
    assert all(stage.passed for stage in stages)
    assert len(candidates[PartCategory.HELMET]) == 2
    first = candidates[PartCategory.HELMET][0]
    bundle_dir = tmp_path / "candidates" / PartCategory.HELMET.value / first.part_id
    assert (bundle_dir / f"{first.part_id}.stl").exists()
    assert (bundle_dir / "preview.svg").exists()
    assert (bundle_dir / "metadata.json").exists()
    assert (bundle_dir / "cleanup_report.json").exists()


def test_staged_pipeline_assembles_validated_selected_parts(tmp_path: Path) -> None:
    spec = StudioMiniatureSpec.from_prompt("original sci-fi heavy infantry rifle backpack vents", target_faces=250_000)
    output, result = StagedMiniaturePipeline().export(spec, tmp_path / "staged_mini.glb", candidates_per_category=2)
    assert result.quality_report.passed
    assert output.exists()
    assert output.with_suffix(".glb.studio_stages.json").exists()
    assert output.with_suffix(".glb.studio_selection.json").exists()
    assert set(result.selected_parts) >= {
        PartCategory.HELMET,
        PartCategory.CHEST_ARMOR,
        PartCategory.SHOULDER_PADS,
        PartCategory.ARMS,
        PartCategory.LEGS,
        PartCategory.WEAPON,
        PartCategory.BACKPACK,
        PartCategory.BASE,
    }


def test_staged_pipeline_generates_distinct_pre_sculpt_archetypes(tmp_path: Path) -> None:
    prompts = {
        "high_elf": "High Elf Warrior with glaive layered fantasy armor cape heroic posture",
        "orc": "Orc Brute with cleaver hunched muscular body tusks crude armor",
        "dwarf": "Dwarf Warrior with axe runic heavy armor stocky proportions shield",
        "astra": "Astra Shock Trooper with helmet flak armor las rifle field pack braced firing advance",
        "terminator": "Space Terminator with massive exo plate armor huge pauldrons heavy storm rifle reactor backpack",
    }
    expected_roles = {
        "high_elf": "high_elf_warrior",
        "orc": "orc_brute",
        "dwarf": "dwarf_warrior",
        "astra": "astra_shock_trooper",
        "terminator": "space_terminator",
    }
    pipeline = StagedMiniaturePipeline()
    silhouettes: dict[str, tuple[float, float, float]] = {}
    signatures: dict[str, tuple[float, ...]] = {}
    for key, prompt in prompts.items():
        spec = StudioMiniatureSpec.from_prompt(prompt, target_faces=250_000)
        concept, _stage = pipeline.concept_profile(spec)
        assert concept["design"]["role"] == expected_roles[key]
        assert concept["ai_shape_plan"]["archetype"] == expected_roles[key]
        assert concept["ai_shape_plan"]["source"] in {"local_semantic_ai_shape_planner", "external_ai_shape_planner"}
        assert concept["ai_shape_plan"]["part_directives"]
        candidates, stages = pipeline.generate_candidates(spec, candidates_per_category=1)
        assert all(stage.passed for stage in stages)
        selected, selection_stage = pipeline.select_parts(candidates)
        assert selection_stage.passed
        mesh, _assembly_stage = pipeline.assemble(spec, selected)
        assert mesh.metadata.get("ai_shape_directives_applied")
        design = pipeline.archetype_generator.generate(spec)
        shape = pipeline.shape_language_engine.generate(spec, design)
        silhouette_stage = pipeline.silhouette_validation(mesh, design, shape)
        assert silhouette_stage.passed, silhouette_stage.issues
        preview_paths = write_black_silhouette_previews(mesh, tmp_path / "pre_sculpt_silhouettes", prefix=key)
        assert set(preview_paths) == {"front", "side", "rear", "45"}
        assert all(Path(path).exists() for path in preview_paths.values())
        components = set(mesh.metadata.get("studio_components", []))
        assert shape.archetype == expected_roles[key]
        assert set(shape.required_silhouette_tags).issubset(components)
        silhouettes[key] = tuple(round(float(value), 1) for value in mesh.extents)
        signatures[key] = silhouette_similarity_signature(mesh)
    assert len(set(silhouettes.values())) == len(silhouettes)
    assert len(set(signatures.values())) == len(signatures)
    for first_key, first_signature in signatures.items():
        for second_key, second_signature in signatures.items():
            if first_key >= second_key:
                continue
            distance = sum(abs(a - b) for a, b in zip(first_signature, second_signature))
            assert distance > 0.20, f"{first_key} and {second_key} share a mannequin-like silhouette"


def test_archetype_separation_fails_above_40_percent_similarity() -> None:
    prompts = [
        "High Elf Warrior with glaive layered fantasy armor cape heroic posture",
        "Orc Brute with cleaver hunched muscular body tusks crude armor",
        "Astra Shock Trooper with helmet flak armor las rifle field pack braced firing advance",
        "Human Knight with crested helm kite shield longsword tabard",
    ]
    pipeline = StagedMiniaturePipeline()
    meshes = []
    for prompt in prompts:
        spec = StudioMiniatureSpec.from_prompt(prompt, target_faces=250_000)
        candidates, stages = pipeline.generate_candidates(spec, candidates_per_category=1)
        assert all(stage.passed for stage in stages)
        selected, selection_stage = pipeline.select_parts(candidates)
        assert selection_stage.passed
        mesh, assembly_stage = pipeline.assemble(spec, selected)
        assert assembly_stage.passed
        meshes.append(mesh)
    for index, first in enumerate(meshes):
        for second in meshes[index + 1:]:
            assert silhouette_similarity_ratio(first, second) <= 0.40


def test_mannequin_detector_rejects_primitive_tagged_blockout() -> None:
    blockout = trimesh.creation.capsule(radius=4.0, height=20.0)
    blockout.metadata["studio_components"] = ["body", "head", "weapon", "mannequin_core", "primitive_capsule_limb"]
    design = StagedMiniaturePipeline().archetype_generator.generate(StudioMiniatureSpec.from_prompt("High Elf Warrior with glaive"))
    shape = StagedMiniaturePipeline().shape_language_engine.generate(StudioMiniatureSpec.from_prompt("High Elf Warrior with glaive"), design)
    passed, issues, metrics = MannequinDetector().evaluate(blockout, design, shape)
    assert not passed
    assert metrics["mannequin_score"] > 0.10
    assert any("MANNEQUIN_DETECTOR" in issue for issue in issues)


def test_miniature_sculpt_quality_gate_rejects_watertight_primitive_blockout() -> None:
    blockout = trimesh.creation.uv_sphere(radius=12.0, count=[96, 48])
    blockout.metadata["studio_components"] = ["body", "helmet", "chest_armor", "shoulder_pad", "arms", "legs", "backpack", "weapon", "base"]
    report = MiniatureSculptQualityGate(min_faces=1, min_vertices=1, min_height_mm=1.0, max_height_mm=30.0).evaluate(blockout)
    assert not report.passed
    assert any("large_smooth_primitive_surfaces_dominate" in issue for issue in report.issues)
    assert any("no_armor_seams_or_panel_lines_detected" in issue for issue in report.issues)


def test_sculpt_engine_creates_detail_maps_before_geometry() -> None:
    maps = SculptEngine().generate_detail_maps({"prompt": "original sci-fi veteran with armor trim vents cloth insignia"})
    assert maps.normal_map.shape == (512, 512, 3)
    assert maps.displacement_map.shape == (512, 512)
    assert {"armor_trim", "panel_lines", "cloth_folds", "surface_wear", "insignia"}.issubset(maps.detail_masks)
    assert float(maps.displacement_map.max()) > float(maps.displacement_map.min())


def test_foundation_first_sculpt_still_adds_readable_detail_geometry() -> None:
    base = trimesh.creation.uv_sphere(radius=8.0, count=[48, 24])
    base.metadata["studio_components"] = ["body", "head", "helmet", "weapon", "base", "high_elf_warrior_shape"]
    sculpted, report = SculptEngine(target_preoptimization_faces=80_000).sculpt(
        base,
        {
            "prompt": "High Elf Warrior with glaive layered armor cape leaf trim",
            "character_foundation_first": True,
        },
    )
    components = set(sculpted.metadata.get("studio_components", []))
    assert report.passed
    assert {"armor_trim", "panel_line", "rivet", "weapon_detail", "face_detail", "micro_engraving"}.issubset(components)
    assert report.critic_scores["overall"] >= 85.0


def test_detail_critic_rejects_smooth_blockout() -> None:
    blockout = trimesh.creation.uv_sphere(radius=12.0, count=[96, 48])
    blockout.metadata["studio_components"] = ["body", "helmet", "weapon", "base"]
    scores = DetailCritic().evaluate(blockout)
    issues = DetailCritic().issues(scores)
    assert "armor_surfaces_are_smooth" in issues
    assert "weapons_are_featureless" in issues
    assert "faces_lack_sculptural_features" in issues
    assert "model_resembles_blockout" in issues
