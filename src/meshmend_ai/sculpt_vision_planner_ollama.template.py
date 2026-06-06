from __future__ import annotations

import argparse
import base64
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are MeshMend's vision planner. Return JSON only.
You do not generate meshes, polygons, STL, or code. You identify a concept image's subject and produce a structured miniature plan that MeshMend's native geometry backend can build.
Use only supported archetypes and part kinds from the schema. If a part is important but unsupported, put it in rig.unsupported_parts.
Prefer accurate subject identity over generic miniature language.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Ollama vision adapter for MeshMend native sculpt planner")
    parser.add_argument("--image", dest="image_path", default="")
    parser.add_argument("--prompt", dest="prompt_path", required=True)
    parser.add_argument("--schema", dest="schema_path", required=True)
    parser.add_argument("--out", dest="plan_path", required=True)
    args = parser.parse_args()

    image_path = Path(args.image_path) if args.image_path else None
    prompt = Path(args.prompt_path).read_text(encoding="utf-8") if Path(args.prompt_path).exists() else ""
    schema = json.loads(Path(args.schema_path).read_text(encoding="utf-8"))
    plan = call_ollama(prompt, image_path, schema)
    Path(args.plan_path).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan))
    return 0


def call_ollama(prompt: str, image_path: Path | None, schema: dict[str, Any]) -> dict[str, Any]:
    model = os.environ.get("MESHMEND_OLLAMA_VISION_MODEL", "qwen2.5vl:7b").strip() or "qwen2.5vl:7b"
    host = os.environ.get("MESHMEND_OLLAMA_URL", "http://127.0.0.1:11434/api/generate").strip()
    images = []
    if image_path is not None and image_path.exists():
        images.append(base64.b64encode(image_path.read_bytes()).decode("ascii"))
    body = {
        "model": model,
        "stream": False,
        "format": "json",
        "prompt": build_prompt(prompt, schema),
        "images": images,
        "options": {"temperature": 0.1},
    }
    request = urllib.request.Request(host, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=float(os.environ.get("MESHMEND_OLLAMA_TIMEOUT_SECONDS", "180"))) as response:
        payload = json.loads(response.read().decode("utf-8"))
    text = str(payload.get("response") or "")
    raw = parse_json_object(text)
    return normalize_plan(raw)


def build_prompt(prompt: str, schema: dict[str, Any]) -> str:
    return f"""{SYSTEM_PROMPT}

User prompt:
{prompt or '(none)'}

Supported schema:
{json.dumps(schema, indent=2)}

Return one JSON object with keys: spec, evidence, rig, details, blockers.
For concept art, identify the actual subject. Examples:
- hooded rider on reptilian mount with raised sword -> archetype mounted_beast, weapon sword, parts beast_body/beast_head/beast_leg/tail/rider/saddle/weapon/base/spikes/skull_or_rock_base_detail.
- humanoid soldier with rifle -> archetype armored_humanoid, weapon rifle.

Do not output markdown.
"""


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise RuntimeError("Ollama response did not contain a JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise RuntimeError("Ollama response JSON was not an object")
    return value


def normalize_plan(raw: dict[str, Any]) -> dict[str, Any]:
    spec = object_value(raw.get("spec"))
    evidence = object_value(raw.get("evidence"))
    rig = object_value(raw.get("rig"))
    details = object_value(raw.get("details"))
    archetype = str(spec.get("archetype") or evidence.get("archetype") or rig.get("archetype") or "armored_humanoid")
    visible_tokens = visible_schema_tokens(raw)
    detail_weapon = object_value(details.get("weapon"))
    weapon = str(spec.get("weapon") or detail_weapon.get("kind") or "sidearm")
    if weapon == "sidearm":
        for candidate in ("sword", "rifle", "axe", "staff", "spear", "lance"):
            if candidate in visible_tokens:
                weapon = candidate
                break
    spec.setdefault("archetype", archetype)
    spec.setdefault("scale_mm", 32)
    spec.setdefault("pose", evidence.get("pose") or "concept_pose")
    spec.setdefault("armor_style", "plate_armor")
    if not spec.get("weapon") or str(spec.get("weapon")).lower() == "sidearm":
        spec["weapon"] = weapon
    spec.setdefault("base_style", "round_scenic")
    spec.setdefault("detail_language", list(details.get("motifs") or []))
    spec.setdefault("proportions", {})
    spec.setdefault("concept_cues", [])
    evidence.setdefault("provider", "ollama_vision")
    evidence.setdefault("confidence", 0.7)
    evidence.setdefault("subject", archetype.replace("_", " "))
    evidence.setdefault("archetype", archetype)
    evidence.setdefault("pose", spec["pose"])
    evidence.setdefault("style_tags", spec["detail_language"])
    evidence.setdefault("detected_parts", list_value(evidence.get("detected_parts")))
    evidence.setdefault("silhouette", spec["proportions"])
    evidence.setdefault("warnings", list_value(evidence.get("warnings")))
    rig.setdefault("archetype", archetype)
    rig["parts"] = list_value(rig.get("parts")) or default_parts(archetype, weapon)
    rig["parts"] = augment_parts_from_visible_details(rig["parts"], spec, details, weapon)
    rig["pose_tags"] = list_value(rig.get("pose_tags")) or [spec["pose"]]
    rig["unsupported_parts"] = list_value(rig.get("unsupported_parts"))
    details.setdefault("style", ";".join(spec["detail_language"][:5]) if isinstance(spec["detail_language"], list) else archetype)
    details["materials"] = list_value(details.get("materials"))
    details["zones"] = list_value(details.get("zones"))
    details["motifs"] = list_value(details.get("motifs")) or (spec["detail_language"] if isinstance(spec["detail_language"], list) else [])
    details.setdefault("min_feature_mm", 0.18)
    details["surface_language"] = list_value(details.get("surface_language"))
    details["material_zones"] = list_value(details.get("material_zones"))
    details["required_landmarks"] = sorted(set(list_value(details.get("required_landmarks")) + infer_required_landmarks(spec, rig, details)))
    blockers = [str(item) for item in list_value(raw.get("blockers"))]
    if archetype in {"armored_humanoid", "orc_warrior", "mounted_beast", "armored_mech"}:
        blockers = [item for item in blockers if "unsupported" not in item.replace("-", "_").lower()]
    return {"spec": spec, "evidence": evidence, "rig": rig, "details": details, "blockers": blockers}


def object_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def infer_required_landmarks(spec: dict[str, Any], rig: dict[str, Any], details: dict[str, Any]) -> list[str]:
    landmarks: set[str] = set()
    weapon = str(spec.get("weapon") or "").lower()
    if weapon and weapon != "sidearm":
        landmarks.add(weapon)
    archetype = str(spec.get("archetype") or "").lower()
    if archetype == "mounted_beast":
        landmarks.update({"mounted_beast", "saddle", "rider"})
    for part in list_value(rig.get("parts")):
        if isinstance(part, dict):
            kind = str(part.get("kind") or "").lower()
            primitive = str(part.get("primitive") or "").lower()
            if kind in {"weapon", "shield", "cape", "tail", "horn", "saddle", "rider", "banner", "skull_or_rock_base_detail"}:
                landmarks.add(primitive if kind == "weapon" and primitive else kind)
    for motif in list_value(details.get("motifs")):
        text = str(motif).lower()
        for token in ("cape", "cloak", "scales", "claws", "saddle", "skull", "horn", "spike", "shield", "banner"):
            if token in text:
                landmarks.add("skull_or_rock_base_detail" if token == "skull" else token)
    for key in ("cape", "cloak", "shield", "banner", "saddle"):
        if isinstance(details.get(key), dict):
            landmarks.add("cape" if key == "cloak" else key)
    detail_weapon = object_value(details.get("weapon"))
    if detail_weapon.get("kind"):
        landmarks.add(str(detail_weapon["kind"]).lower())
    return sorted(landmarks)


def augment_parts_from_visible_details(parts: list[Any], spec: dict[str, Any], details: dict[str, Any], weapon: str) -> list[Any]:
    augmented = [part for part in parts if isinstance(part, dict)]
    kinds = {str(part.get("kind") or "").lower() for part in augmented}
    visible_tokens = visible_schema_tokens({"spec": spec, "details": details})
    part_kinds = set(visible_tokens)
    if "weapon" not in kinds:
        augmented.append({"id": "weapon", "kind": "weapon", "primitive": weapon or "sidearm", "anchor": "right_hand", "side": None, "center": [5.7, -1.2, 15.8], "scale": [1.2, 0.22, 2.4], "required": True, "confidence": 0.8})
    else:
        for part in augmented:
            if str(part.get("kind") or "").lower() == "weapon" and (not part.get("primitive") or str(part.get("primitive")).lower() == "sidearm") and weapon != "sidearm":
                part["primitive"] = weapon
    if ("cape" in part_kinds or isinstance(details.get("cape"), dict) or isinstance(details.get("cloak"), dict)) and "cape" not in kinds:
        augmented.append({"id": "cape", "kind": "cape", "primitive": "cloth", "anchor": "back", "side": None, "center": [0, 2.7, 11.2], "scale": [3.6, 0.42, 6.8], "required": True, "confidence": 0.75})
    if ("shield" in part_kinds or isinstance(details.get("shield"), dict)) and "shield" not in kinds:
        augmented.append({"id": "shield", "kind": "shield", "primitive": "shield", "anchor": "left_arm", "side": "left", "center": [-4.7, -1.25, 11.2], "scale": [1.45, 0.28, 2.4], "required": True, "confidence": 0.75})
    if ("banner" in part_kinds or isinstance(details.get("banner"), dict)) and "banner" not in kinds:
        augmented.append({"id": "banner", "kind": "banner", "primitive": "banner", "anchor": "back", "side": "right", "center": [5.8, -1.1, 12.0], "scale": [1.9, 0.14, 3.0], "required": True, "confidence": 0.7})
    return augmented


def visible_schema_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    supported = {
        "sword", "rifle", "axe", "staff", "spear", "lance", "shield", "cape", "cloak", "banner", "saddle", "tail",
        "horn", "spike", "spikes", "skull", "skulls", "scales", "claws", "helmet", "hood", "head", "torso", "weapon",
    }
    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                key_text = str(key).lower()
                if key_text in supported:
                    tokens.add("cape" if key_text == "cloak" else key_text)
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
        elif isinstance(item, str):
            text = item.lower()
            for token in supported:
                if token in text:
                    tokens.add("cape" if token == "cloak" else token)
    visit(value)
    return tokens


def default_parts(archetype: str, weapon: str) -> list[dict[str, Any]]:
    if archetype == "mounted_beast":
        return [
            {"id": "beast_body", "kind": "beast_body", "primitive": "ellipsoid", "anchor": "base_center", "side": None, "center": [0, 0, 8], "scale": [9, 3, 3], "required": True, "confidence": 0.8},
            {"id": "beast_head", "kind": "beast_head", "primitive": "ellipsoid", "anchor": "beast_body", "side": None, "center": [-7, 0, 10], "scale": [2.5, 1.5, 1.7], "required": True, "confidence": 0.8},
            {"id": "tail", "kind": "tail", "primitive": "capsule", "anchor": "beast_body", "side": None, "center": [12, 0, 7], "scale": [5, 0.4, 0.4], "required": True, "confidence": 0.8},
            {"id": "rider", "kind": "rider", "primitive": "humanoid_rig", "anchor": "saddle", "side": None, "center": [0, 0, 13], "scale": [0.56, 0.56, 0.56], "required": True, "confidence": 0.8},
            {"id": "weapon", "kind": "weapon", "primitive": weapon or "sword", "anchor": "rider_right_hand", "side": None, "center": [3, -1, 20], "scale": [1, 0.2, 2], "required": True, "confidence": 0.8},
        ]
    return [
        {"id": "torso", "kind": "torso", "primitive": "ellipsoid", "anchor": "base_center", "side": None, "center": [0, 0, 13], "scale": [3.7, 2, 4.9], "required": True, "confidence": 0.8},
        {"id": "head", "kind": "head", "primitive": "ellipsoid", "anchor": "torso", "side": None, "center": [0, 0, 18.5], "scale": [1.3, 1, 1.5], "required": True, "confidence": 0.8},
        {"id": "weapon", "kind": "weapon", "primitive": weapon or "sidearm", "anchor": "right_hand", "side": None, "center": [5.7, -1.2, 15.8], "scale": [1.2, 0.22, 2.4], "required": True, "confidence": 0.8},
    ]


if __name__ == "__main__":
    raise SystemExit(main())
