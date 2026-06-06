# MeshMend store-quality backend contract

MeshMend's bundled local generators are draft/procedural or experimental. To enable true store/studio-quality 8K miniature generation, configure an external certified production runner.

## Required environment

```powershell
$env:MESHMEND_PRODUCTION_ENGINE = "external"
$env:MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED = "1"
$env:MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND = "python D:\MeshMend\src\meshmend_ai\external_local_store_quality_generator.py --input {input_json} --prompt {prompt_path} --output-dir {output_dir} --quality {quality} --target-polycount {target_polycount}"
$env:MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND = "python D:\MeshMend\src\meshmend_ai\external_local_store_quality_generator.py --input {input_json} --prompt {prompt_path} --image {image_path} --output-dir {output_dir} --quality {quality} --target-polycount {target_polycount}"

# No API key is required. Install/configure a local Hunyuan3D-2/2.1 worker.
$env:MESHMEND_ALLOW_HUNYUAN_STORE_QUALITY = "1"
# Optional if Hunyuan3D is not installed in the worker Python environment:
$env:MESHMEND_HUNYUAN3D_PATH = "D:\path\to\Hunyuan3D-2"
```

Or launch the service with CLI flags:

```powershell
python cli.py --model-service --restart-model-service `
  --store-quality-text-command "python D:\MeshMend\src\meshmend_ai\external_local_store_quality_generator.py --input {input_json} --prompt {prompt_path} --output-dir {output_dir} --quality {quality} --target-polycount {target_polycount}" `
  --store-quality-image-command "python D:\MeshMend\src\meshmend_ai\external_local_store_quality_generator.py --input {input_json} --prompt {prompt_path} --image {image_path} --output-dir {output_dir} --quality {quality} --target-polycount {target_polycount}" `
  --certify-store-quality-backend
```

## No-API local external generator

`external_local_store_quality_generator.py` is the no-API production adapter. It runs local Hunyuan3D through MeshMend's external backend contract, so the model service can use `MESHMEND_PRODUCTION_ENGINE=external` without hosted API keys.

It still requires local model software/assets:

- Hunyuan3D-2/2.1 installed in `C:\Python313\python.exe`, or `MESHMEND_HUNYUAN3D_PATH` pointing to a local clone installed with its requirements.
- For text-to-3D, either local diffusers support or `MESHMEND_FREE_LOCAL_TEXT_TO_IMAGE_COMMAND` so MeshMend can create a concept image before Hunyuan image-to-3D.

This runner emits `store_quality_certified=true` only after the generated local mesh passes MeshMend's store-quality mesh gates and score contract.

For strict certification, configure a local no-API reviewer command that returns `store_quality_scores`:

```powershell
$env:MESHMEND_LOCAL_QUALITY_REVIEW_COMMAND = "python D:\path\to\local_reviewer.py --prompt {prompt_path} --spec {spec_path} --model {model_path} --out {review_json}"
```

The reviewer must write JSON containing:

```json
{
  "store_quality_scores": {
    "semantic_fidelity_score": 0.85,
    "anatomy_score": 0.85,
    "detail_density_score": 0.85,
    "surface_finish_score": 0.85,
    "printability_score": 0.85,
    "certifier": "local-reviewer-name"
  }
}
```

If you do not have a local reviewer yet, MeshMend can estimate non-semantic scores from mesh validation only when explicitly enabled:

```powershell
$env:MESHMEND_ALLOW_LOCAL_QUALITY_SCORE_ESTIMATES = "1"
```

Leave that off if you want MeshMend to fail rather than certify a model it cannot semantically review.

## Real HTTP external generator adapter

`external_http_store_quality_generator.py` is the production adapter for a real HTTP/API model generator. It is not a procedural fallback. It:

1. Builds MeshMend's enhanced miniature prompt and required landmark spec.
2. Calls `MESHMEND_HTTP_GENERATOR_SUBMIT_URL` with JSON.
3. Optionally polls `MESHMEND_HTTP_GENERATOR_STATUS_URL_TEMPLATE` until the provider finishes.
4. Downloads the generated `.stl`, `.glb`, `.obj`, `.ply`, `.3mf`, `.fbx`, or `.usdz`.
5. Validates mesh density, watertightness, connected components, and depth.
6. Emits MeshMend's certified `result.json` only when the configured upstream is explicitly marked certified.

Common provider configuration:

```powershell
$env:MESHMEND_HTTP_GENERATOR_SUBMIT_URL = "https://api.provider.example/v1/generations"
$env:MESHMEND_HTTP_GENERATOR_STATUS_URL_TEMPLATE = "https://api.provider.example/v1/generations/{job_id}"
$env:MESHMEND_HTTP_GENERATOR_API_KEY = "..."
$env:MESHMEND_HTTP_GENERATOR_AUTH_HEADER = "Authorization"
$env:MESHMEND_HTTP_GENERATOR_AUTH_PREFIX = "Bearer"
$env:MESHMEND_HTTP_GENERATOR_JOB_ID_PATHS = "id,job_id,task_id,data.id"
$env:MESHMEND_HTTP_GENERATOR_STATUS_PATHS = "status,state,data.status"
$env:MESHMEND_HTTP_GENERATOR_MODEL_URL_PATHS = "output.model_url,model_url,download_url,model_urls.glb,model_urls.stl"
$env:MESHMEND_HTTP_GENERATOR_CERTIFIES_STORE_QUALITY = "1"
$env:MESHMEND_HTTP_GENERATOR_PROVIDER = "provider-name"
$env:MESHMEND_HTTP_GENERATOR_CERTIFIER = "provider-name/model/review-gate"
```

If your provider expects a custom JSON body, set `MESHMEND_HTTP_GENERATOR_REQUEST_JSON`. Available placeholders include `{prompt}`, `{negative_prompt}`, `{quality}`, `{target_polycount}`, `{scale_mm}`, `{workflow}`, `{spec}`, and `{image_data_uri}`.

Example:

```powershell
$env:MESHMEND_HTTP_GENERATOR_REQUEST_JSON = '{"prompt": {prompt}, "mode": "preview-to-model", "quality": "high", "metadata": {spec}}'
```

## Runner input placeholders

MeshMend expands these placeholders in the configured command:

- `{input_json}`: full request JSON from MeshMend.
- `{output_dir}`: directory where the runner must write outputs.
- `{prompt}`: shell-quoted prompt text.
- `{prompt_path}`: prompt text file path.
- `{image_path}`: decoded input image path for image-to-3D requests, otherwise empty.
- `{quality}`: requested quality string.
- `{target_polycount}`: requested target polygon count.

The legacy `external_store_quality_generator.py` scaffold can still wrap a local command. It creates and passes these placeholders to `MESHMEND_EXTERNAL_GENERATOR_COMMAND`:

- `{enhanced_prompt_path}`: prompt rewritten for 32mm commercial miniature sculpting.
- `{spec_path}`: JSON miniature spec with archetype, weapon, required landmarks, target density, and printability constraints.
- `{image_path}`: decoded image path for image-to-3D, or the literal value `none` for text-to-3D.

## Required runner output

The runner must write either `result.json` into `{output_dir}` or print JSON to stdout. For store-quality requests, MeshMend requires a local `model_file` and certification metadata:

```json
{
  "model_file": "final_miniature.stl",
  "model_format": "stl",
  "provider": "your_backend_name",
  "capability_tier": "certified_store_quality_external",
  "geometry_source": "certified_external_3d_generator",
  "store_quality_certified": true,
  "store_quality_scores": {
    "semantic_fidelity_score": 0.88,
    "anatomy_score": 0.86,
    "detail_density_score": 0.90,
    "surface_finish_score": 0.84,
    "printability_score": 0.92,
    "certifier": "your_backend_or_artist_review_gate"
  },
  "mesh_info": {
    "semantic_fidelity_score": 0.85,
    "detail_source": "native_generated_sculpt_geometry"
  },
  "consumed_credits": 0
}
```

`model_file` must be a file in `{output_dir}` with a supported extension: `.stl`, `.glb`, `.obj`, `.ply`, `.3mf`, `.fbx`, or `.usdz`.

When using `external_store_quality_generator.py`, the underlying real generator must return `store_quality_certified: true` in its own `result.json` or stdout. The scaffold will not certify an arbitrary mesh by itself unless you explicitly set `MESHMEND_EXTERNAL_TRUST_UNCERTIFIED_OUTPUT=1` for local testing.

## MeshMend validation gates

For store/studio/8K requests, MeshMend validates the external result before accepting it:

- `store_quality_certified` must be `true`.
- `store_quality_scores` must include `semantic_fidelity_score`, `anatomy_score`, `detail_density_score`, `surface_finish_score`, `printability_score`, and `certifier`.
  - Default minimum score: `0.80`.
  - Tuned by `MESHMEND_CERTIFIED_MIN_QUALITY_SCORE`.
  - For local testing only, this check can be disabled with `MESHMEND_REQUIRE_EXTERNAL_QUALITY_SCORES=0`.
- A local `model_file` must exist in `{output_dir}`.
- The mesh must load successfully.
- The mesh must be non-empty.
- The mesh must meet the face-count ratio: `faces >= target_polycount * MESHMEND_CERTIFIED_MIN_FACE_RATIO`.
  - Default ratio: `0.75`.
- The mesh must be watertight.
- Component count must be `<= MESHMEND_CERTIFIED_MAX_COMPONENTS`.
  - Default: `3`.
- Depth ratio must be `>= MESHMEND_CERTIFIED_MIN_DEPTH_RATIO`.
  - Default: `0.18`.
- Mesh max extent must be plausible for the requested miniature scale.
  - Defaults: `0.35x <= max_extent / scale_mm <= 4.0x`.
  - Tuned by `MESHMEND_CERTIFIED_MIN_SCALE_RATIO` and `MESHMEND_CERTIFIED_MAX_SCALE_RATIO`.

These gates prove topology and basic density. Your certified backend is still responsible for true miniature sculpt quality: prompt fidelity, anatomy, pose, readable surface hierarchy, material detail, 360-degree forms, and commercial STL aesthetics.

## Minimal production backend responsibilities

A certified backend should implement at least:

```diagram
╭──────────────╮
│ MeshMend req │
╰──────┬───────╯
       ▼
╭────────────────────╮
│ Miniature spec      │
│ class/pose/gear     │
╰──────┬─────────────╯
       ▼
╭────────────────────╮
│ Sculpt generator    │
│ anatomy/armor/cloth │
╰──────┬─────────────╯
       ▼
╭────────────────────╮
│ Detail hierarchy    │
│ bevels/folds/cuts   │
╰──────┬─────────────╯
       ▼
╭────────────────────╮
│ Fused printable STL │
╰────────────────────╯
```

Hunyuan/image reconstruction and MeshMend native procedural scaffolds should not be certified unless they are replaced with a true high-detail miniature sculpt pipeline.
