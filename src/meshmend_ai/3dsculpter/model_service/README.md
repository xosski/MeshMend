# MeshMend Production Model Service

This service is the local "hosted" backend for MeshMend. The API app now calls
this service first for text/image-to-3D generation instead of using the embedded
procedural sculptor path.

## Run

```powershell
python .\3dsculpter\model_service\main.py
```

The API/backend expects it at `http://127.0.0.1:8090` by default. The MeshMend
desktop creator will also try to auto-start this service unless
`MESHMEND_AUTO_START_MODEL_SERVICE=0` is set.

## Configure the real production model runner

MeshMend does not pretend the old procedural sculptor is studio-quality. By
default the worker requires an external/local production 3D generator command:

```powershell
$env:MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND = 'python D:\models\text_to_3d.py --prompt-file {prompt_path} --out {output_dir} --quality {quality}'
$env:MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND = 'python D:\models\image_to_3d.py --image {image_path} --prompt-file {prompt_path} --out {output_dir} --quality {quality}'
python .\3dsculpter\model_service\main.py
```

The command must write one supported model file (`.stl`, `.glb`, `.obj`, `.ply`,
`.3mf`, `.fbx`, `.usdz`) or a `result.json` into `{output_dir}`.

If `/health` reports `ready_for_studio_quality: false`, the service is running
but the real production runner is still missing. In that state MeshMend will
refuse to create another blocky draft and will ask for these runner commands.

## Free local/no-API Hunyuan3D mode

For a free local backend with no API key or cloud signup, install Hunyuan3D-2 or
Hunyuan3D-2.1 locally and run MeshMend with:

```powershell
python .\cli.py --model-service
```

`cli.py` now auto-selects `free_local_hunyuan` for GUI/model-service launches.
It looks for Hunyuan3D in common locations such as `D:\models\Hunyuan3D-2` and
auto-detects its `.venv\Scripts\python.exe` for the worker. You can still point
it explicitly if your install lives elsewhere:

```powershell
python .\cli.py --model-service --hunyuan3d-path D:\models\Hunyuan3D-2
```

For the GUI, simply run:

```powershell
python .\cli.py
```

To opt out of automatic Hunyuan configuration:

```powershell
python .\cli.py --no-free-local-hunyuan
```

Default model settings favor the larger non-mini shape model for better geometry:

```powershell
$env:MESHMEND_HUNYUAN3D_MODEL = 'tencent/Hunyuan3D-2'
$env:MESHMEND_HUNYUAN3D_SUBFOLDER = 'hunyuan3d-dit-v2-0'
$env:MESHMEND_HUNYUAN3D_OUTPUT_FORMAT = 'stl'
$env:MESHMEND_HUNYUAN3D_STEPS = '48'
$env:MESHMEND_HUNYUAN3D_OCTREE_RESOLUTION = '384'
$env:MESHMEND_DETAIL_RELIEF_MM = '0.055'
```

If VRAM is too low, switch back to the smaller/faster model:

```powershell
$env:MESHMEND_HUNYUAN3D_MODEL = 'tencent/Hunyuan3D-2mini'
$env:MESHMEND_HUNYUAN3D_SUBFOLDER = 'hunyuan3d-dit-v2-mini-turbo'
```

STL stores geometry only; texture/detail that exists only as a color map in GLB
will not appear in an STL unless the shape model actually sculpted it into the
mesh. MeshMend adds sparse structured grooves by default; all-over micro-noise
is disabled unless `MESHMEND_ENABLE_MICRO_NOISE=1` is set.

Image-to-3D requests go straight to Hunyuan3D. Text prompts are handled as:

```text
MeshMend text prompt -> local concept image -> local Hunyuan3D image-to-3D -> MeshMend output
```

For text prompts you can either let MeshMend try its local Diffusers fallback, or
set your own no-API image generator command:

```powershell
$env:MESHMEND_FREE_LOCAL_TEXT_TO_IMAGE_COMMAND = 'python D:\models\txt2img.py --prompt-file {prompt_path} --out {output_dir}'
```

That command should write a `.png`, `.jpg`, `.jpeg`, or `.webp` concept image
into `{output_dir}`. Hunyuan3D then converts that image into the mesh.

Supported placeholders:

- `{input_json}` full request JSON
- `{prompt_path}` text file containing the prompt
- `{image_path}` decoded image for image-to-3D jobs
- `{output_dir}` output folder
- `{quality}` low/standard/high
- `{target_polycount}` requested polygon budget

For development only, you can opt back into the legacy procedural sculptor:

```powershell
$env:MESHMEND_PRODUCTION_ENGINE = 'legacy_sculptor'
```

That mode is not intended for production/studio miniature quality.
