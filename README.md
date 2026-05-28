# MeshMend AI

MeshMend is a local-first 3D miniature creation and STL repair toolkit. It can repair existing meshes, launch a desktop creation/repair GUI, and use a local no-API 3D generation backend to create printable tabletop miniatures from text or image prompts.

The current creation pipeline is designed for home/local use:

```text
Prompt or image
  -> local concept generation / reference image
  -> local Hunyuan3D image-to-3D shape generation
  -> MeshMend post-processing backend
  -> scaled, single-subject, printable STL
```

## Features

- Repair STL/OBJ/PLY meshes for 3D printing.
- Bridge detached mesh islands and fix common mesh issues.
- Desktop GUI for repair and model creation.
- Local model-service backend for text/image-to-3D generation.
- Free local Hunyuan3D mode with no hosted API key or per-generation credits.
- Post-processing backend for generated miniatures:
  - single-subject enforcement
  - millimeter scale normalization
  - sheet/thin-mesh thickening
  - high-density STL subdivision
  - structured miniature sculpt relief

## Repository layout

```text
meshmend_ai/
├─ cli.py                                  # Main CLI entry point
├─ gui.py                                  # Tkinter GUI
├─ repair.py                               # Mesh repair pipeline
├─ sculptor.py                             # Creation bridge / hosted backend client
├─ 3dsculpter/
│  ├─ model_service/
│  │  ├─ main.py                           # Local model-service API
│  │  ├─ production_worker.py              # Hunyuan/custom runner worker
│  │  ├─ postprocess_backend.py            # Miniature STL post-processing backend
│  │  └─ README.md                         # Backend-specific setup notes
│  └─ ...
└─ training_data/                          # Optional local training assets/checkpoints
```

## Requirements

- Windows is currently the tested development environment.
- Python for MeshMend itself. Python 3.11 is recommended for the AI stack; Python 3.13 may work for the GUI/service but many ML wheels may not.
- For local Hunyuan3D generation:
  - NVIDIA GPU strongly recommended.
  - Hunyuan3D installed locally in its own Python environment.
  - Public model weights may download from Hugging Face on first run. This does not require a hosted generation API key.

## Basic usage

From the repository directory:

```powershell
cd D:\MeshMend\src\meshmend_ai
python .\cli.py --help
```

Launch the GUI:

```powershell
python .\cli.py
```

Repair a mesh from the command line:

```powershell
python .\cli.py input.stl output_repaired.stl --assistant --report
```

Stop any running local model service:

```powershell
python .\cli.py --stop-model-service
```

## Local no-API Hunyuan3D setup

Do **not** rely on `pip install hy3dgen` alone unless it provides the official `hy3dgen.shapegen.Hunyuan3DDiTFlowMatchingPipeline` in the exact worker Python environment. The recommended setup is to install the official Hunyuan3D repository.

Example install:

```powershell
mkdir D:\models
cd D:\models
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
cd D:\models\Hunyuan3D-2

py -3.11 -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

Check the Hunyuan install:

```powershell
D:\models\Hunyuan3D-2\.venv\Scripts\python.exe -c "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; print('ok')"
```

MeshMend also has a diagnostic command:

```powershell
cd D:\MeshMend\src\meshmend_ai
python .\cli.py --check-hunyuan `
  --hunyuan3d-path D:\models\Hunyuan3D-2 `
  --hunyuan3d-python D:\models\Hunyuan3D-2\.venv\Scripts\python.exe
```

## Start the local generation backend

Start the model service in one terminal and leave it running:

```powershell
cd D:\MeshMend\src\meshmend_ai

python .\cli.py --model-service `
  --restart-model-service `
  --hunyuan3d-path D:\models\Hunyuan3D-2 `
  --hunyuan3d-python D:\models\Hunyuan3D-2\.venv\Scripts\python.exe `
  --hunyuan3d-model tencent/Hunyuan3D-2 `
  --hunyuan3d-subfolder hunyuan3d-dit-v2-0 `
  --hunyuan3d-output-format stl
```

Then open a second terminal for the GUI:

```powershell
cd D:\MeshMend\src\meshmend_ai
python .\cli.py
```

Health check:

```powershell
(Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8090/health).Content
```

The health response should show:

- `production_engine: free_local_hunyuan`
- `ready_for_studio_quality: true`
- a valid `model_worker_python`
- `hunyuan_import.ok: true`

## Miniature detail and scale controls

Generated STL post-processing is controlled with environment variables:

```powershell
$env:MESHMEND_MIN_EXPORT_FACES = '500000'       # High-density STL target
$env:MESHMEND_DETAIL_RELIEF_MM = '0.08'        # Structured sculpt relief depth
$env:MESHMEND_MIN_THICKNESS_RATIO = '0.22'     # Prevent flat/sheet outputs
$env:MESHMEND_ENABLE_MICRO_NOISE = '0'         # Keep random noise disabled
```

If generation is too slow or files are too large:

```powershell
$env:MESHMEND_MIN_EXPORT_FACES = '250000'
```

If sculpt relief is too subtle:

```powershell
$env:MESHMEND_DETAIL_RELIEF_MM = '0.10'
```

If the GPU cannot run the larger Hunyuan model, switch to the smaller model:

```powershell
python .\cli.py --model-service `
  --restart-model-service `
  --hunyuan3d-path D:\models\Hunyuan3D-2 `
  --hunyuan3d-python D:\models\Hunyuan3D-2\.venv\Scripts\python.exe `
  --hunyuan3d-model tencent/Hunyuan3D-2mini `
  --hunyuan3d-subfolder hunyuan3d-dit-v2-mini-turbo `
  --hunyuan3d-output-format stl
```

## Output files

Model-service jobs write task data under:

```text
3dsculpter/model_service/tasks/<task_id>/
3dsculpter/model_service/outputs/<task_id>/
```

Useful generated files include:

- `input.json` — request payload
- `concept_*.png` — candidate concept images
- `concept_single_subject.png` — selected/cropped concept image
- `meshmend_hunyuan.stl` — generated STL
- `result.json` — final metadata including face count, scale, and postprocess report
- `worker_command.json` / `worker_error.txt` — debugging info if worker startup fails

## Notes and limitations

- STL stores geometry only. Texture/color detail from a GLB will not appear in STL unless converted into geometry.
- Hunyuan3D is a base-form generator in this workflow. MeshMend's post-processing backend adds printable density, scale, single-subject enforcement, and structured relief, but it is not yet a full human sculptor replacement.
- Higher detail requires more faces, more GPU time, and larger files.
- If text prompts generate blurry concept images, the final STL will also lack subject-specific detail. The backend generates multiple concept candidates for high-quality jobs and selects the sharpest, but a stronger local text-to-image checkpoint may improve results.

## License

# PRIVATE USE LICENSE AGREEMENT

## AI-Generated 3D Miniatures & Model Files

**Effective Date:** 5/28/2026
**Licensor:** Erick King/GhostCore
**Product Name:** MeshMend

---

## 1. Grant of License

Upon purchase, the Licensor grants the purchaser (“Licensee”) a non-exclusive, non-transferable, revocable license to use the provided digital files and generated assets for personal, private use only.

This license permits the Licensee to:

* Download and store the files
* 3D print physical copies for personal use
* Resize or support files for personal printing purposes
* Paint, modify, or kitbash physical prints for personal hobby use
* Share photographs of printed models online for non-commercial purposes

---

## 2. Restrictions

The Licensee may NOT:

* Resell, redistribute, trade, gift, or share the digital STL/OBJ/3MF files
* Upload files to any public or private repository, drive, marketplace, torrent, or file-sharing service
* Sell physical prints derived from the files
* Use the files for print-on-demand services
* Train AI systems using the files, renders, meshes, or derivatives
* Claim authorship or ownership of the original digital assets
* Mint the files or derivatives as NFTs or blockchain assets
* Commercially license or sublicense the files to third parties
* Reverse engineer branding, metadata, or creator identification systems

---

## 3. AI-Generated Content Notice

Some or all assets covered under this license may include AI-assisted or AI-generated components, including but not limited to:

* Concept generation
* Mesh generation
* Surface detailing
* Texturing references
* Pose generation
* Render enhancement

Final curation, modification, assembly, and publishing rights remain exclusively with the Licensor.

---

## 4. Intellectual Property

All digital assets, derivatives, renders, promotional imagery, naming conventions, faction concepts, lore, and associated branding remain the intellectual property of the Licensor unless otherwise stated in writing.

Purchase of the files does not transfer ownership of the intellectual property.

---

## 5. No Warranty

The files are provided “as is” without warranties of any kind, expressed or implied.

The Licensor is not responsible for:

* Printer compatibility issues
* Failed prints
* Resin or filament settings
* Hardware damage
* User modifications
* Unsupported slicer configurations

The Licensee assumes all responsibility for safe operation of 3D printing equipment.

---

## 6. Termination

Violation of this agreement automatically terminates the license.

Upon termination, the Licensee must destroy all digital copies of the files and cease further use.

---

## 7. Commercial Licensing

Commercial licenses for physical print sales or merchant distribution may be available separately through direct agreement with the Licensor.

Contact: [Your Email / Store URL]

---

## 8. Acceptance of Terms

By downloading, accessing, printing, or using these files, the Licensee agrees to all terms outlined in this agreement.

---

**Copyright © [2026] [Ghost Division]. All Rights Reserved.**

