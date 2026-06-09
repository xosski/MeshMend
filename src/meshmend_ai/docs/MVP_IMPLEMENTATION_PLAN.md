# MVP Implementation Plan

## Implemented first slice

- Create `meshmend/` package with app/core/ai/sculpt/repair/export/library/compliance modules.
- Load STL, OBJ, GLB, and PLY through `trimesh`.
- Display meshes in a PySide6 desktop window with optional OpenGL viewport.
- Repair meshes through the existing `repair.py` engine: duplicate/degenerate cleanup, boundary-loop capping, normals fix, validation, optional component bridging.
- Auto-scale miniatures to 28mm, 32mm, or 75mm height.
- Add circular tabletop base geometry.
- Decimate through optional local simplification backends.
- Remesh/subdivide for local density increase.
- Export slicer-ready STL/OBJ/GLB/PLY plus printability JSON sidecar.
- Generate through the existing local MeshMend `native_generation.py` adapter by default, with a procedural placeholder adapter retained for tests/fallback.
- Add originality/compliance prompt sanitization for protected direct-copy cues and symbols.
- Add tests for repair, export, scale/base, compliance, and local generation.

## Next MVP hardening

- Replace text fallback viewport with bundled OpenGL renderer dependency in packaged builds.
- Add pymeshlab/manifold3d repair path when installed.
- Add SQLite project file for scene history and adapter settings.
- Add drag/drop import and recent project list.
- Add brush picking in viewport for sculpt tools.
- Add boolean add/subtract with manifold3d or Blender bridge.

## Definition of done for this MVP

The MVP is useful if a Windows user can install requirements, run `python -m meshmend`, open an STL/OBJ, repair it through the existing MeshMend repair engine, scale it, add a base, inspect printability, export a repaired STL, and generate a local mesh through the existing MeshMend native generator without any paid/cloud service.
