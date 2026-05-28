# 3D Sculptor - Warhammer Miniature Generator

A pure Python GUI application for generating AI-powered Warhammer-adjacent miniatures using Stable Diffusion and advanced mesh processing.

## Features

- **AI-Powered Generation**: Generate unique miniatures from text descriptions using Stable Diffusion
- **Real-time 3D Viewer**: Interactive 3D visualization with rotation, zoom, and pan controls
- **Multiple Styles**: Pre-configured styles including Sci-Fi Soldiers, Fantasy Knights, Chaos Warriors, and more
- **Mesh Manipulation**: Load, save, scale, and edit 3D models
- **Export Formats**: Export to STL, OBJ, and PLY formats
- **Adjustable Detail Levels**: Control the complexity and detail of generated models
- **Miniature Scaling**: Built-in support for standard tabletop miniature scales (28mm)

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended, but CPU mode available)
- 8GB+ RAM

## Installation

### 1. Clone and Setup

```bash
cd d:/3dsculpter
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*Note: First installation may take 15-30 minutes as it downloads AI models (~8GB)*

### 3. Run the Application

```bash
python main.py
```

## Usage

### Generating a Miniature

1. **Enter Description**: Describe your miniature in the "Description" field
   - Example: "Space marine warrior with red armor and bolter weapon"
   - Be specific about style, color, and details for better results

2. **Select Style**: Choose a pre-configured style from the dropdown
   - Each style has predefined modifiers for consistent theming

3. **Set Detail Level**: Adjust from 1 (simple) to 5 (highly detailed)

4. **Adjust Scale**: Set the miniature size in millimeters (standard is 28mm)

5. **Click "Generate Model"**: Wait for the AI to create your miniature

### Viewing and Manipulating Models

- **Rotate**: Use the rotation buttons (↻ X, ↻ Y, ↻ Z) or drag in the 3D view
- **Zoom**: Click "+" or "-" buttons, or scroll in 3D view
- **Reset View**: Returns camera to default position

### Exporting Your Model

1. Click **"Export as STL"** button
2. Choose a location and filename
3. Select format (STL, OBJ, or PLY)
4. Your model is ready for 3D printing or further editing

### Importing Models

1. Click **"Import Model"** button
2. Select a 3D file (STL, OBJ, PLY)
3. Model loads into the viewer

## System Architecture

```
main.py                 # Entry point
├── app/
│   ├── main_window.py  # PyQt6 UI and main application window
│   ├── viewer_3d.py    # Vispy-based 3D viewer
│   ├── ai_generator.py # AI model generation using Diffusers
│   └── mesh_utils.py   # Mesh loading, saving, and manipulation
```

## Performance Tips

1. **First Run**: The application downloads ~8GB of AI models. This is one-time.
2. **GPU Mode**: Ensure CUDA is installed for ~5x faster generation
3. **Detail Level**: Higher detail = longer generation time
4. **Batch Processing**: Generate multiple models in sequence for efficiency

## Troubleshooting

### "CUDA out of memory"
- Reduce detail level to 2-3
- Close other applications
- Use CPU mode (slower but works)

### "Model looks flat/2D"
- Increase detail level
- Try a different style
- Adjust the prompt to be more descriptive

### "Very slow generation"
- Check if running on CPU (slower ~3-5 min vs ~30-60 sec on GPU)
- Ensure no other GPU tasks running
- Try smaller detail level

### "ImportError: vispy/PyQt6"
```bash
pip install --upgrade PyQt6 vispy
```

## Planned Features

- [ ] Real-time mesh sculpting tools
- [ ] Texture generation and painting
- [ ] Multi-part model assembly
- [ ] Batch generation
- [ ] Model library and favorites
- [ ] STL-to-printable optimization
- [ ] Undo/redo history
- [ ] Custom style creation

## Model Sources

- **Stable Diffusion v1.5**: Image generation
- **Trimesh**: Mesh processing and conversion
- **Vispy**: Hardware-accelerated 3D visualization

## License

MIT License - Feel free to use for personal and commercial projects

## Support

For issues, feature requests, or contributions, please open an issue or submit a pull request.

---

**Made with ❤️ for tabletop hobbyists and 3D printing enthusiasts**
