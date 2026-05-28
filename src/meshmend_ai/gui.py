from __future__ import annotations

import os
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from meshmend_ai.assistant import MeshMendAssistant
    from meshmend_ai.detail_quality import MAX_DETAIL_TRIANGLES, assess_8k_detail
    from meshmend_ai.generative_model import Local3DGenerativeModel, default_training_data_dir
    from meshmend_ai.high_resolution_latent import LocalMeshLatentGenerator
    from meshmend_ai.neural_diffusion import Neural3DDiffusionModel, NeuralTrainingConfig
    from meshmend_ai.sculptor import get_sculptor_foundation
else:
    from .assistant import MeshMendAssistant
    from .detail_quality import MAX_DETAIL_TRIANGLES, assess_8k_detail
    from .generative_model import Local3DGenerativeModel, default_training_data_dir
    from .high_resolution_latent import LocalMeshLatentGenerator
    from .neural_diffusion import Neural3DDiffusionModel, NeuralTrainingConfig
    from .sculptor import get_sculptor_foundation


MESH_FILE_TYPES = (
    ("3D mesh files", "*.stl *.obj *.ply"),
    ("STL files", "*.stl"),
    ("OBJ files", "*.obj"),
    ("PLY files", "*.ply"),
    ("All files", "*.*"),
)


class MeshMendGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("MeshMend AI Repair")
        self.root.geometry("840x760")
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.creation_image_path = tk.StringVar()
        self.training_source_dir = tk.StringVar(value=str(default_training_data_dir() / "raw_stl"))
        self.creation_scale_mm = tk.StringVar(value="32")
        self.print_detail_um = tk.StringVar(value="50")
        self.max_detail_triangles = tk.StringVar(value=str(MAX_DETAIL_TRIANGLES))
        self.neural_resolution = tk.StringVar(value="96")
        self.neural_autoencoder_epochs = tk.StringVar(value="50")
        self.neural_diffusion_epochs = tk.StringVar(value="90")
        self.connector_radius = tk.StringVar(value="auto")
        self.max_bridge_distance = tk.StringVar(value="")
        self.use_perseus = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Choose a 3D model to repair.")
        self.progress_status = tk.StringVar(value="Idle")
        self.training_status = tk.StringVar(value="Choose a folder of STL/OBJ/PLY files and optional matching images.")
        self._result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.sculptor = get_sculptor_foundation()
        self._build_ui()

    def run(self) -> int:
        self.root.mainloop()
        return 0

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(outer, text="MeshMend AI Repair", font=("Segoe UI", 16, "bold"))
        title.pack(anchor=tk.W)
        subtitle = ttk.Label(
            outer,
            text="Import or create an STL/OBJ/PLY, let the AI assistant repair detached pieces and holes, then save the fixed model.",
            wraplength=700,
        )
        subtitle.pack(anchor=tk.W, pady=(2, 14))

        notebook = ttk.Notebook(outer)
        notebook.pack(fill=tk.BOTH, expand=True)
        workflow_tab = ttk.Frame(notebook, padding=8)
        training_tab = ttk.Frame(notebook, padding=8)
        notebook.add(workflow_tab, text="Repair + Create")
        notebook.add(training_tab, text="Train 3D Model")

        file_frame = ttk.LabelFrame(workflow_tab, text="Repair file")
        file_frame.pack(fill=tk.X, pady=(0, 12))
        self._path_row(file_frame, "Input model", self.input_path, self.choose_input).pack(fill=tk.X, padx=10, pady=(10, 5))
        self._path_row(file_frame, "Save repaired as", self.output_path, self.choose_output).pack(fill=tk.X, padx=10, pady=(5, 10))

        options = ttk.LabelFrame(workflow_tab, text="Assistant options")
        options.pack(fill=tk.X, pady=(0, 12))
        option_grid = ttk.Frame(options)
        option_grid.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(option_grid, text="Connector radius").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        ttk.Entry(option_grid, textvariable=self.connector_radius, width=14).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(option_grid, text="Use 'auto' for model-based sizing").grid(row=0, column=2, sticky=tk.W, padx=(8, 0))
        ttk.Label(option_grid, text="Max bridge distance").grid(row=1, column=0, sticky=tk.W, padx=(0, 8), pady=(8, 0))
        ttk.Entry(option_grid, textvariable=self.max_bridge_distance, width=14).grid(row=1, column=1, sticky=tk.W, pady=(8, 0))
        ttk.Label(option_grid, text="Blank means no limit").grid(row=1, column=2, sticky=tk.W, padx=(8, 0), pady=(8, 0))
        ttk.Checkbutton(option_grid, text="Use Perseus explanation layer", variable=self.use_perseus).grid(
            row=2, column=0, columnspan=3, sticky=tk.W, pady=(8, 0)
        )

        sculptor = ttk.LabelFrame(workflow_tab, text="3D Sculptor foundation")
        sculptor.pack(fill=tk.X, pady=(0, 12))
        sculptor_status = "available" if self.sculptor.available else "not found"
        ttk.Label(
            sculptor,
            text=f"Bundled 3D model creator: {self.sculptor.root} ({sculptor_status})",
            wraplength=700,
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))
        sculptor_buttons = ttk.Frame(sculptor)
        sculptor_buttons.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(sculptor_buttons, text="Open 3D Sculptor", command=self.open_sculptor).pack(side=tk.LEFT)
        ttk.Button(sculptor_buttons, text="Import latest sculptor model", command=self.import_latest_sculptor_model).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(sculptor_buttons, text="Check AI modules + learning", command=self.check_modules).pack(side=tk.LEFT, padx=(8, 0))

        creator = ttk.LabelFrame(workflow_tab, text="Chat/create 3D model")
        creator.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(creator, text="Describe the model to create, optionally attach an image, then generate an STL.").pack(
            anchor=tk.W, padx=10, pady=(10, 4)
        )
        self.chat_prompt = tk.Text(creator, height=4, wrap=tk.WORD)
        self.chat_prompt.pack(fill=tk.X, padx=10, pady=(0, 6))
        detail_row = ttk.Frame(creator)
        detail_row.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Label(detail_row, text="Scale", width=10).pack(side=tk.LEFT)
        scale_combo = ttk.Combobox(
            detail_row,
            textvariable=self.creation_scale_mm,
            values=("15", "20", "25", "28", "32", "35", "40", "48", "54", "75"),
            width=8,
        )
        scale_combo.pack(side=tk.LEFT)
        ttk.Label(detail_row, text="mm").pack(side=tk.LEFT, padx=(6, 14))
        ttk.Label(detail_row, text="Print detail", width=10).pack(side=tk.LEFT)
        detail_combo = ttk.Combobox(
            detail_row,
            textvariable=self.print_detail_um,
            values=("100", "50", "25"),
            width=8,
        )
        detail_combo.pack(side=tk.LEFT)
        ttk.Label(detail_row, text="um  (50=8K draft, 25=max detail)").pack(side=tk.LEFT, padx=(6, 14))
        ttk.Label(detail_row, text="Triangle cap").pack(side=tk.LEFT)
        ttk.Entry(detail_row, textvariable=self.max_detail_triangles, width=12).pack(side=tk.LEFT, padx=(6, 0))
        image_row = ttk.Frame(creator)
        image_row.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Label(image_row, text="Image", width=10).pack(side=tk.LEFT)
        ttk.Entry(image_row, textvariable=self.creation_image_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(image_row, text="Browse", command=self.choose_creation_image).pack(side=tk.LEFT)
        creator_buttons = ttk.Frame(creator)
        creator_buttons.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.create_button = ttk.Button(creator_buttons, text="Create 3D model from chat/image", command=self.start_create_model)
        self.create_button.pack(side=tk.LEFT)

        action_frame = ttk.Frame(workflow_tab)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        self.repair_button = ttk.Button(action_frame, text="Repair with AI and Save", command=self.start_repair)
        self.repair_button.pack(side=tk.LEFT)
        ttk.Label(action_frame, textvariable=self.status).pack(side=tk.LEFT, padx=(12, 0))

        self.progress = ttk.Progressbar(workflow_tab, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(workflow_tab, textvariable=self.progress_status).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(workflow_tab, text="Assistant log / repair report").pack(anchor=tk.W)
        self.report_text = tk.Text(workflow_tab, height=10, wrap=tk.WORD)
        self.report_text.pack(fill=tk.BOTH, expand=True)
        self._build_training_tab(training_tab)

    def _path_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar, command) -> ttk.Frame:
        row = ttk.Frame(parent)
        ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(row, text="Browse", command=command).pack(side=tk.LEFT)
        return row

    def _build_training_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="Train MeshMend's local 3D generative model from STL/OBJ/PLY files and optional matching images.",
            wraplength=760,
        ).pack(anchor=tk.W, pady=(0, 12))

        default_dir = default_training_data_dir()
        drop_frame = ttk.LabelFrame(parent, text="Default training folders")
        drop_frame.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(drop_frame, text=f"Place STL/OBJ/PLY files in: {default_dir / 'raw_stl'}", wraplength=740).pack(
            anchor=tk.W, padx=10, pady=(10, 2)
        )
        ttk.Label(drop_frame, text=f"Optional matching images can go in: {default_dir / 'raw_images'}", wraplength=740).pack(
            anchor=tk.W, padx=10, pady=(0, 10)
        )
        ttk.Button(drop_frame, text="Open default training folder", command=self.open_default_training_folder).pack(
            anchor=tk.W, padx=10, pady=(0, 10)
        )

        source_frame = ttk.LabelFrame(parent, text="Train from directory")
        source_frame.pack(fill=tk.X, pady=(0, 12))
        row = ttk.Frame(source_frame)
        row.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(row, text="Source folder", width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.training_source_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(row, text="Browse", command=self.choose_training_dir).pack(side=tk.LEFT)

        neural_options = ttk.LabelFrame(parent, text="Neural diffusion options")
        neural_options.pack(fill=tk.X, pady=(0, 12))
        neural_grid = ttk.Frame(neural_options)
        neural_grid.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(neural_grid, text="Voxel resolution").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        ttk.Combobox(neural_grid, textvariable=self.neural_resolution, values=("32", "64", "96"), width=8).grid(
            row=0, column=1, sticky=tk.W
        )
        ttk.Label(neural_grid, text="Experimental coarse shape model; 96+ recommended, not final tabletop detail").grid(
            row=0, column=2, sticky=tk.W, padx=(8, 0)
        )
        ttk.Label(neural_grid, text="AE epochs").grid(row=1, column=0, sticky=tk.W, padx=(0, 8), pady=(8, 0))
        ttk.Entry(neural_grid, textvariable=self.neural_autoencoder_epochs, width=10).grid(row=1, column=1, sticky=tk.W, pady=(8, 0))
        ttk.Label(neural_grid, text="Diffusion epochs").grid(row=1, column=2, sticky=tk.W, padx=(8, 8), pady=(8, 0))
        ttk.Entry(neural_grid, textvariable=self.neural_diffusion_epochs, width=10).grid(row=1, column=3, sticky=tk.W, pady=(8, 0))

        action = ttk.Frame(parent)
        action.pack(fill=tk.X, pady=(0, 10))
        self.train_button = ttk.Button(action, text="Train Local 3D Model", command=self.start_training)
        self.train_button.pack(side=tk.LEFT)
        self.train_neural_button = ttk.Button(action, text="Train Neural 3D Diffusion", command=self.start_neural_training)
        self.train_neural_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(action, textvariable=self.training_status).pack(side=tk.LEFT, padx=(12, 0))

        self.training_progress = ttk.Progressbar(parent, mode="determinate", maximum=100)
        self.training_progress.pack(fill=tk.X, pady=(0, 10))
        self.training_log = tk.Text(parent, height=16, wrap=tk.WORD)
        self.training_log.pack(fill=tk.BOTH, expand=True)

    def choose_input(self) -> None:
        selected = filedialog.askopenfilename(title="Choose model to repair", filetypes=MESH_FILE_TYPES)
        if not selected:
            return
        self.input_path.set(selected)
        if not self.output_path.get():
            path = Path(selected)
            self.output_path.set(str(path.with_name(f"{path.stem}_meshmend{path.suffix or '.stl'}")))

    def choose_output(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Save repaired model as",
            defaultextension=".stl",
            filetypes=MESH_FILE_TYPES,
        )
        if selected:
            self.output_path.set(selected)

    def choose_creation_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose image for model creation",
            filetypes=(
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*"),
            ),
        )
        if selected:
            self.creation_image_path.set(selected)

    def choose_training_dir(self) -> None:
        selected = filedialog.askdirectory(title="Choose training directory containing STL/image files")
        if selected:
            self.training_source_dir.set(selected)

    def open_default_training_folder(self) -> None:
        path = default_training_data_dir()
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except Exception:
            messagebox.showinfo("Training folder", str(path))

    def open_sculptor(self) -> None:
        try:
            self.sculptor.launch()
            self.status.set("3D Sculptor launched.")
        except Exception as exc:
            messagebox.showerror("3D Sculptor", f"Could not launch 3D Sculptor:\n{exc}")

    def import_latest_sculptor_model(self) -> None:
        latest = self.sculptor.latest_model()
        if latest is None:
            messagebox.showwarning("3D Sculptor", "No generated STL/OBJ/PLY model was found in the sculptor outputs folder.")
            return
        self.input_path.set(str(latest))
        self.output_path.set(str(latest.with_name(f"{latest.stem}_meshmend{latest.suffix}")))
        self.status.set(f"Imported latest sculptor model: {latest.name}")

    def check_modules(self) -> None:
        self.report_text.delete("1.0", tk.END)
        self._append_log("AI module diagnostics:\n")
        for name, value in self.sculptor.module_status().items():
            self._append_log(f"- {name}: {value}\n")
        self._append_log("\nLearning diagnostics:\n")
        learning = MeshMendAssistant(enable_perseus=True).learning_status()
        for name, value in learning.items():
            self._append_log(f"- {name}: {value}\n")

    def start_repair(self) -> None:
        try:
            input_path = Path(self.input_path.get()).expanduser()
            output_path = Path(self.output_path.get()).expanduser()
            if not input_path.exists():
                raise ValueError("Choose an existing input model first.")
            if not output_path.name:
                raise ValueError("Choose where to save the repaired model.")
            connector_radius = self._optional_float(self.connector_radius.get(), allow_auto=True)
            max_bridge_distance = self._optional_float(self.max_bridge_distance.get(), allow_auto=False)
        except Exception as exc:
            messagebox.showerror("Repair setup", str(exc))
            return

        self.repair_button.configure(state=tk.DISABLED)
        self.create_button.configure(state=tk.DISABLED)
        self._set_progress(0, "Starting repair")
        self.status.set("Repairing model...")
        self.report_text.delete("1.0", tk.END)

        thread = threading.Thread(
            target=self._repair_worker,
            args=(input_path, output_path, connector_radius, max_bridge_distance, self.use_perseus.get()),
            daemon=True,
        )
        thread.start()
        self.root.after(100, self._poll_result)

    def _repair_worker(
        self,
        input_path: Path,
        output_path: Path,
        connector_radius: float | None,
        max_bridge_distance: float | None,
        use_perseus: bool,
    ) -> None:
        try:
            self._post_progress(8, "Loading and inspecting model")
            assistant = MeshMendAssistant(enable_perseus=use_perseus)
            self._post_progress(20, "Choosing AI repair settings")
            assistant.build_plan(
                input_path,
                connector_radius=connector_radius,
                max_bridge_distance=max_bridge_distance,
                force_bridge=True,
            )
            self._post_progress(38, "Repairing holes, normals, and detached components")
            result = assistant.repair(
                input_path,
                output_path,
                connector_radius=connector_radius,
                max_bridge_distance=max_bridge_distance,
                force_bridge=True,
            )
            self._post_progress(82, "Running explanation and learning layer")
            if use_perseus and result.perseus is None:
                self._post_progress(88, "Perseus unavailable; using native explanation")
            self._post_progress(100, "Repair complete")
            self._result_queue.put(("ok", result))
        except Exception as exc:
            self._result_queue.put(("error", exc))

    def start_create_model(self) -> None:
        prompt = self.chat_prompt.get("1.0", tk.END).strip()
        image_text = self.creation_image_path.get().strip()
        image_path = Path(image_text).expanduser() if image_text else None
        try:
            if image_path is not None and not image_path.exists():
                raise ValueError("The selected image file does not exist.")
            if not prompt and image_path is None:
                raise ValueError("Enter text, choose an image, or both before creating a model.")
            scale_mm = self._creation_scale_mm()
            print_detail_um = self._print_detail_um()
            max_detail_triangles = self._positive_int(self.max_detail_triangles.get(), "Triangle cap")
        except Exception as exc:
            messagebox.showerror("Create model", str(exc))
            return

        self.repair_button.configure(state=tk.DISABLED)
        self.create_button.configure(state=tk.DISABLED)
        self.report_text.delete("1.0", tk.END)
        self._set_progress(0, "Starting model creation")
        self.status.set("Creating 3D model...")

        thread = threading.Thread(
            target=self._create_model_worker,
            args=(prompt, image_path, scale_mm, print_detail_um, max_detail_triangles),
            daemon=True,
        )
        thread.start()
        self.root.after(100, self._poll_result)

    def _create_model_worker(
        self,
        prompt: str,
        image_path: Path | None,
        scale_mm: float,
        print_detail_um: float,
        max_detail_triangles: int,
    ) -> None:
        try:
            output_path = self.sculptor.create_model(
                prompt,
                image_path,
                progress=self._post_progress,
                scale_mm=scale_mm,
                print_detail_um=print_detail_um,
                max_detail_triangles=max_detail_triangles,
            )
            self._result_queue.put(("created", (output_path, scale_mm, print_detail_um)))
        except Exception as exc:
            self._result_queue.put(("error", exc))

    def start_training(self) -> None:
        source_dir = Path(self.training_source_dir.get()).expanduser()
        if not source_dir.exists():
            messagebox.showerror("Training", "Choose an existing training directory first.")
            return

        self.train_button.configure(state=tk.DISABLED)
        self.train_neural_button.configure(state=tk.DISABLED)
        self.training_progress.configure(value=0)
        self.training_status.set("Training local 3D model...")
        self.training_log.delete("1.0", tk.END)
        thread = threading.Thread(target=self._training_worker, args=(source_dir,), daemon=True)
        thread.start()
        self.root.after(100, self._poll_result)

    def start_neural_training(self) -> None:
        source_dir = Path(self.training_source_dir.get()).expanduser()
        if not source_dir.exists():
            messagebox.showerror("Neural training", "Choose an existing training directory first.")
            return
        try:
            resolution = self._positive_int(self.neural_resolution.get(), "Voxel resolution")
            if resolution % 8 != 0:
                raise ValueError("Voxel resolution must be divisible by 8.")
            autoencoder_epochs = self._positive_int(self.neural_autoencoder_epochs.get(), "AE epochs")
            diffusion_epochs = self._positive_int(self.neural_diffusion_epochs.get(), "Diffusion epochs")
        except Exception as exc:
            messagebox.showerror("Neural training", str(exc))
            return

        self.train_button.configure(state=tk.DISABLED)
        self.train_neural_button.configure(state=tk.DISABLED)
        self.training_progress.configure(value=0)
        self.training_status.set("Training neural 3D diffusion model...")
        self.training_log.delete("1.0", tk.END)
        thread = threading.Thread(
            target=self._neural_training_worker,
            args=(source_dir, resolution, autoencoder_epochs, diffusion_epochs),
            daemon=True,
        )
        thread.start()
        self.root.after(100, self._poll_result)

    def _training_worker(self, source_dir: Path) -> None:
        try:
            result = Local3DGenerativeModel.train_from_directory(source_dir, progress=self._post_training_progress)
            LocalMeshLatentGenerator.train_from_directory(source_dir, progress=self._post_training_progress)
            self._result_queue.put(("training_done", result))
        except Exception as exc:
            self._result_queue.put(("training_error", exc))

    def _neural_training_worker(
        self,
        source_dir: Path,
        resolution: int,
        autoencoder_epochs: int,
        diffusion_epochs: int,
    ) -> None:
        try:
            config = NeuralTrainingConfig(
                autoencoder_epochs=autoencoder_epochs,
                diffusion_epochs=diffusion_epochs,
                resolution=resolution,
            )
            result = Neural3DDiffusionModel.train_from_directory(source_dir, config=config, progress=self._post_training_progress)
            self._result_queue.put(("neural_training_done", result))
        except Exception as exc:
            self._result_queue.put(("training_error", exc))

    def _post_training_progress(self, percent: int, message: str) -> None:
        self._result_queue.put(("training_progress", (percent, message)))

    def _poll_result(self) -> None:
        try:
            status, payload = self._result_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_result)
            return

        if status == "progress":
            percent, message = payload
            self._set_progress(int(percent), str(message))
            self.root.after(100, self._poll_result)
            return

        if status == "training_progress":
            percent, message = payload
            self.training_progress.configure(value=max(0, min(100, int(percent))))
            self.training_status.set(str(message))
            self.training_log.insert(tk.END, f"{int(percent)}% - {message}\n")
            self.training_log.see(tk.END)
            self.root.after(100, self._poll_result)
            return

        if status == "training_error":
            self.train_button.configure(state=tk.NORMAL)
            self.train_neural_button.configure(state=tk.NORMAL)
            self.training_progress.configure(value=0)
            self.training_status.set("Training failed")
            messagebox.showerror("Training failed", str(payload))
            return

        if status == "training_done":
            self.train_button.configure(state=tk.NORMAL)
            self.train_neural_button.configure(state=tk.NORMAL)
            self.training_progress.configure(value=100)
            self.training_status.set("Training complete")
            self.training_log.insert(
                tk.END,
                f"\nTraining complete. Examples: {payload.examples} | Images: {payload.images}\nCheckpoint: {payload.checkpoint_path}\n",
            )
            self.training_log.see(tk.END)
            messagebox.showinfo("Training complete", payload.message)
            return

        if status == "neural_training_done":
            self.train_button.configure(state=tk.NORMAL)
            self.train_neural_button.configure(state=tk.NORMAL)
            self.training_progress.configure(value=100)
            self.training_status.set("Neural training complete")
            self.training_log.insert(
                tk.END,
                f"\nNeural training complete. Examples: {payload.examples} | Resolution: {payload.resolution}³\n"
                f"Checkpoint: {payload.checkpoint_path}\n"
                f"Manifest: {payload.manifest_path}\n"
                f"Checkpoint size: {payload.checkpoint_size_bytes} bytes\n"
                f"Dataset signature: {payload.data_signature}\n",
            )
            self.training_log.see(tk.END)
            messagebox.showinfo("Neural training complete", payload.message)
            return

        self.repair_button.configure(state=tk.NORMAL)
        self.create_button.configure(state=tk.NORMAL)
        if status == "error":
            self.status.set("Repair failed.")
            self._set_progress(0, "Failed")
            messagebox.showerror("Repair failed", str(payload))
            return

        if status == "created":
            output_path, scale_mm, print_detail_um = payload
            output_path = Path(output_path)
            self.input_path.set(str(output_path))
            self.output_path.set(str(output_path.with_name(f"{output_path.stem}_meshmend{output_path.suffix}")))
            self.status.set(f"Created model: {output_path.name}")
            self._set_progress(100, "Model creation complete")
            detail_summary = self._detail_summary_for_path(output_path, float(print_detail_um))
            self._append_log(
                f"Created {float(scale_mm):g}mm 3D model from chat/image:\n{output_path}\n\n{detail_summary}\n\nIt is now selected as the input model for AI repair.\n"
            )
            messagebox.showinfo("Model created", f"Created model:\n{output_path}\n\nIt is ready for repair or export.")
            return

        result = payload
        self.status.set(f"Saved repaired model to {result.report.output_path}")
        self._set_progress(100, "Repair complete")
        self.report_text.insert(tk.END, result.explanation)
        messagebox.showinfo("Repair complete", f"Saved repaired model to:\n{result.report.output_path}")

    def _post_progress(self, percent: int, message: str) -> None:
        self._result_queue.put(("progress", (percent, message)))

    def _set_progress(self, percent: int, message: str) -> None:
        self.progress.configure(value=max(0, min(100, int(percent))))
        self.progress_status.set(f"{max(0, min(100, int(percent)))}% - {message}")

    def _append_log(self, text: str) -> None:
        self.report_text.insert(tk.END, text)
        self.report_text.see(tk.END)

    @staticmethod
    def _detail_summary_for_path(path: Path, print_detail_um: float = 100.0) -> str:
        try:
            import trimesh

            mesh = trimesh.load(path, force="mesh", process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.geometry.values())
            return assess_8k_detail(mesh, target_pitch_mm=print_detail_um / 1000.0).summary()
        except Exception as exc:
            return f"High-resolution detail check unavailable: {exc}"

    def _print_detail_um(self) -> float:
        value = float(self.print_detail_um.get().strip())
        if value < 25 or value > 100:
            raise ValueError("Print detail must be between 25 and 100 um.")
        return value

    def _creation_scale_mm(self) -> float:
        text = self.creation_scale_mm.get().strip().lower().removesuffix("mm").strip()
        value = float(text)
        if value < 10 or value > 100:
            raise ValueError("Miniature scale must be between 10mm and 100mm.")
        return value

    @staticmethod
    def _positive_int(value: str, label: str) -> int:
        parsed = int((value or "").strip())
        if parsed <= 0:
            raise ValueError(f"{label} must be greater than zero.")
        return parsed

    @staticmethod
    def _optional_float(value: str, *, allow_auto: bool) -> float | None:
        text = (value or "").strip()
        if not text or (allow_auto and text.lower() == "auto"):
            return None
        parsed = float(text)
        if parsed <= 0:
            raise ValueError("Numeric options must be greater than zero.")
        return parsed


def launch_gui() -> int:
    return MeshMendGUI().run()


if __name__ == "__main__":
    raise SystemExit(launch_gui())
