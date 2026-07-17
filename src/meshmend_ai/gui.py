from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser

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

_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))
from meshmend.ai import GenerationRequest, get_adapter
from meshmend.core import add_circular_base, auto_scale_to_height, build_printability_report, decimate_mesh, load_mesh, remesh_subdivide
from meshmend.export import export_slicer_ready
from meshmend.repair import repair_mesh


MESH_FILE_TYPES = (
    ("3D mesh files", "*.stl *.obj *.glb *.ply"),
    ("STL files", "*.stl"),
    ("OBJ files", "*.obj"),
    ("GLB files", "*.glb"),
    ("PLY files", "*.ply"),
    ("All files", "*.*"),
)

DONATION_URL = "https://buy.stripe.com/28EbJ1f7ceo3ckyeES5kk00"
DONATION_SUPPRESS_FILE = Path.home() / ".meshmend_ai" / "hide_donation_popup"


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
        self.local_adapter = tk.StringVar(value="existing")
        self.local_target_faces = tk.StringVar(value="180000")
        self.local_add_base = tk.BooleanVar(value=True)
        self.local_repair = tk.BooleanVar(value=True)
        self.use_perseus = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Choose a 3D model to repair.")
        self.progress_status = tk.StringVar(value="Idle")
        self.training_status = tk.StringVar(value="Choose a folder of STL/OBJ/PLY files and optional matching images.")
        self._result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.sculptor = get_sculptor_foundation()
        self._donation_dialog: tk.Toplevel | None = None
        self._creation_progress_dialog: tk.Toplevel | None = None
        self._creation_progress_label: tk.StringVar | None = None
        self._creation_progress_bar: ttk.Progressbar | None = None
        self._build_ui()
        if os.environ.get("MESHMEND_DONATION_POPUP", "1").strip().lower() not in {"0", "false", "no", "off"}:
            self.root.after(700, self._show_donation_popup)

    def run(self) -> int:
        self.root.mainloop()
        return 0

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X)
        title = ttk.Label(header, text="MeshMend AI Repair", font=("Segoe UI", 16, "bold"))
        title.pack(side=tk.LEFT, anchor=tk.W)
        ttk.Button(header, text="Donate / Support MeshMend", command=self.open_donation_page).pack(side=tk.RIGHT)
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
        self.restart_service_button = ttk.Button(sculptor_buttons, text="Restart model service", command=self.restart_model_service)
        self.restart_service_button.pack(side=tk.LEFT, padx=(8, 0))
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
        ttk.Label(detail_row, text="um  (50=studio, 25=max detail)").pack(side=tk.LEFT, padx=(6, 14))
        ttk.Label(detail_row, text="Triangle cap").pack(side=tk.LEFT)
        ttk.Entry(detail_row, textvariable=self.max_detail_triangles, width=12).pack(side=tk.LEFT, padx=(6, 0))
        image_row = ttk.Frame(creator)
        image_row.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Label(image_row, text="Image", width=10).pack(side=tk.LEFT)
        ttk.Entry(image_row, textvariable=self.creation_image_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(image_row, text="Browse", command=self.choose_creation_image).pack(side=tk.LEFT)
        creator_buttons = ttk.Frame(creator)
        creator_buttons.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.create_button = ttk.Button(creator_buttons, text="Create local studio model", command=self.start_create_model)
        self.create_button.pack(side=tk.LEFT)

        local_mvp = ttk.LabelFrame(workflow_tab, text="Local MVP workbench — no paid API required")
        local_mvp.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(
            local_mvp,
            text="Run the new fully-local MeshMend MVP tools from this GUI: adapter generation, repair, scale/base, export, and printability report.",
            wraplength=700,
        ).pack(anchor=tk.W, padx=10, pady=(10, 6))
        local_row = ttk.Frame(local_mvp)
        local_row.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Label(local_row, text="Adapter", width=10).pack(side=tk.LEFT)
        ttk.Combobox(
            local_row,
            textvariable=self.local_adapter,
            values=("existing", "placeholder"),
            width=14,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(local_row, text="Scale", width=6).pack(side=tk.LEFT)
        ttk.Combobox(
            local_row,
            textvariable=self.creation_scale_mm,
            values=("28", "32", "75"),
            width=8,
        ).pack(side=tk.LEFT)
        ttk.Label(local_row, text="mm").pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(local_row, text="Target faces").pack(side=tk.LEFT)
        ttk.Entry(local_row, textvariable=self.local_target_faces, width=12).pack(side=tk.LEFT, padx=(6, 14))
        ttk.Checkbutton(local_row, text="Repair", variable=self.local_repair).pack(side=tk.LEFT)
        ttk.Checkbutton(local_row, text="Base", variable=self.local_add_base).pack(side=tk.LEFT, padx=(8, 0))
        local_buttons = ttk.Frame(local_mvp)
        local_buttons.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(local_buttons, text="Open 3D MVP Workbench", command=self.open_local_mvp_workbench).pack(side=tk.LEFT)
        ttk.Button(local_buttons, text="Printability Report", command=self.start_local_printability_report).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(local_buttons, text="Local Repair/Scale/Base/Export", command=self.start_local_mvp_export).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(local_buttons, text="Generate Local MVP Miniature", command=self.start_local_mvp_generate).pack(side=tk.LEFT, padx=(8, 0))

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

    def _show_donation_popup(self) -> None:
        if DONATION_SUPPRESS_FILE.exists() or self._donation_dialog is not None:
            return

        dialog = tk.Toplevel(self.root)
        self._donation_dialog = dialog
        dialog.title("Support MeshMend")
        dialog.resizable(False, False)
        dialog.transient(self.root)

        frame = ttk.Frame(dialog, padding=18)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Help keep MeshMend free", font=("Segoe UI", 13, "bold")).pack(anchor=tk.W)
        ttk.Label(
            frame,
            text=(
                "MeshMend is currently a free project working toward studio-level AI miniature "
                "creation for home users. Donations help fund model-service improvements, quality "
                "updates, testing, and future features while keeping the project free."
            ),
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(8, 12))

        hide_popup = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Don't show this again", variable=hide_popup).pack(anchor=tk.W, pady=(0, 12))

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X)

        def close_dialog() -> None:
            if hide_popup.get():
                try:
                    DONATION_SUPPRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
                    DONATION_SUPPRESS_FILE.write_text("hidden\n", encoding="utf-8")
                except OSError:
                    pass
            self._donation_dialog = None
            dialog.destroy()

        def donate() -> None:
            self.open_donation_page()
            close_dialog()

        ttk.Button(buttons, text="Donate", command=donate).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Maybe later", command=close_dialog).pack(side=tk.RIGHT, padx=(0, 8))
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)

        self.root.update_idletasks()
        dialog.update_idletasks()
        x = self.root.winfo_x() + max((self.root.winfo_width() - dialog.winfo_width()) // 2, 0)
        y = self.root.winfo_y() + max((self.root.winfo_height() - dialog.winfo_height()) // 3, 0)
        dialog.geometry(f"+{x}+{y}")
        dialog.lift(self.root)

    @staticmethod
    def open_donation_page() -> None:
        webbrowser.open(DONATION_URL, new=2)

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

    def restart_model_service(self) -> None:
        self.restart_service_button.configure(state=tk.DISABLED)
        self.create_button.configure(state=tk.DISABLED)
        self.status.set("Restarting MeshMend model service...")
        self._set_progress(0, "Stopping model service")
        self._append_log("Restarting local model service...\n")
        thread = threading.Thread(target=self._restart_model_service_worker, daemon=True)
        thread.start()
        self.root.after(100, self._poll_result)

    def _restart_model_service_worker(self) -> None:
        try:
            cli_path = Path(__file__).resolve().parent / "cli.py"
            if not cli_path.exists():
                raise RuntimeError(f"Could not find cli.py at {cli_path}")
            commands = [
                ([sys.executable, str(cli_path), "--stop-model-service"], 20, "Stopping model service"),
                ([sys.executable, str(cli_path), "--model-service", "--restart-model-service"], 85, "Starting model service"),
            ]
            logs: list[str] = []
            for command, percent, message in commands:
                self._post_progress(percent, message)
                completed = subprocess.run(
                    command,
                    cwd=str(cli_path.parent),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=float(os.environ.get("MESHMEND_GUI_SERVICE_RESTART_TIMEOUT_SECONDS", "180")),
                )
                logs.append("> " + " ".join(command) + "\n" + (completed.stdout or "") + (completed.stderr or ""))
                if completed.returncode != 0:
                    raise RuntimeError(logs[-1].strip() or f"model service command exited {completed.returncode}")
            self._post_progress(100, "Model service restarted")
            self._result_queue.put(("service_restarted", "\n".join(logs)))
        except Exception as exc:
            self._result_queue.put(("service_error", exc))

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

    def open_local_mvp_workbench(self) -> None:
        try:
            subprocess.Popen([sys.executable, "-m", "meshmend"], cwd=str(Path(__file__).resolve().parent))
            self.status.set("Opened local MVP workbench.")
        except Exception as exc:
            messagebox.showerror("Local MVP workbench", str(exc))

    def start_local_printability_report(self) -> None:
        try:
            input_path = Path(self.input_path.get()).expanduser()
            if not input_path.exists():
                raise ValueError("Choose an existing input model first.")
        except Exception as exc:
            messagebox.showerror("Printability report", str(exc))
            return
        self._set_progress(0, "Running local printability report")
        self.status.set("Inspecting printability...")
        thread = threading.Thread(target=self._local_printability_worker, args=(input_path,), daemon=True)
        thread.start()
        self.root.after(100, self._poll_result)

    def _local_printability_worker(self, input_path: Path) -> None:
        try:
            mesh = load_mesh(input_path)
            report = build_printability_report(mesh)
            self._result_queue.put(("mvp_report", {"input": str(input_path), "printability": report.to_dict()}))
        except Exception as exc:
            self._result_queue.put(("mvp_error", exc))

    def start_local_mvp_export(self) -> None:
        try:
            input_path = Path(self.input_path.get()).expanduser()
            output_path = Path(self.output_path.get()).expanduser()
            if not input_path.exists():
                raise ValueError("Choose an existing input model first.")
            if not output_path.name:
                raise ValueError("Choose where to save the exported model.")
            scale_mm = self._creation_scale_mm()
            target_faces = self._positive_int(self.local_target_faces.get(), "Target faces")
        except Exception as exc:
            messagebox.showerror("Local MVP export", str(exc))
            return
        self.repair_button.configure(state=tk.DISABLED)
        self.create_button.configure(state=tk.DISABLED)
        self.report_text.delete("1.0", tk.END)
        self._set_progress(0, "Starting local MVP export")
        self.status.set("Running local MVP mesh pipeline...")
        thread = threading.Thread(
            target=self._local_mvp_export_worker,
            args=(input_path, output_path, scale_mm, target_faces, self.local_repair.get(), self.local_add_base.get()),
            daemon=True,
        )
        thread.start()
        self.root.after(100, self._poll_result)

    def _local_mvp_export_worker(
        self,
        input_path: Path,
        output_path: Path,
        scale_mm: float,
        target_faces: int,
        run_repair: bool,
        add_base: bool,
    ) -> None:
        try:
            self._post_progress(10, "Loading mesh")
            mesh = load_mesh(input_path)
            repair_actions: list[str] = []
            if run_repair:
                self._post_progress(30, "Repairing with existing MeshMend repair engine")
                repaired = repair_mesh(mesh)
                mesh = repaired.mesh
                repair_actions = repaired.actions
            self._post_progress(50, f"Scaling to {scale_mm:g}mm")
            mesh = auto_scale_to_height(mesh, scale_mm)
            if add_base:
                self._post_progress(62, "Adding circular base")
                mesh = add_circular_base(mesh)
            if target_faces > 0:
                if len(mesh.faces) > target_faces:
                    self._post_progress(74, f"Decimating toward {target_faces:,} faces")
                    mesh = decimate_mesh(mesh, target_faces)
                elif len(mesh.faces) < target_faces:
                    self._post_progress(74, f"Remeshing toward {target_faces:,} faces")
                    mesh = remesh_subdivide(mesh, target_faces)
            self._post_progress(88, "Exporting slicer-ready mesh")
            export_slicer_ready(mesh, output_path)
            report = build_printability_report(mesh).to_dict()
            self._post_progress(100, "Local MVP export complete")
            self._result_queue.put(("mvp_exported", {"output": str(output_path), "actions": repair_actions, "printability": report}))
        except Exception as exc:
            self._result_queue.put(("mvp_error", exc))

    def start_local_mvp_generate(self) -> None:
        prompt = self.chat_prompt.get("1.0", tk.END).strip()
        try:
            if not prompt:
                raise ValueError("Enter a structured prompt first.")
            output_text = self.output_path.get().strip()
            if output_text:
                output_path = Path(output_text).expanduser()
            else:
                selected = filedialog.asksaveasfilename(
                    title="Save generated local MVP model as",
                    defaultextension=".stl",
                    filetypes=MESH_FILE_TYPES,
                )
                if not selected:
                    return
                output_path = Path(selected).expanduser()
                self.output_path.set(str(output_path))
            scale_mm = self._creation_scale_mm()
            target_faces = self._positive_int(self.local_target_faces.get(), "Target faces")
            adapter_name = self.local_adapter.get().strip() or "existing"
        except Exception as exc:
            messagebox.showerror("Local MVP generation", str(exc))
            return
        self.repair_button.configure(state=tk.DISABLED)
        self.create_button.configure(state=tk.DISABLED)
        self.report_text.delete("1.0", tk.END)
        self._set_progress(0, "Starting local MVP generation")
        self.status.set("Generating local MVP miniature...")
        thread = threading.Thread(
            target=self._local_mvp_generate_worker,
            args=(prompt, output_path, adapter_name, scale_mm, target_faces, self.local_repair.get(), self.local_add_base.get()),
            daemon=True,
        )
        thread.start()
        self.root.after(100, self._poll_result)

    def _local_mvp_generate_worker(
        self,
        prompt: str,
        output_path: Path,
        adapter_name: str,
        scale_mm: float,
        target_faces: int,
        run_repair: bool,
        add_base: bool,
    ) -> None:
        try:
            self._post_progress(15, f"Generating with local adapter: {adapter_name}")
            mesh = get_adapter(adapter_name).generate(
                GenerationRequest(prompt=prompt, height_mm=scale_mm, target_faces=target_faces, add_base=add_base)
            )
            repair_actions: list[str] = []
            if run_repair:
                self._post_progress(70, "Repairing generated mesh")
                repaired = repair_mesh(mesh)
                mesh = repaired.mesh
                repair_actions = repaired.actions
            self._post_progress(88, "Exporting generated mesh")
            export_slicer_ready(mesh, output_path)
            report = build_printability_report(mesh).to_dict()
            self._post_progress(100, "Local MVP generation complete")
            self._result_queue.put(("mvp_generated", {"output": str(output_path), "adapter": adapter_name, "actions": repair_actions, "printability": report}))
        except Exception as exc:
            self._result_queue.put(("mvp_error", exc))

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
                force_bridge=False,
            )
            self._post_progress(38, "Repairing structural mesh defects without smoothing or remeshing")
            result = assistant.repair(
                input_path,
                output_path,
                connector_radius=connector_radius,
                max_bridge_distance=max_bridge_distance,
                force_bridge=False,
            )
            self._post_progress(82, "Running explanation and learning layer")
            if use_perseus and result.perseus is None:
                self._post_progress(88, "Perseus unavailable; using native explanation")
            self._post_progress(100, "Repair complete")
            self._result_queue.put(("ok", result))
        except Exception as exc:
            self._result_queue.put(("error", exc))

    def _creation_requests_store_quality(self, prompt: str, image_text: str) -> bool:
        lowered = prompt.lower()
        if image_text.strip():
            return True
        try:
            if self._print_detail_um() <= 50:
                return True
        except Exception:
            pass
        return any(
            term in lowered
            for term in (
                "8k",
                "8 k",
                "studio",
                "studio quality",
                "store quality",
                "store-quality",
                "store level",
                "store-level",
                "production quality",
                "display quality",
                "maximum detail",
                "max detail",
                "high detail",
                "intricate",
                "marketplace",
            )
        )

    def start_create_model(self) -> None:
        prompt = self.chat_prompt.get("1.0", tk.END).strip()
        image_text = self.creation_image_path.get().strip()
        use_legacy_service = os.environ.get("MESHMEND_GUI_USE_LEGACY_MODEL_SERVICE_CREATE", "0").strip().lower() in {"1", "true", "yes", "on"}
        if not use_legacy_service and not self._creation_requests_store_quality(prompt, image_text):
            self.start_local_mvp_generate()
            return
        self._append_log(
            "Routing create request through MeshMend production/store-quality local backend. "
            "Use 'Generate Local MVP Miniature' for procedural draft output.\n"
        )
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
        self._show_creation_progress_dialog("Starting model creation")

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
            self._result_queue.put(("create_error", exc))

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

        if status == "service_restarted":
            self.restart_service_button.configure(state=tk.NORMAL)
            self.create_button.configure(state=tk.NORMAL)
            self.repair_button.configure(state=tk.NORMAL)
            self.status.set("Model service restarted.")
            self._set_progress(100, "Model service restarted")
            if payload:
                self._append_log(str(payload).strip() + "\n")
            messagebox.showinfo("Model service", "MeshMend model service restarted.")
            return

        if status == "service_error":
            self.restart_service_button.configure(state=tk.NORMAL)
            self.create_button.configure(state=tk.NORMAL)
            self.repair_button.configure(state=tk.NORMAL)
            self.status.set("Model service restart failed.")
            self._set_progress(0, "Service restart failed")
            messagebox.showerror("Model service", str(payload))
            return

        if status == "mvp_error":
            self.repair_button.configure(state=tk.NORMAL)
            self.create_button.configure(state=tk.NORMAL)
            self.restart_service_button.configure(state=tk.NORMAL)
            self.status.set("Local MVP operation failed.")
            self._set_progress(0, "Local MVP failed")
            self._append_log(str(payload).strip() + "\n")
            messagebox.showerror("Local MVP", str(payload))
            return

        if status == "mvp_report":
            self.repair_button.configure(state=tk.NORMAL)
            self.create_button.configure(state=tk.NORMAL)
            self.restart_service_button.configure(state=tk.NORMAL)
            self.status.set("Printability report complete.")
            self._set_progress(100, "Printability report complete")
            self.report_text.delete("1.0", tk.END)
            self._append_log(json.dumps(payload, indent=2) + "\n")
            return

        if status in {"mvp_exported", "mvp_generated"}:
            self.repair_button.configure(state=tk.NORMAL)
            self.create_button.configure(state=tk.NORMAL)
            self.restart_service_button.configure(state=tk.NORMAL)
            data = payload if isinstance(payload, dict) else {}
            output = Path(str(data.get("output") or ""))
            if output.name:
                self.input_path.set(str(output))
                self.output_path.set(str(output.with_name(f"{output.stem}_meshmend{output.suffix}")))
            self.status.set("Local MVP generation complete." if status == "mvp_generated" else "Local MVP export complete.")
            self._set_progress(100, "Local MVP complete")
            self.report_text.delete("1.0", tk.END)
            self._append_log(json.dumps(data, indent=2) + "\n")
            messagebox.showinfo("Local MVP", f"Saved model:\n{output}" if output.name else "Local MVP operation complete.")
            return

        self.repair_button.configure(state=tk.NORMAL)
        self.create_button.configure(state=tk.NORMAL)
        self.restart_service_button.configure(state=tk.NORMAL)
        if status == "create_error":
            message = str(payload)
            lowered = message.lower()
            is_concept_failure = "Concept generation" in message or "concept image" in lowered or "text-to-3d stopped before hunyuan" in lowered
            is_store_quality_unavailable = "Store-quality 8K miniature generation is not configured" in message or "Store/studio-quality 8K miniature generation is not available" in message
            is_store_quality_gate = "store-quality gate" in lowered or "without certification" in lowered or "base_form_validated" in lowered or "strict studio requests must return" in lowered or "no_concept_image_received" in lowered or "low_vision_planner_confidence" in lowered or "ai_planner_required" in lowered or "image_to_3d_requires_ai_vision_planner" in lowered or "concept_match" in lowered or "planned_landmarks" in lowered or "multiple_components_" in lowered
            self.status.set("Concept generation failed." if is_concept_failure else "Store-quality gate failed." if is_store_quality_gate else "Store-quality backend not configured." if is_store_quality_unavailable else "Model creation failed.")
            self._set_progress(0, "Concept generation failed" if is_concept_failure else "Store-quality gate failed" if is_store_quality_gate else "Store-quality backend not configured" if is_store_quality_unavailable else "Creation failed")
            self._close_creation_progress_dialog()
            self._append_log(message.strip() + "\n")
            title = "Concept generation failed" if is_concept_failure else "Store-quality gate failed" if is_store_quality_gate else "Store-quality backend not configured" if is_store_quality_unavailable else "Create model failed"
            messagebox.showerror(title, message)
            return

        if status == "error":
            self.status.set("Repair failed.")
            self._set_progress(0, "Failed")
            self._close_creation_progress_dialog()
            messagebox.showerror("Repair failed", str(payload))
            return

        if status == "created":
            output_path, scale_mm, print_detail_um = payload
            output_path = Path(output_path)
            self.input_path.set(str(output_path))
            self.output_path.set(str(output_path.with_name(f"{output_path.stem}_meshmend{output_path.suffix}")))
            self.status.set(f"Created model: {output_path.name}")
            self._set_progress(100, "Model creation complete")
            self._close_creation_progress_dialog()
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
        value = max(0, min(100, int(percent)))
        self.progress.configure(value=value)
        self.progress_status.set(f"{value}% - {message}")
        if self._creation_progress_dialog is not None and self._creation_progress_dialog.winfo_exists():
            if self._creation_progress_label is not None:
                self._creation_progress_label.set(f"{value}% - {message}")
            if self._creation_progress_bar is not None:
                self._creation_progress_bar.configure(value=value)
                self._creation_progress_bar.update_idletasks()

    def _show_creation_progress_dialog(self, message: str) -> None:
        self._close_creation_progress_dialog()
        dialog = tk.Toplevel(self.root)
        self._creation_progress_dialog = dialog
        self._creation_progress_label = tk.StringVar(value=f"0% - {message}")
        dialog.title("Creating 3D model")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.protocol("WM_DELETE_WINDOW", lambda: None)

        frame = ttk.Frame(dialog, padding=18)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Model creation in progress", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(
            frame,
            text="This can take several minutes while MeshMend native geometry, mesh cleanup, solidification, and detail passes run.",
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(8, 10))
        self._creation_progress_bar = ttk.Progressbar(frame, mode="determinate", maximum=100, length=420)
        self._creation_progress_bar.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(frame, textvariable=self._creation_progress_label, wraplength=420).pack(anchor=tk.W)

        self.root.update_idletasks()
        dialog.update_idletasks()
        x = self.root.winfo_x() + max((self.root.winfo_width() - dialog.winfo_width()) // 2, 0)
        y = self.root.winfo_y() + max((self.root.winfo_height() - dialog.winfo_height()) // 3, 0)
        dialog.geometry(f"+{x}+{y}")
        dialog.lift(self.root)

    def _close_creation_progress_dialog(self) -> None:
        if self._creation_progress_dialog is not None:
            try:
                if self._creation_progress_dialog.winfo_exists():
                    self._creation_progress_dialog.destroy()
            except tk.TclError:
                pass
        self._creation_progress_dialog = None
        self._creation_progress_label = None
        self._creation_progress_bar = None

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
