import flet as ft
from flet import Colors, Icons
from pathlib import Path
import asyncio
import json

from agent_chat.models import ChatBotModel
from agent_chat.models import nlp


class ConfigView(ft.Container):
    def __init__(self, model: ChatBotModel):
        # State
        self.model = model
        self.selected_model_path: Path | None = None
        self.intents_last_confirmed: str | None = None
        self.last_trained_text: str | None = None
        self.edited_since_confirm: bool = False
        
        # Controls
        self.model_path_text = ft.Text("No model selected")
        self.pick_model_btn = ft.ElevatedButton("Choose model (.keras or .h5)", icon=Icons.FOLDER_OPEN)
        self.file_picker = ft.FilePicker(on_result=self._on_pick_model)
        self.progress_bar = ft.ProgressBar(width=400, visible=False)
        self.train_btn = ft.ElevatedButton("Train", disabled=True, icon=Icons.PLAY_ARROW)
        self.confirm_btn = ft.OutlinedButton("Confirm", icon=Icons.CHECK_CIRCLE)
        self.edit_again_btn = ft.TextButton("Edit again", icon=Icons.EDIT)

        # Mode: checkbox instead of tabs
        self.use_generate = ft.Checkbox(
            label="Generate new from intents.json",
            value=False,
            on_change=self._on_mode_change,
        )

        # Custom model meta
        self.custom_version_text = ft.Text(value="", visible=False, color=Colors.BLUE_700)
        self.custom_label_field = ft.TextField(
            label="Model label",
            value="",
            visible=False,
            on_change=self._on_label_change,
        )
        self.intents_editor = ft.TextField(
            label="intents.json",
            value="",
            multiline=True,
            min_lines=10,
            max_lines=18,
            expand=True,
            visible=True,
            bgcolor=Colors.WHITE,
        )

        # Events
        self.pick_model_btn.on_click = lambda _: self.file_picker.pick_files(allow_multiple=False)
        self.intents_editor.on_change = self._on_editor_change
        self.confirm_btn.on_click = self._confirm_intents
        self.edit_again_btn.on_click = self._edit_intents_again
        self.train_btn.on_click = self._start_training

        # UI composition
        self.local_section = ft.Column([
            ft.Text("Local model"),
            ft.Row([self.pick_model_btn], alignment=ft.MainAxisAlignment.START),
            self.model_path_text,
        ], spacing=10)

        self.generate_section = ft.Column([
            ft.Text("New model from intents.json"),
            self.intents_editor,
            ft.Row([self.custom_version_text], alignment=ft.MainAxisAlignment.START),
            self.custom_label_field,
            ft.Row([self.confirm_btn, self.edit_again_btn, self.train_btn], spacing=10),
            self.progress_bar,
        ], spacing=10, expand=True, scroll=ft.ScrollMode.AUTO)

        content = ft.Column([
            ft.Text("Configuration", size=22, weight=ft.FontWeight.BOLD, color=Colors.BLUE_700),
            ft.Divider(),
            ft.Row([self.use_generate]),
            ft.Divider(),
            self.local_section,
            ft.Divider(),
            self.generate_section,
        ], expand=True, spacing=10)

        super().__init__(content=content, padding=20, expand=True)

    def did_mount(self):
        # Attach file picker to page and load initial state
        self.page.overlay.append(self.file_picker)
        # Restore persisted state via client_storage
        try:
            self._restore_persisted_state()  # defined below
        except Exception:
            pass
        self._load_intents_json()
        # Only auto-select latest if no model was restored
        try:
            if not getattr(self.model, "model_path", None):
                self._select_latest_generated_if_any()
        except Exception:
            self._select_latest_generated_if_any()
        # Apply initial mode state
        self._on_mode_change(None)

    # --- Helpers ---
    def _generated_dir(self) -> Path:
        d = Path("storage/generated_models")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _select_latest_generated_if_any(self):
        # Use latest generated if present (by default)
        # Look for latest generated model in native Keras format first, then legacy H5
        models = list(self._generated_dir().glob("*.keras")) + list(self._generated_dir().glob("*.h5"))
        models = sorted(models, key=lambda p: p.stat().st_mtime, reverse=True)
        if models:
            latest = models[0]
            self.model.set_model_path(str(latest))
            self.model_path_text.value = f"Using latest generated: {latest}"
            self.update()

    def _load_intents_json(self):
        try:
            path = Path("storage/intents.json")
            self.intents_editor.value = path.read_text(encoding="utf-8") if path.exists() else "{\n  \"intents\": []\n}"
        except Exception:
            self.intents_editor.value = "{\n  \"intents\": []\n}"

    def _on_pick_model(self, e: ft.FilePickerResultEvent):
        if e.files:
            p = Path(e.files[0].path)
            self.selected_model_path = p
            self.model.set_model_path(str(p))
            self.model_path_text.value = str(p)
            # Persist selection
            try:
                self._persist_state()
            except Exception:
                pass
            self.update()

    def _on_editor_change(self, e):
        self.edited_since_confirm = True
        # In generate mode: disable until confirmed
        if self.use_generate.value:
            self.train_btn.disabled = True
        self.update()

    def _confirm_intents(self, _):
        self.intents_last_confirmed = self.intents_editor.value
        self.edited_since_confirm = False
        # Hide text until next edit
        self.intents_editor.visible = False
        # Enable training only if changed since last trained
        self.train_btn.disabled = (self.last_trained_text == self.intents_last_confirmed)
        self.update()

    def _edit_intents_again(self, _):
        self.intents_editor.visible = True
        self.train_btn.disabled = True
        self.update()

    def _on_mode_change(self, _):
        # Toggle between local model (off) and generate new (on)
        is_local = not self.use_generate.value
        try:
            self.model.set_use_generated(not is_local)
        except Exception:
            pass
        # Persist toggle
        try:
            self._persist_state()
        except Exception:
            pass
        # Section visibility
        self.local_section.visible = is_local
        self.generate_section.visible = not is_local
        # Version/label visible only in generate mode
        self.custom_version_text.visible = not is_local
        self.custom_label_field.visible = not is_local
        if not is_local:
            # Sync fields with current model
            self._refresh_custom_meta()
        # Controls state
        self.intents_editor.disabled = is_local
        self.confirm_btn.disabled = is_local
        self.edit_again_btn.disabled = is_local
        self.train_btn.disabled = True  # always when switching mode
        self.update()

    def _refresh_custom_meta(self):
        # Update version text and label value from model
        ver = getattr(self.model, "custom_version", 0)
        label = getattr(self.model, "custom_label", "") or ""
        self.custom_version_text.value = f"Model version: v{ver}"
        self.custom_label_field.value = label
        # Keep current visibility
        self.update()

    def _on_label_change(self, e):
        # Update label as user types
        self.model.set_custom_meta(label=self.custom_label_field.value)
        # Persist label change
        try:
            self._persist_state()
        except Exception:
            pass
        self._refresh_custom_meta()

    async def _train_async(self):
        # Real training if Keras is available; otherwise simulated
        self.train_btn.disabled = True
        self.progress_bar.visible = True
        self.progress_bar.value = 0
        self.update()
        print("[ui] Inicio de entrenamiento")
        # Write intents to disk
        intents_path = Path("storage/intents.json")
        try:
            intents_data = nlp.load_intents(intents_path) if intents_path.exists() else None
        except Exception:
            intents_data = None
        if self.intents_last_confirmed:
            try:
                nlp.save_intents(json.loads(self.intents_last_confirmed), intents_path)
                intents_data = nlp.load_intents(intents_path)
            except Exception:
                pass

        async def _tick_progress(target: float):
            # Smoothly animate progress to target
            step = 0.02
            while self.progress_bar.value < target:
                self.progress_bar.value = min(target, self.progress_bar.value + step)
                self.update()
                await asyncio.sleep(0.05)

        model_path: Path | None = None
        try:
            if not intents_data:
                raise ValueError("Intents empty")
            # Initial progress
            await _tick_progress(0.1)
            # Real training
            artifacts = await asyncio.get_running_loop().run_in_executor(
                None, lambda: nlp.train_and_save(intents_data, self._generated_dir())
            )
            model_path = artifacts.model_path
            await _tick_progress(0.95)
        except RuntimeError:
            # No Keras backend: simulated fallback + save vocab
            for i in range(11):
                self.progress_bar.value = i / 10
                self.update()
                await asyncio.sleep(0.12)
            ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
            # Simulated artifact uses .keras extension to align with native format
            model_path = self._generated_dir() / f"model_{ts}.keras"
            model_path.write_text("mock model bytes", encoding="utf-8")
            # Persist vocab to allow inference with real files
            try:
                if intents_data:
                    nlp.write_vocab_sidecars_from_intents(intents_data, model_path)
            except Exception:
                pass
        except Exception:
            # Generic error: close bar and notify
            self.progress_bar.visible = False
            self.page.snack_bar = ft.SnackBar(ft.Text("Training error. Check intents.json"))
            self.page.snack_bar.open = True
            self.update()
            return
        finally:
            self.progress_bar.visible = False
            print("[ui] Fin de entrenamiento")

        # Activate the generated model
        if model_path:
            self.model.set_model_path(str(model_path))
        self.model.bump_version(label=self.custom_label_field.value)
        # Persist new model and meta
        try:
            self._persist_state()
        except Exception:
            pass
        self._refresh_custom_meta()
        self.page.snack_bar = ft.SnackBar(ft.Text(f"Training complete. Model: {model_path.name}"))
        self.page.snack_bar.open = True
        self.last_trained_text = self.intents_last_confirmed
        self.update()

    def _start_training(self, _):
        if not self.use_generate.value:
            return
        self.page.run_task(self._train_async)

    # --- Persistence via client_storage ---
    def _persist_state(self):
        cs = self.page.client_storage
        cs.set("chat_use_generated", "1" if getattr(self.model, "use_generated", False) else "0")
        cs.set("chat_model_path", getattr(self.model, "model_path", "") or "")
        cs.set("chat_custom_version", str(getattr(self.model, "custom_version", 0)))
        cs.set("chat_custom_label", getattr(self.model, "custom_label", "") or "")

    def _restore_persisted_state(self):
        cs = self.page.client_storage
        use_gen = cs.get("chat_use_generated")
        if use_gen is not None:
            self.use_generate.value = str(use_gen).lower() in ("1", "true", "yes")
            try:
                self.model.set_use_generated(self.use_generate.value)
            except Exception:
                pass
        ver = cs.get("chat_custom_version")
        if ver is not None:
            try:
                self.model.set_custom_meta(version=int(str(ver)))
            except Exception:
                pass
        label = cs.get("chat_custom_label")
        if label is not None:
            try:
                self.model.set_custom_meta(label=str(label))
            except Exception:
                pass
        model_path = cs.get("chat_model_path")
        if model_path:
            p = Path(str(model_path))
            if p.exists():
                self.selected_model_path = p
                self.model.set_model_path(str(p))
                self.model_path_text.value = str(p)

