import flet as ft

from flet import Colors, Icons
# from agent_chat.controllers import ChatController
from pathlib import Path
import asyncio
from agent_chat.models import ChatBotModel


class ConfigView(ft.Container):
    def __init__(self, model: ChatBotModel):
        # Estado
        self.model = model
        self.selected_model_path: Path | None = None
        self.intents_last_confirmed: str | None = None
        self.last_trained_text: str | None = None
        self.edited_since_confirm: bool = False

        # Controles
        self.model_path_text = ft.Text("Ningún modelo seleccionado")
        self.pick_model_btn = ft.ElevatedButton("Elegir modelo (.h5)", icon=Icons.FOLDER_OPEN)
        self.file_picker = ft.FilePicker(on_result=self._on_pick_model)
        self.progress_bar = ft.ProgressBar(width=400, visible=False)
        self.train_btn = ft.ElevatedButton("Entrenar", disabled=True, icon=Icons.PLAY_ARROW)
        self.confirm_btn = ft.OutlinedButton("Confirmar", icon=Icons.CHECK_CIRCLE)
        self.edit_again_btn = ft.TextButton("Editar de nuevo", icon=Icons.EDIT)
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
        # Tabs para modo
        self.tabs = ft.Tabs(
            selected_index=0,
            tabs=[
                ft.Tab(text="Usar local"),
                ft.Tab(text="Generar nuevo"),
            ],
            on_change=self._on_mode_change,
        )

        # Eventos
        self.pick_model_btn.on_click = lambda _: self.file_picker.pick_files(allow_multiple=False)
        self.intents_editor.on_change = self._on_editor_change
        self.confirm_btn.on_click = self._confirm_intents
        self.edit_again_btn.on_click = self._edit_intents_again
        self.train_btn.on_click = self._start_fake_training

        # Composición UI
        local_section = ft.Column([
            ft.Text("Modelo local"),
            ft.Row([self.pick_model_btn], alignment=ft.MainAxisAlignment.START),
            self.model_path_text,
        ], spacing=10)

        generate_section = ft.Column([
            ft.Text("Nuevo modelo desde intents.json"),
            self.intents_editor,
            ft.Row([self.confirm_btn, self.edit_again_btn, self.train_btn], spacing=10),
            self.progress_bar,
        ], spacing=10)

        content = ft.Column([
            ft.Text("Configuración", size=22, weight=ft.FontWeight.BOLD, color=Colors.BLUE_700),
            ft.Divider(),
            self.tabs,
            ft.Divider(),
            local_section,
            ft.Divider(),
            generate_section,
        ], expand=True, spacing=10)

        super().__init__(content=content, padding=20, expand=True)

    def did_mount(self):
        # Adjuntar file picker a la página y cargar estado inicial
        self.page.overlay.append(self.file_picker)
        self._load_intents_json()
        self._select_latest_generated_if_any()

    # --- Helpers ---
    def _generated_dir(self) -> Path:
        d = Path("chatbot/generated_models")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _select_latest_generated_if_any(self):
        # Usa siempre el último generado si existe (por defecto)
        models = sorted(self._generated_dir().glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
        if models:
            latest = models[0]
            self.model.set_model_path(str(latest))
            self.model_path_text.value = f"Usando último generado: {latest}"
            self.update()

    def _load_intents_json(self):
        try:
            path = Path("chatbot/intents.json")
            self.intents_editor.value = path.read_text(encoding="utf-8") if path.exists() else "{\n  \"intents\": []\n}"
        except Exception:
            self.intents_editor.value = "{\n  \"intents\": []\n}"

    def _on_pick_model(self, e: ft.FilePickerResultEvent):
        if e.files:
            p = Path(e.files[0].path)
            self.selected_model_path = p
            self.model.set_model_path(str(p))
            self.model_path_text.value = str(p)
            self.update()

    def _on_editor_change(self, e):
        self.edited_since_confirm = True
        # En modo generar: deshabilitar hasta confirmar
        if self.tabs.selected_index == 1:
            self.train_btn.disabled = True
        self.update()

    def _confirm_intents(self, _):
        self.intents_last_confirmed = self.intents_editor.value
        self.edited_since_confirm = False
        # Ocultar texto hasta nueva edición
        self.intents_editor.visible = False
        # Habilitar entrenar sólo si hay cambios respecto al último entrenado
        self.train_btn.disabled = (self.last_trained_text == self.intents_last_confirmed)
        self.update()

    def _edit_intents_again(self, _):
        self.intents_editor.visible = True
        self.train_btn.disabled = True
        self.update()

    def _on_mode_change(self, _):
        # Si está en modo local (0), la sección de entrenamiento queda deshabilitada
        is_local = self.tabs.selected_index == 0
        self.intents_editor.disabled = is_local
        self.confirm_btn.disabled = is_local
        self.edit_again_btn.disabled = is_local
        self.train_btn.disabled = True  # siempre al cambiar de modo
        self.update()

    async def _fake_training_async(self):
        # Entrenamiento simulado con barra de progreso y creación de archivo .h5
        self.train_btn.disabled = True
        self.progress_bar.visible = True
        self.progress_bar.value = 0
        self.update()
        for i in range(11):
            self.progress_bar.value = i / 10
            self.update()
            await asyncio.sleep(0.15)
        self.progress_bar.visible = False
        # Guardar intents confirmados (mock) y "generar" archivo modelo
        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self._generated_dir() / f"model_{ts}.h5"
        out_path.write_text("mock model bytes", encoding="utf-8")
        # Seleccionar el último generado como activo
        self.model.set_model_path(str(out_path))
        self.page.snack_bar = ft.SnackBar(ft.Text(f"Entrenamiento listo. Modelo: {out_path.name}"))
        self.page.snack_bar.open = True
        # Marcar como último entrenado
        self.last_trained_text = self.intents_last_confirmed
        self.update()

    def _start_fake_training(self, _):
        # Sólo permitir si está en modo generar
        if self.tabs.selected_index != 1:
            return
        self.page.run_task(self._fake_training_async)

