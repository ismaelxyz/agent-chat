import flet as ft
from flet import Colors, Icons
from agent_chat.controllers.chat_controller import ChatController
from agent_chat.models import ChatBotModel

from .chat_view import ChatView
from .config_view import ConfigView

__all__ = ["run", "ChatView", "ConfigView"]


def run(page: ft.Page):
    page.title = "Chatbot"
    page.bgcolor = Colors.BLUE_GREY_50
    page.window_width = 480
    page.window_height = 720
    page.padding = 0

    # Global State
    model = ChatBotModel()
    controller = ChatController(model)

    # Views
    chat_view = ChatView(controller)
    config_view = ConfigView(model)
    chat_view.visible = True
    config_view.visible = False

    # App bar + Drawer
    def open_drawer(_):
        page.drawer.open = True
        page.update()

    # Small status near title if no active model
    status_text = ft.Text("", size=12, color=Colors.RED_700)

    def _refresh_app_status():
        try:
            active = getattr(model, "has_active_model")()
        
        except Exception:
            active = False

        status_text.value = "Actualmente no hay un modelo funcionando" if not active else ""
        page.update()

    page.appbar = ft.AppBar(
        title=ft.Row([
            ft.Text("Agent Chat"),
            ft.Container(width=8),
            status_text,
        ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        bgcolor=Colors.BLUE_100,
        leading=ft.IconButton(Icons.MENU, on_click=open_drawer),
    )

    def go_chat(_):
        chat_view.visible = True
        config_view.visible = False
        page.drawer.open = False
        page.update()

    _refresh_app_status()
    page.update()

    def go_config(_):
        chat_view.visible = False
        config_view.visible = True
        page.drawer.open = False
        page.update()

    page.drawer = ft.NavigationDrawer(
        controls=[
            ft.NavigationDrawerDestination(icon=Icons.CHAT, label="Chat"),
            ft.NavigationDrawerDestination(icon=Icons.SETTINGS, label="Configuración"),
        ],
        on_change=lambda e: (go_chat(e) if e.control.selected_index == 0 else go_config(e))
    )

    # Layout principal
    page.add(ft.Column([chat_view, config_view], expand=True))

    # Preload model on startup
    async def _preload():
        try:
            chat_view.set_loading(True)
            # Attempt to restore from persisted state (frontend may not be ready yet)
            cs = page.client_storage
            model_path = None
            try:
                # Prefer async API and cap wait to avoid hanging
                model_path = await __import__("asyncio").wait_for(
                    cs.get_async("chat_model_path"), timeout=1.5
                )
            except Exception:
                # Graceful fallback: skip restore and continue
                model_path = None
            # use_gen = cs.get("chat_use_generated")

            print("Restoring model from:", model_path)
            if model_path:
                controller.select_model(str(model_path))

            else:
                # Fallback: pick latest generated automatically
                try:
                    from pathlib import Path
                    gen_dir = Path("storage/generated_models")
                    if gen_dir.exists():
                        models = list(gen_dir.glob("*.keras")) + list(gen_dir.glob("*.h5"))
                        models = sorted(models, key=lambda p: p.stat().st_mtime, reverse=True)
                        if models:
                            controller.select_model(str(models[0]))
                except Exception:
                    pass
        finally:
            chat_view.set_loading(False)
            _refresh_app_status()

    page.run_task(_preload)

    # Hook model changes from ConfigView to update status
    # When user selects a model or training completes, call refresh
    orig_select_model = controller.select_model
    def _select_model_and_refresh(p):
        orig_select_model(p)
        _refresh_app_status()
        
    controller.select_model = _select_model_and_refresh

    # Also wrap model.set_model_path so direct calls trigger status update
    orig_model_set = model.set_model_path
    def _model_set_and_refresh(p):
        orig_model_set(p)
        _refresh_app_status()
    model.set_model_path = _model_set_and_refresh


