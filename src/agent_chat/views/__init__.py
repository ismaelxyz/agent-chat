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

    # Estado global
    model = ChatBotModel()
    controller = ChatController(model)

    # Vistas
    chat_view = ChatView(controller)
    config_view = ConfigView(model)
    chat_view.visible = True
    config_view.visible = False

    # App bar + Drawer
    def open_drawer(_):
        page.drawer.open = True
        page.update()

    page.appbar = ft.AppBar(
        title=ft.Text("Agent Chat"),
        bgcolor=Colors.BLUE_100,
        leading=ft.IconButton(Icons.MENU, on_click=open_drawer),
    )

    def go_chat(_):
        chat_view.visible = True
        config_view.visible = False
        page.drawer.open = False
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

