import flet as ft
from flet import Colors, Icons
from .controller import ChatController
from .model import ChatBotModel

def run(page: ft.Page):
    page.title = "Chatbot Básico"
    page.bgcolor = Colors.BLUE_GREY_50
    page.window_width = 400
    page.window_height = 600

    model = ChatBotModel()
    controller = ChatController(model)

    chat_view = ft.ListView(expand=True, spacing=10, auto_scroll=True)
    input_box = ft.TextField(hint_text="Escribe tu mensaje...", expand=True, autofocus=True)
    send_btn = ft.IconButton(icon=Icons.SEND, tooltip="Enviar", bgcolor=Colors.BLUE_200, icon_color=Colors.WHITE)

    def update_chat():
        chat_view.controls.clear()
        for sender, text in controller.get_messages():
            if sender == "user":
                chat_view.controls.append(
                    ft.Row([
                        ft.Container(
                            content=ft.Text(text, color=Colors.WHITE),
                            bgcolor=Colors.BLUE_400,
                            border_radius=ft.border_radius.only(20, 20, 0, 20),
                            padding=10,
                            margin=5,
                            alignment=ft.alignment.center_right,
                        )
                    ], alignment=ft.MainAxisAlignment.END)
                )
            else:
                chat_view.controls.append(
                    ft.Row([
                        ft.Container(
                            content=ft.Text(text, color=Colors.BLACK),
                            bgcolor=Colors.GREY_200,
                            border_radius=ft.border_radius.only(20, 20, 20, 0),
                            padding=10,
                            margin=5,
                            alignment=ft.alignment.center_left,
                        )
                    ], alignment=ft.MainAxisAlignment.START)
                )
        page.update()

    def send_message(e=None):
        text = input_box.value.strip()
        if text:
            controller.send_user_message(text)
            input_box.value = ""
            update_chat()

    send_btn.on_click = send_message
    input_box.on_submit = send_message

    page.add(
        ft.Container(
            content=ft.Column([
                ft.Text("Chatbot Básico", size=24, weight=ft.FontWeight.BOLD, color=Colors.BLUE_700),
                ft.Divider(),
                ft.Container(chat_view, expand=True, height=400, bgcolor=Colors.WHITE, border_radius=10, padding=10),
                ft.Row([
                    input_box,
                    send_btn
                ], alignment=ft.MainAxisAlignment.CENTER)
            ], expand=True, spacing=10),
            padding=20,
            expand=True
        )
    )

def main():
    ft.app(target=run)

if __name__ == "__main__":
    main()
