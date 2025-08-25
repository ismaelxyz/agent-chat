import flet as ft
from flet import Colors, Icons
from agent_chat.controllers import ChatController


class ChatView(ft.Container):
    def __init__(self, controller: ChatController):
        self.controller = controller
        self.chat_list = ft.ListView(expand=True, spacing=10, auto_scroll=True)
        self.input_box = ft.TextField(hint_text="Type your message...", expand=True, autofocus=True)
        self.send_btn = ft.IconButton(icon=Icons.SEND, tooltip="Send", bgcolor=Colors.BLUE_200, icon_color=Colors.WHITE)

        # Event wiring
        self.send_btn.on_click = self.send_message
        self.input_box.on_submit = self.send_message

        content = ft.Column([
            ft.Text("Chat", size=22, weight=ft.FontWeight.BOLD, color=Colors.BLUE_700),
            ft.Divider(),
            ft.Container(self.chat_list, expand=True, height=420, bgcolor=Colors.WHITE, border_radius=10, padding=10),
            ft.Row([self.input_box, self.send_btn], alignment=ft.MainAxisAlignment.CENTER),
        ], expand=True, spacing=10)

        super().__init__(content=content, padding=20, expand=True)

    def send_message(self, e=None):
        text = self.input_box.value.strip()
        if not text:
            return
        self.controller.send_user_message(text)
        self.input_box.value = ""
        self.update_chat()

    def update_chat(self):
        self.chat_list.controls.clear()
        for sender, text in self.controller.get_messages():
            if sender == "user":
                self.chat_list.controls.append(
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
                self.chat_list.controls.append(
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
        self.page.update()
