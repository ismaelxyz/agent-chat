from agent_chat.models import ChatBotModel

class ChatController:
    def __init__(self, model: ChatBotModel):
        self.model = model
        self.messages = []  # (sender, text)

    def send_user_message(self, text: str):
        self.messages.append(("user", text))
        response = self.model.get_response(text)
        self.messages.append(("bot", response))
        return response

    def get_messages(self):
        return self.messages

    # Optional: control model selection via controller
    def select_model(self, path: str | None):
        self.model.set_model_path(path)
