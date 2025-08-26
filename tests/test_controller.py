from agent_chat.controllers.chat_controller import ChatController
from agent_chat.models import ChatBotModel


def test_controller_message_flow():
    model = ChatBotModel()
    ctl = ChatController(model)

    reply = ctl.send_user_message("hello")
    assert "Hi" in reply

    msgs = ctl.get_messages()
    assert msgs[0] == ("user", "hello")
    assert msgs[1][0] == "bot"
    assert "Hi" in msgs[1][1]


def test_controller_select_model_calls_model(monkeypatch):
    model = ChatBotModel()
    called = {"times": 0, "arg": None}

    def fake_set(path):
        called["times"] += 1
        called["arg"] = path

    monkeypatch.setattr(model, "set_model_path", fake_set)
    ctl = ChatController(model)
    ctl.select_model("/tmp/model.keras")

    assert called["times"] == 1
    assert called["arg"] == "/tmp/model.keras"
