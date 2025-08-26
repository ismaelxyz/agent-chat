import types
import pytest

from agent_chat.models import ChatBotModel, nlp


@pytest.fixture()
def bot():
    return ChatBotModel()


def test_bot_fallback_basic(bot):
    assert "Hi" in bot.get_response("hello there")
    assert "Bye" in bot.get_response("ok bye")
    assert "I don't understand" in bot.get_response("zzzz")


def test_bot_empty_input(bot):
    assert bot.get_response("") == "Please write a message."


def test_bot_with_loaded_intents_prediction(monkeypatch, bot):
    # Arrange: load intents data and a fake model that predicts index 0 -> 'greet'
    bot._intents_data = {
        "intents": [
            {"tag": "greet", "responses": ["hola", "hello there"]},
            {"tag": "bye", "responses": ["adios"]},
        ]
    }

    class FakeModel:
        def predict(self, X, verbose=0):
            # Always predict class 0 with high prob
            return [[0.99, 0.01]]

    # Provide vocab/classes aligned with prediction index
    intent_model = nlp.IntentModel(model=FakeModel(), words=["hello"], classes=["greet", "bye"])
    bot._intent_model = intent_model

    # Act
    out = bot.get_response("anything")

    # Assert: must pick one of greet responses
    assert out in {"hola", "hello there"}


def test_set_model_path_handles_errors(monkeypatch, bot):
    # Simulate load_artifacts raising so fallback remains active
    monkeypatch.setattr(nlp, "load_artifacts", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no backend")))
    bot.set_model_path("/path/to/nonexistent.keras")
    assert bot._intent_model is None

