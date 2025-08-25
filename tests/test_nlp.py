import json
from pathlib import Path

from agent_chat.models import ChatBotModel
from agent_chat.models import nlp


def test_tokenize_and_bow_basic(tmp_path: Path):
    vocab = ["hello", "world"]
    bow = nlp.bag_of_words("Hello world!", vocab)
    assert bow.tolist() == [1.0, 1.0]


def test_build_training_data_and_vectorize():
    intents = {
        "intents": [
            {"tag": "greet", "patterns": ["hello", "hi"], "responses": ["hey"]},
            {"tag": "bye", "patterns": ["bye"], "responses": ["bye"]},
        ]
    }
    words, classes, docs = nlp.build_training_data(intents)
    assert "hello" in words and "hi" in words
    assert set(classes) == {"greet", "bye"}
    X, y = nlp.vectorize_training(words, classes, docs)
    assert X.shape[0] == len(docs)
    assert y.shape[0] == len(docs)


def test_chatbot_fallback_response():
    bot = ChatBotModel()
    # No model set -> fallback keyword matching
    assert "Hi" in bot.get_response("hello there")
    # Unknown -> generic
    msg = bot.get_response("zzzz")
    assert "I don't understand" in msg
