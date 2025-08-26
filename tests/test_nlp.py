from pathlib import Path
import re
import pytest

from agent_chat.models import nlp


@pytest.fixture()
def sample_intents():
    return {
        "intents": [
            {"tag": "greet", "patterns": ["hello", "hi"], "responses": ["hey", "hello there"]},
            {"tag": "bye", "patterns": ["bye"], "responses": ["bye"]},
        ]
    }


def test_bag_of_words_monkeypatched(monkeypatch):
    # Avoid NLTK resources in CI: patch tokenization
    monkeypatch.setattr(nlp, "tokenize_and_lemmatize", lambda s: re.findall(r"\w+", s.lower()))
    vocab = ["hello", "world"]
    bow = nlp.bag_of_words("Hello world!", vocab)
    assert bow.tolist() == [1.0, 1.0]


def test_build_training_data_and_vectorize(sample_intents, monkeypatch):
    # Patch nltk.word_tokenize to simple split to avoid corpus downloads
    monkeypatch.setattr(nlp.nltk, "word_tokenize", lambda s: s.split())
    words, classes, docs = nlp.build_training_data(sample_intents)
    assert "hello" in words and "hi" in words
    assert set(classes) == {"greet", "bye"}
    X, y = nlp.vectorize_training(words, classes, docs)
    assert X.shape[0] == len(docs)
    assert y.shape[0] == len(docs)


def test_respond_from_intents(sample_intents):
    msg = nlp.respond_from_intents("greet", sample_intents)
    assert msg in {"hey", "hello there"}
    assert nlp.respond_from_intents("unknown", sample_intents).startswith("I don't")
