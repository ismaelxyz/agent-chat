import os
from pathlib import Path
import json
import pytest

from agent_chat.models import nlp


@pytest.fixture()
def tiny_intents():
    return {
        "intents": [
            {"tag": "greet", "patterns": ["hello", "hi"], "responses": ["hey"]},
            {"tag": "bye", "patterns": ["bye"], "responses": ["bye"]},
        ]
    }


def test_write_vocab_sidecars_from_intents(tmp_path: Path, tiny_intents):
    # Create a dummy model path, ensure sidecars are written
    model_path = tmp_path / "model_dummy.keras"
    model_path.write_text("not a real model")

    words_p, classes_p = nlp.write_vocab_sidecars_from_intents(tiny_intents, model_path)
    assert words_p.exists() and classes_p.exists()


def test_train_and_save_smoke(tmp_path: Path, tiny_intents, monkeypatch):
    # If Keras/Tensorflow is not available in CI, skip
    try:
        import keras  # noqa: F401
    except Exception:
        pytest.skip("Keras backend not available in this environment")

    # Speed up training dramatically for test
    artifacts = nlp.train_and_save(tiny_intents, tmp_path, epochs=1, batch_size=8)
    assert artifacts.model_path.exists()
    assert artifacts.words_path.exists()
    assert artifacts.classes_path.exists()
