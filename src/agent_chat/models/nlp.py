"""
NLP utilities and model training/loading for the chatbot.

This module adapts the original scripts under `chatbot/` into reusable
functions and classes suitable for MVC usage and unit testing.

Notes:
- Heavy deps (TensorFlow / Keras backend) might be unavailable in some envs.
  Training will raise a RuntimeError in that case so the UI can degrade
  gracefully. Loading a pre-trained model also requires a compatible backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import pickle
import random

import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer


# ---------- NLTK helpers ----------

_NLTK_READY = False


def ensure_nltk() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/wordnet")
        nltk.data.find("corpora/omw-1.4")
        _NLTK_READY = True
        return
    except LookupError:
        pass
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    _NLTK_READY = True


lemmatizer = WordNetLemmatizer()


def tokenize_and_lemmatize(text: str) -> List[str]:
    ensure_nltk()
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(tok.lower()) for tok in tokens]


def bag_of_words(sentence: str, words_vocab: List[str]) -> np.ndarray:
    """Convert sentence into a BoW vector aligned to words_vocab ordering."""
    tokens = tokenize_and_lemmatize(sentence)
    bag = np.zeros(len(words_vocab), dtype=np.float32)
    vocab_index = {w: i for i, w in enumerate(words_vocab)}
    for t in tokens:
        i = vocab_index.get(t)
        if i is not None:
            bag[i] = 1.0
    return bag


# ---------- Intents ----------


def load_intents(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_intents(data: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- Artifacts + Inference ----------


@dataclass
class IntentArtifacts:
    model_path: Path
    words_path: Path
    classes_path: Path
    intents_path: Path | None = None


@dataclass
class IntentModel:
    model: Any  # Keras Model, typed as Any to avoid importing heavy symbols at module import
    words: List[str]
    classes: List[str]

    def predict_tag(self, sentence: str) -> str:
        bow = bag_of_words(sentence, self.words)
        # Predict on a batch of size 1
        res = self.model.predict(np.array([bow]), verbose=0)[0]
        max_index = int(np.argmax(res))
        return self.classes[max_index]


def _derive_sidecars(model_path: Path) -> Tuple[Path, Path]:
    base = model_path.with_suffix("")
    words_path = base.with_name(base.name + "_words.pkl")
    classes_path = base.with_name(base.name + "_classes.pkl")
    return words_path, classes_path


def load_artifacts(model_path: str | Path, *, words_path: str | Path | None = None,
                   classes_path: str | Path | None = None) -> IntentModel:
    """Load Keras model + vocabulary + classes.

    If words/classes paths are not provided, we try alongside the model with
    the convention: <modelbase>_words.pkl and <modelbase>_classes.pkl, else
    fall back to `chatbot/words.pkl` and `chatbot/classes.pkl`.
    """
    ensure_nltk()
    model_p = Path(model_path)
    try:
        from keras.models import load_model  # import lazily
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("Keras backend not available to load model") from e

    # Resolve sidecar paths
    w_p, c_p = _derive_sidecars(model_p)
    words_p = Path(words_path) if words_path else (w_p if w_p.exists() else Path("chatbot/words.pkl"))
    classes_p = Path(classes_path) if classes_path else (c_p if c_p.exists() else Path("chatbot/classes.pkl"))

    if not words_p.exists() or not classes_p.exists():
        raise FileNotFoundError("Vocabulary or classes files not found for model")

    with words_p.open("rb") as f:
        words = pickle.load(f)
    with classes_p.open("rb") as f:
        classes = pickle.load(f)

    model = load_model(str(model_p))
    return IntentModel(model=model, words=words, classes=classes)


def respond_from_intents(tag: str, intents_data: Dict[str, Any]) -> str:
    intents = intents_data.get("intents", [])
    for it in intents:
        if it.get("tag") == tag:
            responses = it.get("responses", [])
            if responses:
                return random.choice(responses)
    return "I don't understand."


# ---------- Training ----------


def build_training_data(intents: Dict[str, Any]) -> Tuple[List[str], List[str], List[Tuple[List[str], str]]]:
    """Return (words_vocab, classes, documents) where documents is a list
    of (token_list, tag)."""
    ensure_nltk()
    words: List[str] = []
    classes: List[str] = []
    documents: List[Tuple[List[str], str]] = []
    ignore = {"?", "!", "¿", ".", ","}

    for intent in intents.get("intents", []):
        tag = intent.get("tag")
        for pattern in intent.get("patterns", []):
            tok = nltk.word_tokenize(pattern)
            words.extend(tok)
            documents.append((tok, tag))
        if tag and tag not in classes:
            classes.append(tag)

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]
    words = sorted(set(words))
    return words, classes, documents


def vectorize_training(words: List[str], classes: List[str], documents: List[Tuple[List[str], str]]) -> Tuple[np.ndarray, np.ndarray]:
    output_empty = [0] * len(classes)
    training: List[Tuple[List[int], List[int]]] = []

    for tokens, tag in documents:
        token_lemmas = [lemmatizer.lemmatize(w.lower()) for w in tokens]
        bag = [1 if w in token_lemmas else 0 for w in words]
        row = list(output_empty)
        row[classes.index(tag)] = 1
        training.append((bag, row))

    random.shuffle(training)
    train_x = np.array([t[0] for t in training], dtype=np.float32)
    train_y = np.array([t[1] for t in training], dtype=np.float32)
    return train_x, train_y


def train_and_save(intents: Dict[str, Any], out_dir: str | Path, *,
                   epochs: int = 100, batch_size: int = 5) -> IntentArtifacts:
    """Train a small dense NN and save artifacts next to the model file.

    Returns paths to model (.h5), words.pkl and classes.pkl.
    """
    ensure_nltk()
    try:
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from keras.optimizers import SGD
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("Keras backend not available to train model") from e

    words, classes, documents = build_training_data(intents)
    if not words or not classes:
        raise ValueError("Intents are empty or invalid; cannot train.")
    train_x, train_y = vectorize_training(words, classes, documents)

    model = Sequential(name="chatbot_dense")
    model.add(Dense(128, input_shape=(train_x.shape[1],), activation="relu", name="inp_layer"))
    model.add(Dropout(0.5, name="drop1"))
    model.add(Dense(64, activation="relu", name="hidden"))
    model.add(Dropout(0.5, name="drop2"))
    model.add(Dense(train_y.shape[1], activation="softmax", name="out"))

    sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"model_{ts}"
    model_path = base.with_suffix(".h5")
    words_path = base.with_name(base.name + "_words.pkl")
    classes_path = base.with_name(base.name + "_classes.pkl")

    # Save artifacts
    model.save(str(model_path))
    with words_path.open("wb") as f:
        pickle.dump(words, f)
    with classes_path.open("wb") as f:
        pickle.dump(classes, f)

    return IntentArtifacts(model_path=model_path, words_path=words_path, classes_path=classes_path, intents_path=None)


def write_vocab_sidecars_from_intents(intents: Dict[str, Any], model_path: str | Path) -> Tuple[Path, Path]:
    """Utility for environments without Keras: persist words/classes for predict-time use
    even if the model is a mock file. Returns (words_path, classes_path).
    """
    words, classes, documents = build_training_data(intents)
    model_p = Path(model_path)
    words_p, classes_p = _derive_sidecars(model_p)
    with words_p.open("wb") as f:
        pickle.dump(words, f)
    with classes_p.open("wb") as f:
        pickle.dump(classes, f)
    return words_p, classes_p
