from __future__ import annotations

from pathlib import Path
from typing import Optional

from .nlp import (
    load_intents,
    load_artifacts,
    respond_from_intents,
)


class ChatBotModel:
    """Chat bot model wrapper that can use a trained Keras model if available.

    Fallback: if model cannot be loaded, use a naive keyword mapping and
    return a generic message.
    """

    def __init__(self):
        # Active artifacts
        self.model_path: Optional[str] = None
        self._intent_model = None  # loaded keras + vocab
        self._intents_data = None  # loaded intents.json to pick responses
        # Custom meta for display/testing
        self.custom_version: int = 0
        self.custom_label: str = ""
        # Whether to use generated (custom) model vs local/manual
        self.use_generated: bool = False

    # Initial state is provided by UI via client_storage restore
    # (no file-based persistence here)

    # ---- Configuration API ----
    def set_model_path(self, path: Optional[str]):
        """Set and attempt to load model + sidecars. If fails, keep fallback."""
        self.model_path = path
        self._intent_model = None
        if not path:
            return
        try:
            # Try to load intents file colocated in project
            intents_path = Path("storage/intents.json")
            if intents_path.exists():
                self._intents_data = load_intents(intents_path)
            self._intent_model = load_artifacts(path)
        except Exception:
            # Keep fallback
            self._intent_model = None
        finally:
            pass

    def set_custom_meta(self, *, version: int | None = None, label: str | None = None):
        if version is not None:
            self.custom_version = version
        if label is not None:
            self.custom_label = label
    # UI layer persists via client_storage

    def bump_version(self, label: str | None = None):
        self.custom_version += 1
        if label is not None:
            self.custom_label = label
    # UI layer persists via client_storage

    def set_use_generated(self, value: bool):
        self.use_generated = bool(value)
        # UI layer persists via client_storage
    # (Persistence is handled by ConfigView via Flet client_storage)

    # ---- Inference ----
    def has_active_model(self) -> bool:
        """Return True if a model and its intents are loaded and usable."""
        return self._intent_model is not None and self._intents_data is not None

    def get_response(self, message: str) -> str:
        text = (message or "").strip()
        if not text:
            return "Please write a message."

        # Try neural model
        if self._intent_model is not None and self._intents_data is not None:
            try:
                tag = self._intent_model.predict_tag(text)
                return respond_from_intents(tag, self._intents_data)
            except Exception as e:
                # If model inference fails, drop to fallback
                pass
        # Lightweight keyword-based fallback
        low = text.lower()
        if any(k in low for k in ("hello", "hi", "hey")):
            return "Hi there!" + self._suffix()
        if any(k in low for k in ("bye", "goodbye", "see you")):
            return "Bye!" + self._suffix()

        return "I don't understand, can you rephrase?" + self._suffix()

    def _suffix(self) -> str:
        return (
            f"\n(Model: {self.model_path} - v{self.custom_version} {self.custom_label})"
            if self.model_path
            else ""
        )
