# Modelo para el chatbot (maquetado / mock)
class ChatBotModel:
    def __init__(self):
        # Palabras clave y respuestas
        self.responses = {
            "hola": "¡Hola! ¿En qué puedo ayudarte?",
            "adios": "¡Hasta luego!",
            "gracias": "¡De nada!",
            "ayuda": "¿Sobre qué tema necesitas ayuda?",
            "nombre": "Soy un chatbot básico hecho con Flet."
        }
        # Ruta del modelo seleccionado (mock: solo informativo por ahora)
        self.model_path: str | None = None
        # Identificador de modelo custom
        self.custom_version: int = 0
        self.custom_label: str = ""

    def set_model_path(self, path: str | None):
        """Define la ruta del modelo a usar (solo informativo por ahora)."""
        self.model_path = path

    def set_custom_meta(self, *, version: int | None = None, label: str | None = None):
        """Actualiza metadatos del modelo custom (versión/label)."""
        if version is not None:
            self.custom_version = version
        if label is not None:
            self.custom_label = label

    def bump_version(self, label: str | None = None):
        """Incrementa la versión del modelo custom y opcionalmente actualiza el label."""
        self.custom_version += 1
        if label is not None:
            self.custom_label = label

    def get_response(self, message: str) -> str:
        message = message.lower()
        for keyword, response in self.responses.items():
            if keyword in message:
                return response
        # Añade un guiño al modelo seleccionado si existe (para pruebas visuales)
        suffix = f"\n(Modelo: {self.model_path} - v{self.custom_version} {self.custom_label})" if self.model_path else ""
        return "No entiendo, ¿puedes reformular tu pregunta?" + suffix
