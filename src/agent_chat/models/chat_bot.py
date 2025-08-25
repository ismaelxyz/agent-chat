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

    def set_model_path(self, path: str | None):
        """Define la ruta del modelo a usar (solo informativo por ahora)."""
        self.model_path = path

    def get_response(self, message: str) -> str:
        message = message.lower()
        for keyword, response in self.responses.items():
            if keyword in message:
                return response
        # Añade un guiño al modelo seleccionado si existe (para pruebas visuales)
        suffix = f"\n(Modelo: {self.model_path})" if self.model_path else ""
        return "No entiendo, ¿puedes reformular tu pregunta?" + suffix
