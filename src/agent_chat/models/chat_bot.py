# Modelo para el chatbot
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

    def get_response(self, message: str) -> str:
        message = message.lower()
        for keyword, response in self.responses.items():
            if keyword in message:
                return response
        return "No entiendo, ¿puedes reformular tu pregunta?"
