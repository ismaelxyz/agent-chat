Agent Chat
==========

This app integrates a simple intent-based chatbot into an MVC-ish Flet UI.

Key parts:
- src/agent_chat/models/nlp.py: training/loading utilities adapted from the original scripts in `chatbot/`.
- src/agent_chat/models/chat_bot.py: runtime ChatBotModel with neural model support and a safe fallback.
- src/agent_chat/views/: ChatView and ConfigView to chat and configure/train.

Training
- In the app, go to Configuration, edit intents.json, Confirm, then Train. If Keras/TensorFlow is not available, a mock .h5 is created plus vocabulary sidecars to keep inference stable.

Tests
- Basic tests cover preprocessing and fallback logic in `tests/test_nlp.py`.
