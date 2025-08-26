import flet as ft
from agent_chat.views import run
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def main():
    ft.app(target=run)

if __name__ == "__main__":
    main()
