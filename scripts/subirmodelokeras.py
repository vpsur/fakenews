import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

HF_TOKEN  = os.getenv("HF_TOKEN")
HF_USER   = os.getenv("HF_USER")
REPO_ID   = f"{HF_USER}/fake-news-detector"

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

api = HfApi()

archivos = [
    'keras_model.keras',
]

for nombre in archivos:
    path = os.path.join(MODELS_DIR, nombre)
    if os.path.exists(path):
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=nombre,
            repo_id=REPO_ID,
            token=HF_TOKEN
        )
        print(f"   ✅ Subido: {nombre}")
    else:
        print(f"   ⚠️  No encontrado: {nombre}")

print(f"\n🎉 Disponible en: https://huggingface.co/{REPO_ID}")