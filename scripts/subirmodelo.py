import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Cargar credenciales desde .env
load_dotenv()

HF_TOKEN  = os.getenv("HF_TOKEN")
HF_USER   = os.getenv("HF_USER")
REPO_NAME = "fake-news-detector"
REPO_ID   = f"{HF_USER}/{REPO_NAME}"

# Rutas
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Crear repositorio
api = HfApi()
api.create_repo(repo_id=REPO_ID, token=HF_TOKEN, exist_ok=True)
print(f"✅ Repositorio creado: https://huggingface.co/{REPO_ID}")

# Modelo LR y sus dependencias
archivos = [
    'lr_pipeline.pkl',      # modelo Logistic Regression
    'tfidf_title.pkl',      # vectorizador título
    'tfidf_content.pkl',    # vectorizador contenido
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

print(f"\n🎉 Modelo disponible en: https://huggingface.co/{REPO_ID}")