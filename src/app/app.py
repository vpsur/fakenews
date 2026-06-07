import gradio as gr
import joblib
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
from huggingface_hub import hf_hub_download
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)
stop_words_en = set(stopwords.words('english'))

# ── Cargar modelo desde HuggingFace Hub ───────────────────────────────────────
REPO_ID = "vrammun/fake-news-detector"

print("Cargando modelo...")
modelo        = joblib.load(hf_hub_download(REPO_ID, "lr_pipeline.pkl"))
tfidf_title   = joblib.load(hf_hub_download(REPO_ID, "tfidf_title.pkl"))
tfidf_content = joblib.load(hf_hub_download(REPO_ID, "tfidf_content.pkl"))
print("✅ Modelo cargado")

# ── Limpieza ──────────────────────────────────────────────────────────────────
def limpiar_texto(texto):
    if not texto:
        return ""
    texto = texto.lower()
    texto = re.sub(r'<.*?>', '', texto)
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    palabras = texto.split()
    return " ".join([p for p in palabras if p not in stop_words_en])

# ── Predicción ────────────────────────────────────────────────────────────────
def predecir(titulo, contenido):
    if not titulo.strip() and not contenido.strip():
        return (
            "<div style='text-align:center; padding:20px; color:#888;'>"
            "⚠️ Introduce el título o contenido de la noticia."
            "</div>"
        )

    titulo_clean    = limpiar_texto(titulo)
    contenido_clean = limpiar_texto(contenido)

    title_vec    = tfidf_title.transform([titulo_clean])
    content_vec  = tfidf_content.transform([contenido_clean])
    num_features = csr_matrix(np.zeros((1, 6)))
    X = hstack([title_vec, content_vec, num_features])

    proba     = modelo.predict_proba(X)[0]
    pred      = modelo.predict(X)[0]
    confianza = max(proba) * 100
    prob_fake = proba[0] * 100
    prob_real = proba[1] * 100

    if pred == 1:
        color   = "#22c55e"
        emoji   = "✅"
        label   = "REAL"
        mensaje = "This news appears to be <b>legitimate</b>."
    else:
        color   = "#ef4444"
        emoji   = "❌"
        label   = "FAKE"
        mensaje = "This news appears to be <b>fake or misleading</b>."

    html = f"""
    <div style="font-family: 'Inter', sans-serif; padding: 24px; border-radius: 16px;
                background: #0f172a; color: white; max-width: 600px; margin: auto;">

        <div style="text-align: center; margin-bottom: 24px;">
            <span style="font-size: 64px;">{emoji}</span>
            <h1 style="font-size: 36px; font-weight: 800; color: {color}; margin: 8px 0;">
                {label}
            </h1>
            <p style="color: #94a3b8; font-size: 15px;">{mensaje}</p>
        </div>

        <div style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="color: #94a3b8; font-size: 13px;">🟢 Real</span>
                <span style="color: white; font-size: 13px; font-weight: 600;">{prob_real:.1f}%</span>
            </div>
            <div style="background: #1e293b; border-radius: 999px; height: 10px;">
                <div style="background: #22c55e; width: {prob_real:.1f}%; height: 10px;
                            border-radius: 999px;"></div>
            </div>
        </div>

        <div style="margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="color: #94a3b8; font-size: 13px;">🔴 Fake</span>
                <span style="color: white; font-size: 13px; font-weight: 600;">{prob_fake:.1f}%</span>
            </div>
            <div style="background: #1e293b; border-radius: 999px; height: 10px;">
                <div style="background: #ef4444; width: {prob_fake:.1f}%; height: 10px;
                            border-radius: 999px;"></div>
            </div>
        </div>

        <div style="text-align: center; padding: 12px; background: #1e293b;
                    border-radius: 12px; color: #64748b; font-size: 12px;">
            Modelo: Logistic Regression · TF-IDF · 47.000+ noticias
        </div>
    </div>
    """
    return html

# ── Interfaz ──────────────────────────────────────────────────────────────────
css = """
body { background-color: #0f172a !important; }
.gradio-container { max-width: 900px !important; margin: auto !important; }
textarea { background: #1e293b !important; color: white !important; border: 1px solid #334155 !important; border-radius: 12px !important; }
label { color: #94a3b8 !important; font-weight: 500 !important; }
button.primary { background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
                 border: none !important; border-radius: 12px !important;
                 font-weight: 700 !important; font-size: 16px !important;
                 padding: 12px !important; }
"""

with gr.Blocks(css=css, title="Fake News Detector") as demo:

    gr.HTML("""
    <div style="text-align: center; padding: 32px 0 16px; font-family: Inter, sans-serif;">
        <h1 style="font-size: 42px; font-weight: 800; color: white; margin: 0;">
            🔍 Fake News Detector
        </h1>
        <p style="color: #64748b; font-size: 16px; margin-top: 8px;">
            Paste a news article below to detect if it's real or fake
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            titulo    = gr.Textbox(lines=2,  label="📰 Headline", placeholder="Enter the news headline...")
            contenido = gr.Textbox(lines=10, label="📄 Content",  placeholder="Enter the news content...")
            btn       = gr.Button("🔍 Analyze", variant="primary")

            gr.Examples(
                examples=[
                    ["✅ Senate passes bipartisan infrastructure bill with 69 votes",
                     "The United States Senate passed a 1.2 trillion dollar bipartisan infrastructure bill on Tuesday with 69 votes in favor and 30 against. The legislation includes funding for roads bridges broadband internet and public transit systems."],
                    ["❌ Hillary Clinton arrested by FBI agents at her home in New York",
                     "Federal agents stormed the home of former secretary of state Hillary Clinton early this morning and placed her under arrest for crimes against the american people. Share this before they delete it forever patriots."],
                ],
                inputs=[titulo, contenido]
            )

        with gr.Column(scale=1):
            resultado = gr.HTML(label="Result")

    btn.click(fn=predecir, inputs=[titulo, contenido], outputs=resultado)

demo.launch()