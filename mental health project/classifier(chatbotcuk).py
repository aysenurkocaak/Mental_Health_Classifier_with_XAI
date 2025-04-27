import gradio as gr
import torch
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


MODEL_PATH = Path("xgboost_pc.pkl")
ENCODER_PATH = Path("label_encoder.pkl")
PCA_PATH = Path("pca.pkl")

xgb_model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)
pca = joblib.load(PCA_PATH)
label_names = list(le.classes_)

#mpnet
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model.eval()
model = model.to("cpu")

#embedding
def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tokens = {k: v.to("cpu") for k, v in tokens.items()}  # 🎯 CPU uyumu
    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1)
    return embedding.numpy()


def classify(user_input):
    if user_input.strip() == "":
        return "⚠️ Lütfen bir şey yazın.", pd.DataFrame(columns=["Sınıf", "Olasılık (%)"])

    emb = get_embedding(user_input)
    emb_pca = pca.transform(emb)
    probs = xgb_model.predict_proba(emb_pca)
    pred_index = int(np.argmax(probs))
    pred_label = label_names[pred_index]

    # 📊 Sınıf isimleri ve olasılıkları
    probs_df = pd.DataFrame({
        "Sınıf": label_names,
        "Olasılık (%)": (probs[0] * 100).round(2)
    })

    return f"🔮 Modelin Tahmini: {pred_label}", probs_df


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, placeholder="Bir cümle yazın...", label="Cümle Girişi"),
    outputs=[
        gr.Textbox(label="🔮 Modelin Tahmini✨"),
        gr.Dataframe(label="✨Sınıf Olasılıkları✨", type="pandas")
    ],
    title="🧠❤️✨ MPNet + XGBoost Mental Health Classifier 🔮⚕️✨",
    description=(
        "✨✨✨ MPNet + PCA + XGBoost modeliyle çalışan bir ruh sağlığı sınıflandırma botu ✨✨✨\n\n"
        "Tahmin Edilebilecek Sınıflar:\n"
        "- Anxiety\n- Bipolar\n- Depression\n- Normal\n- Personality disorder\n- Stress\n- Suicidal"
    )
)

demo.launch()
