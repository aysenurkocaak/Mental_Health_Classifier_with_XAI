import torch
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModel
import joblib
import os

# ==========================
# 📁 MODEL ve DOSYA YOLLARI
# ==========================
MODEL_PATH = "xgboost_pc.pkl"
ENCODER_PATH = "label_encoder.pkl"
PCA_PATH = "pca.pkl"
TEST_TEXT_PATH = "test_texts_with_labels.csv"
OUTPUT_FOLDER = "lime_outputs"
EVAL_CSV_PATH = "lime_prediction_evaluation.csv"

# ==========================
# 🎯 Model, PCA, Encoder yükle
# ==========================
xgb_model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)
pca = joblib.load(PCA_PATH)

# ==========================
# 🤗 Hugging Face MPNet modeli
# ==========================
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model.eval()
model = model.to("cpu")

# ==========================
# 🧠 Model Pipeline: Text → Embedding → PCA → Predict Proba
# ==========================
def predict_proba(texts):
    embeddings = []
    for t in texts:
        tokens = tokenizer(t, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = model(**tokens)
        embed = output.last_hidden_state.mean(dim=1).numpy()
        embed_pca = pca.transform(embed)
        embeddings.append(embed_pca[0])
    return xgb_model.predict_proba(np.array(embeddings))

# ==========================
# 🌈 Kullanıcı cümlesi ile LIME HTML çıktısı döner
# ==========================
def get_lime_html(sentence):
    explainer = LimeTextExplainer(class_names=le.classes_)
    prediction = predict_proba([sentence])
    predicted_index = int(np.argmax(prediction))

    exp = explainer.explain_instance(
        sentence,
        predict_proba,
        num_features=10,
        labels=[predicted_index],
        num_samples=500
    )
    return exp.as_html()

# ==========================
# 📊 Index'e göre cümle al ve LIME açıkla
# ==========================
def get_lime_html_by_index(index, test_text_path=TEST_TEXT_PATH):
    df = pd.read_csv(test_text_path)
    if "statement" not in df.columns:
        raise ValueError("❌ 'statement' sütunu test_texts.csv dosyasında bulunamadı.")
    if index >= len(df):
        raise IndexError(f"❌ İstenilen index ({index}) test_texts.csv dosyasında yok.")
    sentence = df.iloc[index]["statement"]
    return get_lime_html(sentence)

# ==========================
# 💾 LIME AÇIKLAMALARINI KAYDET (SADECE İLK 15 ÖRNEK)
# ==========================
def save_all_lime_explanations(output_dir=OUTPUT_FOLDER):
    os.makedirs(output_dir, exist_ok=True)
    df_texts = pd.read_csv(TEST_TEXT_PATH)
    label_col = "status"  # ✅ Etiket sütunu adı

    results = []

    for i in range(min(15, len(df_texts))):  # ✅ SADECE İLK 15 ÖRNEK
        try:
            # 🔹 LIME Açıklamasını HTML olarak kaydet
            html = get_lime_html_by_index(i)
            path = os.path.join(output_dir, f"lime_index_{i}.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"✅ Kaydedildi: {path}")

            # 🔹 Doğruluk kontrolü için tahmin al
            sentence = df_texts.iloc[i]["statement"]
            true_label = df_texts.iloc[i][label_col]
            embed = predict_proba([sentence])
            pred_index = int(np.argmax(embed))
            pred_label = le.inverse_transform([pred_index])[0]

            results.append({
                "index": i,
                "lime_file": f"lime_index_{i}.html",
                "text": sentence,
                "true_label": true_label,
                "predicted_label": pred_label,
                "correct": true_label == pred_label
            })

        except Exception as e:
            print(f"❌ HATA (index {i}): {e}")

    # 🔹 CSV'yi kaydet
    pd.DataFrame(results).to_csv(EVAL_CSV_PATH, index=False)
    print(f"📁 Tahmin doğrulukları CSV olarak kaydedildi: {EVAL_CSV_PATH}")

# 🎯 Ana çalıştırma
if __name__ == "__main__":
    print("🚀 Sadece ilk 15 örnek için LIME açıklama üretimi başlıyor...")
    save_all_lime_explanations()
    print("✅ İlk 15 açıklama ve değerlendirme başarıyla kaydedildi.")
