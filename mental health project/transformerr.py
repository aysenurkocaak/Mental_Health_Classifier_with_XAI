import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
import joblib
import string
import re
import time

print("🚀 Kod başlatıldı...")

# Dosya yolları
input_path = "Combined Data.csv"
embedding_npy_path = "mpnet_embeddings.npy"
label_npy_path = "mpnet_labels.npy"
embedding_csv_path = "mpnet_embeddings.csv"
encoder_save_path = "label_encoder.pkl"

#Veriyi oku
df = pd.read_csv(input_path)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 🔍 Boş değerleri temizle
df.dropna(subset=['statement', 'status'], inplace=True)

# Temizlik fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Temizlenmiş metin sütunu oluştur
df['clean_text'] = df['statement'].astype(str).apply(clean_text)

# Embedleme
print(" Model yükleniyor...")
model = SentenceTransformer("all-mpnet-base-v2")

texts = df["clean_text"].tolist()
batch_size = 64
chunks = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
X_embed_chunks = []

total_chunks = len(chunks)
print(f"Toplam {total_chunks} adet chunk işlenecek...\n")

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}/{total_chunks} işleniyor...")
    start_time = time.time()

    embeds = model.encode(chunk, batch_size=batch_size, show_progress_bar=False)
    X_embed_chunks.append(embeds)

    elapsed = time.time() - start_time
    print(f"Chunk {i} tamamlandı ({elapsed:.2f} saniye)\n")

# 🔗 Tüm embedleri birleştir
X_embed = np.vstack(X_embed_chunks)
y = df["status"].values

# 🧠 Label encode işlemi
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 💾 Encode dosyasını kaydet
joblib.dump(le, encoder_save_path)

# 💾 Embedleri kaydet
np.save(embedding_npy_path, X_embed)
np.save(label_npy_path, y_encoded)

# 💾 CSV olarak da kaydet
X_embed_df = pd.DataFrame(X_embed)
X_embed_df["label"] = y_encoded
X_embed_df.to_csv(embedding_csv_path, index=False)

# ✅ Bitirme mesajı
print("\n Embedleme tamamlandı.")
print(f" Kaydedilen dosyalar:")
print(f" - Embedding .npy: {embedding_npy_path}")
print(f" - Label .npy:     {label_npy_path}")
print(f" - CSV dosyası:    {embedding_csv_path}")
print(f" - Label encoder:  {encoder_save_path}")