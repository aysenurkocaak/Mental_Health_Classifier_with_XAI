from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import numpy as np
import pandas as pd

# 🚫 Tokenizer uyarılarını kapat (transformers için)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 📁 Dosya yolları
X_embed = np.load("mpnet_embeddings.npy")
y = np.load("mpnet_labels.npy")

# 🔢 PCA uygulama (400 bileşenli ana analiz için)
pca = PCA(n_components=400, random_state=42)
X_pca = pca.fit_transform(X_embed)

# ✅ Bilgi oranı
print(f"✅ PCA ile korunabilen bilgi: {pca.explained_variance_ratio_.sum():.2%}")

# 📄 PCA sonrası dataframe ve kaydetme
df_pca = pd.DataFrame(X_pca)
df_pca["label"] = y

csv_out_path = "mental_health_embeddings_pca200.csv"
df_pca.to_csv(csv_out_path, index=False)

# 💾 PCA objesini kaydet
joblib.dump(pca, "pca.pkl")

# ✅ Label ve PCA boyutu eşleşiyor mu?
assert X_pca.shape[0] == y.shape[0], "❌ Satır sayısı uyuşmuyor!"
print("✅ PCA ile y (label) sayısı eşleşiyor.")

# ===============================
# 🎨 2D PCA ile Görselleştirme
# ===============================

# 🔢 2 bileşenlik PCA
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_embed)

df_2d = pd.DataFrame(X_2d, columns=["PCA1", "PCA2"])
df_2d["label"] = y

# 🎨 LabelEncoder varsa etiket isimlerini göster
try:
    encoder_path = "label_encoder.pkl"
    le = joblib.load(encoder_path)
    df_2d["label_name"] = le.inverse_transform(df_2d["label"])
except:
    df_2d["label_name"] = df_2d["label"]  # yedek çözüm

# 🌈 Küme yapısı scatterplot
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df_2d,
    x="PCA1", y="PCA2",
    hue="label_name",
    palette="Set2",
    s=50,
    alpha=0.7
)
plt.title("🧠 2D PCA ile Ruh Sağlığı Sınıflarının Küme Yapısı")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Sınıf", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
