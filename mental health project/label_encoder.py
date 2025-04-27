import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# 📥 Veri dosyasını oku
df_path = "Combined Data.csv"
df = pd.read_csv(df_path)

# 🔍 Sütun adlarını kontrol et
print("📋 Mevcut sütunlar:", df.columns.tolist())

# 🧠 Tahmini etiket sütunu: string olan ve 2-50 unique değeri olan sütunlardan biri
candidate_columns = [
    col for col in df.columns
    if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 50
]

# ✅ İlk eşleşeni etiket olarak al
if candidate_columns:
    label_col = candidate_columns[0]
    print(f"✅ Etiket sütunu bulundu: '{label_col}'")
else:
    raise ValueError("❌ Etiket sütunu bulunamadı. Lütfen dosyanın içeriğini kontrol et.")

# 🎯 LabelEncoder uygula
le = LabelEncoder()
le.fit(df[label_col])

# 💾 Kaydet
encoder_path = "label_encoder.pkl"
joblib.dump(le, encoder_path)

print("✅ LabelEncoder başarıyla kaydedildi:", encoder_path)
print("🔢 Sınıf eşleşmeleri:")
print(dict(enumerate(le.classes_)))