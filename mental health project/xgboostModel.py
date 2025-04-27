import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
#modellll
df = pd.read_csv("mpnet_randomundersampled.csv")
X = df.drop("label", axis=1)
y = df["label"]
#  Label encoderrrrr burada
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

xgb_model = XGBClassifier(
    max_depth=5,
    learning_rate=0.02,
    n_estimators=500,
    reg_alpha=1,
    reg_lambda=4,
    subsample=0.85,
    colsample_bytree=0.9,
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42
)

# SMOTE + CV akışı
print("\n🔎 Cross-validation (5-fold, macro F1):")
pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy={2: 1500}, random_state=42)),
    ('xgb', xgb_model)
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y_encoded, scoring="f1_macro", cv=cv, n_jobs=-1)
print(f"Ortalama F1 Skoru: {np.mean(cv_scores):.4f}")
print(f" Tüm Fold'lar: {cv_scores}")
print(f" Std Sapma: {np.std(cv_scores):.4f}")

# SMOTE (sadece sınıf 2)
sm = SMOTE(sampling_strategy={2: 1500}, random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
# Modeli yeniden eğit (dengeli veriyle)
xgb_model.fit(X_res, y_res)

# Tahmin yap
y_train_pred = xgb_model.predict(X_res)
y_test_pred = xgb_model.predict(X_test)

# Skorları yazdır
def print_scores(y_true, y_pred, label="SET", label_encoder=None):
    print(f"\n {label} SONUÇLARI")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='macro'))
    print("Recall   :", recall_score(y_true, y_pred, average='macro'))
    print("F1 Score :", f1_score(y_true, y_pred, average='macro'))
    if label_encoder is not None:
        target_names = label_encoder.classes_.astype(str)
        print("Classification Report:\n", classification_report(y_true, y_pred, target_names=target_names))
    else:
        print("Classification Report:\n", classification_report(y_true, y_pred))

#  Eğitim ve test skorlarını gör
print_scores(y_res, y_train_pred, "EĞİTİM", le)
print_scores(y_test, y_test_pred, "TEST", le)
from pathlib import Path
import joblib

# Model ve Encoder'ı Kaydet
try:
    model_path = Path("xgboost_pc.pkl")
    encoder_path = Path("label_encoder.pkl")

    joblib.dump(xgb_model, model_path)
    joblib.dump(le, encoder_path)

    if model_path.exists() and encoder_path.exists():
        print(" Model ve encoder başarıyla kaydedildi.")
    else:
        print(" Model dosyaları bulunamadı.")
except Exception as e:
    print(" Model/Encoder kaydında hata:", e)

print("\n Tüm işlem başarıyla tamamlandı.")

print("Sınıf Sırası:", le.classes_)