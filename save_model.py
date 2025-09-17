import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib # Modeli kaydetmek ve yüklemek için

# 1. VERİYİ YÜKLEME (TÜM VERİYLE EĞİTECEĞİZ)
df = pd.read_csv('sms_clean.csv')
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']
print("Tüm veri yüklendi.")

# 2. EN İYİ PARAMETRELERLE VECTORIZER VE MODELİ OLUŞTUR
# MLflow'dan bulduğumuz en iyi parametreler
best_params = {"ngram_range": (1, 2), "max_features": 5000, "C": 1.0}

print("En iyi parametrelerle vectorizer ve model eğitiliyor...")
vectorizer = TfidfVectorizer(
    ngram_range=best_params["ngram_range"],
    max_features=best_params["max_features"],
    stop_words='english'
)
# ÖNEMLİ: Artık test için veri ayırmadığımızdan, vectorizer'ı tüm X verisiyle eğitiyoruz.
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(C=best_params["C"], random_state=42)
model.fit(X_vec, y)
print("Model eğitimi tamamlandı.")

# 3. VECTORIZER VE MODELİ DOSYALARA KAYDETME
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(model, 'model.joblib')

print("\nVectorizer -> vectorizer.joblib")
print("Model -> model.joblib")
print("Dosyalar başarıyla kaydedildi.")