import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow # MLflow kütüphanesini import et

df = pd.read_csv('sms_clean.csv')

df['label'] = df['label'].map({'ham':0,'spam':1})
X = df['message']
y = df['label']
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

experiment_params = [
    {"ngram_range": (1, 1), "max_features": 5000, "C": 1.0},
    {"ngram_range": (1, 2), "max_features": 5000, "C": 1.0}, # ngram_range değişti
    {"ngram_range": (1, 1), "max_features": 3000, "C": 10.0} # max_features ve C değişti
]  # 1 kelime mi 2 kelime mi #max baz alincak kelime #model detayciligi

# Deneyleri MLflow'da bir isim altında toplayalım
mlflow.set_experiment("SMS Spam Detection Tuning")

# 3. DENEYLERİ ÇALIŞTIRMA VE KAYDETME
for params in experiment_params:
    # Her bir döngüde yeni bir MLflow "Run" (Koşu) başlat
    with mlflow.start_run():
        print(f"Running experiment with params: {params}")

        # a. Parametreleri MLflow'a kaydet
        mlflow.log_params(params)

        # b. Modeli bu parametrelerle eğit
        vectorizer = TfidfVectorizer(
            ngram_range=params["ngram_range"],
            max_features=params["max_features"],
            stop_words='english'
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        model = LogisticRegression(C=params["C"], random_state=42)
        model.fit(X_train_vec, y_train)

        # c. Metrikleri hesapla ve MLflow'a kaydet
        y_pred = model.predict(X_val_vec)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        print(f"  -> F1-Score: {f1:.4f}\n")

print("All experiments are complete!")