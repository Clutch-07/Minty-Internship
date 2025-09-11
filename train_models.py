import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. günde hazırladığın temiz veriyi yükle
df = pd.read_csv('sms_clean.csv')

df['label'] = df['label'].map({'ham':0,'spam':1})
X = df['message']
y = df['label']

print("Veri basariyla yuklendi")
print(f"Toplam {len(df)} adet mesaj gozukuyor.")
print(y.value_counts(normalize=True))

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size = 0.20,
    random_state = 42,
    stratify = y

)

print(f'Egitim seti boyutu: {len(X_train)}')
print(f'Validasyon seti boyutu: {len(X_val)}')

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df = 0.7)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

print("Metin verileri TF-IDF vektörlerine dönüştürüldü.")
print(f"Oluşturulan özellik (feature) sayısı: {X_train_tfidf.shape[1]}")

models = {"Logistic Regression": LogisticRegression(random_state = 42),
          "Multinomial Naive Bayes": MultinomialNB(),
          "Random Forest": RandomForestClassifier(n_estimators = 100, random_state = 42)}

results = {}

for model_name,model in models.items():
    print(f"---{model_name} Modeli Egitiliyor.")

    #Modeli egit
    model.fit(X_train_tfidf,y_train)

    #Validasyon Seti uzerinde tahmin yap.
    y_pred = model.predict(X_val_tfidf)

    #Metrikleri hesapla
    accuracy = accuracy_score(y_val,y_pred)
    precision = precision_score(y_val,y_pred)
    recall = recall_score(y_val,y_pred)
    f1 = f1_score(y_val,y_pred)

    #Sonuclari sakla


    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    #Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val,y_pred)

    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Ham', 'Spam']))

    # Confusion Matrix'i görselleştirelim (isteğe bağlı ama şık)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Değer')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.savefig(f"{model_name.replace(' ', '_')}_cm.png")  # Görseli dosyaya kaydet
    plt.show()

# Sonuçları daha rahat görmek için bir DataFrame'e çevir
results_df = pd.DataFrame(results).T
print("\n--- Model Karşılaştırma Tablosu ---")
print(results_df)
