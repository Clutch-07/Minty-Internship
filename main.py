import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# 1. FastAPI uygulamasını başlat
app = FastAPI(title="SMS Spam Detection API", description="An API to predict if a message is Spam or Ham")

# Pydantic ile istek gövdesinin (request body) yapısını tanımla
# Bu, FastAPI'ye gelen JSON'un {"text": "..."} formatında olmasını sağlar
class Message(BaseModel):
    text: str

# 2. Modeli ve Vectorizer'ı Sadece Bir Kez Yükle
# Uygulama başlarken model dosyalarını belleğe yükleriz.
# Bu sayede her istekte tekrar tekrar yükleme yapmayız.
try:
    vectorizer = joblib.load('vectorizer.joblib')
    model = joblib.load('model.joblib')
    print("Model ve vectorizer başarıyla yüklendi.")
except FileNotFoundError:
    print("Hata: Model veya vectorizer dosyaları bulunamadı. Lütfen 'save_model.py' scriptini çalıştırın.")
    vectorizer = None
    model = None

# 3. Tahmin Endpoint'ini Oluştur
# @app.post("/predict") ifadesi, bu fonksiyonun /predict adresine gelen
# POST isteklerini karşılayacağını belirtir.
@app.post("/predict")
async def predict(message: Message):
    """
    Bir metnin Spam mi yoksa Ham mi olduğunu tahmin eder.

    - **text**: Tahmin edilecek metin.
    """
    if not model or not vectorizer:
        return {"error": "Model is not loaded. Please check server logs."}

    # a. Gelen metni vektöre dönüştür
    message_vec = vectorizer.transform([message.text])

    # b. Tahmin yap
    prediction = model.predict(message_vec)
    probability = model.predict_proba(message_vec)

    # c. Sonucu hazırla
    label = "SPAM" if prediction[0] == 1 else "HAM"
    confidence = probability[0][prediction[0]]

    # d. Sonucu JSON olarak geri döndür
    return {
        "text": message.text,
        "prediction": {
            "label": label,
            "confidence": float(confidence)
        }
    }

# Kök adrese basit bir hoş geldin mesajı ekleyelim
@app.get("/")
def read_root():
    return {"message": "Welcome to the Spam Detection API! Gidin /docs adresine gidin."}