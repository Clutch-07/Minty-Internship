import joblib
from fastapi import FastAPI, HTTPException
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

def predict_text(text: str, model, vectorizer) -> dict:
    """
    Saf tahmin fonksiyonu: girilen metin için etiket ve güven skorunu döndürür.
    Test edilebilirlik için API'den ayrıştırıldı.
    """
    if model is None or vectorizer is None:
        raise RuntimeError("Model is not loaded.")
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    stripped = text.strip()
    if not stripped:
        raise ValueError("Input text is empty.")
    message_vec = vectorizer.transform([stripped])
    prediction = model.predict(message_vec)
    probability = model.predict_proba(message_vec)
    label = "SPAM" if prediction[0] == 1 else "HAM"
    confidence = float(probability[0][prediction[0]])
    return {"label": label, "confidence": confidence}

# 3. Tahmin Endpoint'ini Oluştur
# @app.post("/predict") ifadesi, bu fonksiyonun /predict adresine gelen
# POST isteklerini karşılayacağını belirtir.
@app.post("/predict")
async def predict(message: Message):
    """
    Bir metnin Spam mi yoksa Ham mi olduğunu tahmin eder.

    - **text**: Tahmin edilecek metin.
    """
    # a. Giriş doğrulama ve model kontrolü
    if model is None or vectorizer is None:
        # 500 yerine anlamlı mesaj
        raise HTTPException(status_code=503, detail="Model is not loaded. Please check server logs.")
    if message.text is None or not isinstance(message.text, str) or not message.text.strip():
        raise HTTPException(status_code=422, detail="Field 'text' must be a non-empty string.")

    # b. Tahmin yap
    result = predict_text(message.text, model, vectorizer)

    # c. Sonucu JSON olarak geri döndür
    return {
        "text": message.text,
        "prediction": result
    }

# Kök adrese basit bir hoş geldin mesajı ekleyelim
@app.get("/")
def read_root():
    return {"message": "Welcome to the Spam Detection API! Gidin /docs adresine gidin."}