import pandas as pd

df = pd.read_csv("C:/Users/USER/Desktop/spam.csv", encoding="latin-1")

# df.isnull().sum() #eksik degerleri bulma
# df.dropna() #eksik degerleri silme

# df.duplicated().sum() #yinelenen degerleri bulma
# df.drop_duplicates(inplace=True)   #yinelenen degerleri silme

# Sadece ilk iki kolonu alarak gereksiz kolonları sil ve kolon adlarını düzenle.
df = df.iloc[:, :2]
df.columns = ['label', 'message']

print("--- Veri Seti Hakkında İlk Bilgiler ---")
print(df.info())

# Eksik değerleri kontrol et.
print("\n--- Eksik Değer Kontrolü ---")
print(df.isnull().sum())

# Yinelenen kayıtları bul ve sil.
print("\n--- Yinelenen Kayıt Kontrolü ---")
print(f"Yinelenen kayıt sayısı: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Yinelenen kayıtlar silindikten sonra satır sayısı: {len(df)}")

# Sınıf dağılımını (ham/spam) hesapla.
print("\n--- Sınıf Dağılımı ---")
print(df['label'].value_counts())

# Temizlenmiş dosyayı kaydet.
df.to_csv('sms_clean.csv', index=False)
print("\nTemizlenmiş veri 'sms_clean.csv' olarak kaydedildi.")