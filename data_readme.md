--- Veri Seti Hakkında İlk Bilgiler ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5572 entries, 0 to 5571
Data columns (total 2 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   label    5572 non-null   object
 1   message  5572 non-null   object
dtypes: object(2)
memory usage: 87.2+ KB
None

--- Eksik Değer Kontrolü ---
label      0
message    0
dtype: int64

--- Yinelenen Kayıt Kontrolü ---
Yinelenen kayıt sayısı: 403
Yinelenen kayıtlar silindikten sonra satır sayısı: 5169

--- Sınıf Dağılımı ---
label
ham     4516
spam     653
Name: count, dtype: int64

Temizlenmiş veri 'sms_clean.csv' olarak kaydedildi.
