Output

C:\Users\borac\Documents\GitHub\Minty-Internship\.venv\Scripts\python.exe C:\Users\borac\Documents\GitHub\Minty-Internship\train_models.py 
Veri basariyla yuklendi
Toplam 5169 adet mesaj gozukuyor.
label
0    0.87367
1    0.12633
Name: proportion, dtype: float64
Egitim seti boyutu: 4135
Validasyon seti boyutu: 1034
Metin verileri TF-IDF vektörlerine dönüştürüldü.
Oluşturulan özellik (feature) sayısı: 7329
---Logistic Regression Modeli Egitiliyor.
Accuracy: 0.9584
Precision: 0.9889
Recall: 0.6794
F1-Score: 0.8054

Confusion Matrix:
[[902   1]
 [ 42  89]]

Classification Report:
              precision    recall  f1-score   support

         Ham       0.96      1.00      0.98       903
        Spam       0.99      0.68      0.81       131

    accuracy                           0.96      1034
   macro avg       0.97      0.84      0.89      1034
weighted avg       0.96      0.96      0.96      1034

---Multinomial Naive Bayes Modeli Egitiliyor.
Accuracy: 0.9662
Precision: 0.9898
Recall: 0.7405
F1-Score: 0.8472

Confusion Matrix:
[[902   1]
 [ 34  97]]

Classification Report:
              precision    recall  f1-score   support

         Ham       0.96      1.00      0.98       903
        Spam       0.99      0.74      0.85       131

    accuracy                           0.97      1034
   macro avg       0.98      0.87      0.91      1034
weighted avg       0.97      0.97      0.96      1034

---Random Forest Modeli Egitiliyor.
Accuracy: 0.9739
Precision: 0.9815
Recall: 0.8092
F1-Score: 0.8870

Confusion Matrix:
[[901   2]
 [ 25 106]]

Classification Report:
              precision    recall  f1-score   support

         Ham       0.97      1.00      0.99       903
        Spam       0.98      0.81      0.89       131

    accuracy                           0.97      1034
   macro avg       0.98      0.90      0.94      1034
weighted avg       0.97      0.97      0.97      1034


--- Model Karşılaştırma Tablosu ---
                         Accuracy  Precision    Recall  F1-Score
Logistic Regression      0.958414   0.988889  0.679389  0.805430
Multinomial Naive Bayes  0.966151   0.989796  0.740458  0.847162
Random Forest            0.973888   0.981481  0.809160  0.887029

Process finished with exit code 0

En iyi model: Random Forest


