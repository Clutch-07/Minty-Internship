# python
import unittest

# main.py içindeki saf fonksiyonu içe aktar
from main import predict_text

class MockVectorizer:
    def transform(self, texts):
        # Basitçe listeyi olduğu gibi taşıyormuş gibi davranan "vektör"
        return texts

class MockModelSpam:
    # "free", "winner", "prize" içerirse SPAM varsayımı yapan yalancı model
    spam_keywords = {"free", "winner", "prize", "claim", "credit", "offer"}

    def predict(self, X):
        out = []
        for t in X:
            t_low = t.lower()
            out.append(1 if any(k in t_low for k in self.spam_keywords) else 0)
        return out

    def predict_proba(self, X):
        # Sözde olasılık: keyword yakaladıysa 0.9, yoksa 0.8 HAM
        probs = []
        for y, t in zip(self.predict(X), X):
            if y == 1:
                probs.append([0.1, 0.9])
            else:
                probs.append([0.8, 0.2])
        return probs

class MockModelHam(MockModelSpam):
    # Her şeyi HAM döndürür
    def predict(self, X):
        return [0 for _ in X]

    def predict_proba(self, X):
        return [[0.95, 0.05] for _ in X]

class TestPredictText(unittest.TestCase):
    def setUp(self):
        self.vec = MockVectorizer()

    def test_spam_detection_with_keyword(self):
        model = MockModelSpam()
        result = predict_text("You are a WINNER! Claim your prize now", model, self.vec)
        self.assertIn(result["label"], ["SPAM", "HAM"])
        self.assertEqual(result["label"], "SPAM")
        self.assertGreaterEqual(result["confidence"], 0.5)

    def test_ham_when_no_keyword(self):
        model = MockModelSpam()
        result = predict_text("Let's have lunch at 12?", model, self.vec)
        self.assertEqual(result["label"], "HAM")
        self.assertGreater(result["confidence"], 0.5)

    def test_empty_text_raises(self):
        model = MockModelHam()
        with self.assertRaises(ValueError):
            predict_text("   ", model, self.vec)

if __name__ == "__main__":
    unittest.main()
