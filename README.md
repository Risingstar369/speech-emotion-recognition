# 🎙️ Speech Emotion Recognition

Konuşma seslerinden duygu tanıma projesi. RAVDESS veri seti kullanılarak mutlu, üzgün ve nötr duyguları sınıflandıran bir derin öğrenme modeli geliştirilmiştir.

## 🎯 Tanınan Duygular
- 😊 Happy (Mutlu)
- 😢 Sad (Üzgün)
- 😐 Neutral (Nötr)

## 📂 Veri Seti
**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song
- 480 ses dosyası
- 24 profesyonel aktör
- 🔗 https://zenodo.org/record/1188976

## 🧠 Model
- Özellik çıkarma: MFCC (40 katsayı)
- Mimari: Dense Neural Network
- Veri dengeleme: Oversampling
- **Test doğruluğu: %93**

## 🛠️ Kurulum

```bash
pip install librosa numpy pandas scikit-learn tensorflow matplotlib
```

## 🚀 Çalıştırma

```bash
python model.py
```

## 📊 Sonuçlar
| Sınıf | Doğruluk |
|-------|----------|
| Happy | ✅ Yüksek |
| Sad | ✅ Yüksek |
| Neutral | ✅ Yüksek |
| **Genel** | **%93.10** |

## 👤 
Derin Öğrenme ve Yapay Sinir Ağları
2025-2026 Bahar Dönemi