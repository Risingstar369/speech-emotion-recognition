import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from collections import Counter

data_yolu = "data/archive"
duygular = {"01": "neutral", "03": "happy", "04": "sad"}

dosyalar = []
etiketler = []

for actor in os.listdir(data_yolu):
    actor_yolu = os.path.join(data_yolu, actor)
    if os.path.isdir(actor_yolu):
        for dosya in os.listdir(actor_yolu):
            if dosya.endswith(".wav"):
                parcalar = dosya.split("-")
                duygu_kodu = parcalar[2]
                if duygu_kodu in duygular:
                    dosyalar.append(os.path.join(actor_yolu, dosya))
                    etiketler.append(duygular[duygu_kodu])

print(f"Toplam dosya: {len(dosyalar)}")

# 1. Sinif dagilimi
sayac = Counter(etiketler)
plt.figure(figsize=(8, 5))
plt.bar(sayac.keys(), sayac.values(), color=['#2ecc71', '#e74c3c', '#3498db'])
plt.title('Sinif Dagilimi')
plt.xlabel('Duygu')
plt.ylabel('Dosya Sayisi')
for i, (k, v) in enumerate(sayac.items()):
    plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('sinif_dagilimi.png')
plt.show()

# 2. Her duygudan ornek dalga formu
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
for i, duygu in enumerate(['happy', 'sad', 'neutral']):
    idx = etiketler.index(duygu)
    ses, sr = librosa.load(dosyalar[idx], duration=3)
    axes[i].plot(np.linspace(0, 3, len(ses)), ses, color=['#2ecc71', '#e74c3c', '#3498db'][i])
    axes[i].set_title(f'{duygu.capitalize()} - Dalga Formu')
    axes[i].set_xlabel('Zaman (s)')
    axes[i].set_ylabel('Genlik')
plt.tight_layout()
plt.savefig('dalga_formlari.png')
plt.show()

# 3. MFCC gorsellestirme
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, duygu in enumerate(['happy', 'sad', 'neutral']):
    idx = etiketler.index(duygu)
    ses, sr = librosa.load(dosyalar[idx], duration=3)
    mfcc = librosa.feature.mfcc(y=ses, sr=sr, n_mfcc=40)
    librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[i])
    axes[i].set_title(f'MFCC - {duygu.capitalize()}')
    axes[i].set_ylabel('MFCC Katsayisi')
plt.tight_layout()
plt.savefig('mfcc_gorsellestirme.png')
plt.show()

print("Tum gorseller kaydedildi!")