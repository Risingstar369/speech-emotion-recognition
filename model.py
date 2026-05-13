import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def mfcc_cikar(dosya_yolu):
    try:
        ses, sr = librosa.load(dosya_yolu, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=ses, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except:
        return None

def veri_yukle():
    X, y = [], []
    duygular = {"01": "neutral", "03": "happy", "04": "sad"}
    data_yolu = "data/archive"
    for actor in os.listdir(data_yolu):
        actor_yolu = os.path.join(data_yolu, actor)
        if os.path.isdir(actor_yolu):
            for dosya in os.listdir(actor_yolu):
                if dosya.endswith(".wav"):
                    parcalar = dosya.split("-")
                    duygu_kodu = parcalar[2]
                    if duygu_kodu in duygular:
                        mfcc = mfcc_cikar(os.path.join(actor_yolu, dosya))
                        if mfcc is not None:
                            X.append(mfcc)
                            y.append(duygular[duygu_kodu])
    return np.array(X), np.array(y)

print("Veri yükleniyor...")
X, y = veri_yukle()

# Sınıf dağılımını göster
unique, counts = np.unique(y, return_counts=True)
print("\nSınıf dağılımı:")
for u, c in zip(unique, counts):
    print(f"  {u}: {c} dosya")

# Dengeleme — neutral sınıfını çoğalt (oversampling)
X_df = list(zip(X, y))
siniflar = {}
for xi, yi in X_df:
    siniflar.setdefault(yi, []).append(xi)

max_sayi = max(len(v) for v in siniflar.values())
X_dengeli, y_dengeli = [], []
for sinif, ornekler in siniflar.items():
    if len(ornekler) < max_sayi:
        ornekler = resample(ornekler, n_samples=max_sayi, random_state=42)
    X_dengeli.extend(ornekler)
    y_dengeli.extend([sinif] * max_sayi)

X = np.array(X_dengeli)
y = np.array(y_dengeli)

print(f"\nDengeleme sonrası toplam: {len(X)} örnek")

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Etiket
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Train/test böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

print(f"Eğitim: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"Sınıflar: {le.classes_}")

# Model — basit Dense (MFCC vektörü için daha uygun)
model = Sequential([
    Dense(256, activation='relu', input_shape=(40,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Doğruluğu: %{test_acc*100:.2f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Test')
plt.title('Doğruluk')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim')
plt.plot(history.history['val_loss'], label='Test')
plt.title('Kayıp')
plt.legend()
plt.tight_layout()
plt.savefig('sonuclar.png')
plt.show()

model.save('ser_model.keras')
print("✅ Model kaydedildi!")