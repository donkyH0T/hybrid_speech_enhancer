# train_denoise.py
import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import random

# === параметры ===
SR = 16000
N_FFT = 512
HOP = 128
EPOCHS = 20
BATCH_SIZE = 32  # Увеличил размер батча

# === модель ===
def build_enhanced_denoise_model(n_fft=512):
    model = models.Sequential([
        layers.Input(shape=(n_fft//2 + 1, 2)),
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Conv1D(128, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.Conv1D(2, 3, activation='tanh', padding='same')
    ])
    return model

# === утилиты ===
def audio_to_stft_frames(audio, n_fft=N_FFT, hop=HOP):
    """Преобразует аудио в STFT и возвращает отдельные frames"""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    features = np.stack([stft.real, stft.imag], axis=-1)  # (freq, time, 2)
    
    # Транспонируем для получения отдельных frames
    frames = np.transpose(features, (1, 0, 2))  # (time, freq, 2)
    return frames

# === подготовка данных ===
def prepare_dataset(clean_files, noise_files, num_samples=200000):
    """Подготавливает dataset из парных clean/noisy frames"""
    X, Y = [], []
    
    # Создаем список пар файлов
    file_pairs = []
    for clean_file in clean_files:
        filename = os.path.basename(clean_file)
        noisy_file = os.path.join("data/noise", filename)
        
        if os.path.exists(noisy_file):
            file_pairs.append((clean_file, noisy_file))
    
    print(f"Найдено {len(file_pairs)} парных файлов")
    
    # Перемешиваем пары для разнообразия
    random.shuffle(file_pairs)
    
    for i, (clean_file, noisy_file) in enumerate(file_pairs):
        if len(X) >= num_samples:
            break
            
        if i % 100 == 0:
            print(f"Обработано {i} файлов, собрано {len(X)} frames")
            
        try:
            clean, _ = librosa.load(clean_file, sr=SR)
            noisy, _ = librosa.load(noisy_file, sr=SR)
        except Exception as e:
            print(f"Ошибка загрузки файлов: {e}")
            continue

        # Обеспечиваем одинаковую длину
        min_length = min(len(clean), len(noisy))
        if min_length < 2 * SR:
            continue
            
        clean_segment = clean[:2*SR]
        noisy_segment = noisy[:2*SR]
        
        # Нормализуем ВСЕ аудио целиком перед преобразованием в STFT
        max_val_clean = np.max(np.abs(clean_segment))
        max_val_noisy = np.max(np.abs(noisy_segment))
        
        if max_val_clean > 0:
            clean_segment = clean_segment / max_val_clean
        if max_val_noisy > 0:
            noisy_segment = noisy_segment / max_val_noisy
        
        # Преобразуем в STFT frames
        noisy_frames = audio_to_stft_frames(noisy_segment)
        clean_frames = audio_to_stft_frames(clean_segment)
        
        # Добавляем все frames из этого файла
        for noisy_frame, clean_frame in zip(noisy_frames, clean_frames):
            # Дополнительная нормализация каждого кадра (опционально)
            mag_noisy = np.sqrt(np.sum(noisy_frame**2, axis=-1))
            max_mag_noisy = np.max(mag_noisy)
            if max_mag_noisy > 0:
                noisy_frame = noisy_frame / max_mag_noisy
            
            mag_clean = np.sqrt(np.sum(clean_frame**2, axis=-1))
            max_mag_clean = np.max(mag_clean)
            if max_mag_clean > 0:
                clean_frame = clean_frame / max_mag_clean
            
            X.append(noisy_frame)
            Y.append(clean_frame)
            
            if len(X) >= num_samples:
                break

    return np.array(X), np.array(Y)

# === запуск ===
if __name__ == "__main__":
    # Собираем список файлов
    clean_files = glob.glob("data/clean_speech/*.wav")
    noise_files = glob.glob("data/noise/*.wav")

    if not clean_files:
        raise RuntimeError("❌ Не найдено файлов в data/clean_speech/")
    if not noise_files:
        raise RuntimeError("❌ Не найдено файлов в data/noise/")

    print(f"Найдено {len(clean_files)} чистых файлов и {len(noise_files)} шумовых файлов")

    # Создаем модель
    model = build_enhanced_denoise_model(N_FFT)
    model.compile(optimizer=optimizers.Adam(1e-4), loss="mse")  # Уменьшил learning rate
    model.summary()

    # Подготавливаем данные
    print("Подготовка данных...")
    X_train, Y_train = prepare_dataset(clean_files, noise_files, num_samples=90000)
    
    print(f"Размер dataset: {X_train.shape} -> {Y_train.shape}")
    print(f"Пример формы одного frame: {X_train[0].shape}")

    # Обучаем модель
    print("Начало обучения...")
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=1e-4,
        restore_best_weights=True
    )
    
    # Добавляем уменьшение learning rate
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
    )
    
    history = model.fit(X_train, Y_train, 
                       batch_size=BATCH_SIZE, 
                       epochs=EPOCHS, 
                       validation_split=0.1,
                       shuffle=True,
                       verbose=1,
                       callbacks=[early_stop, reduce_lr]
                       )

    # Сохраняем веса
    model.save_weights("denoise_model.weights.h5")
    print("✅ Модель обучена и сохранена в denoise_model.weights.h5")

    # Тестируем на одном примере
    test_idx = random.randint(0, len(X_train) - 1)
    test_input = X_train[test_idx:test_idx+1]
    test_output = model.predict(test_input)
    
    print(f"Тест: input {test_input.shape} -> output {test_output.shape}")
    print(f"Loss на тестовом примере: {np.mean((test_output - Y_train[test_idx:test_idx+1])**2):.6f}")