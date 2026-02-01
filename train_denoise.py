import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import random
from tensorflow import keras
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# === параметры ===
SR = 16000
N_FFT = 512
HOP_LENGTH = 256
EPOCHS = 100
BATCH_SIZE = 32
MAX_AUDIO_LENGTH = 2 * SR  # 2 секунды

def build_light_denoise_model(input_shape):
    """
    Легкая модель для шумоподавления 2D спектрограмм
    input_shape: (частоты, время, 2) где 2 = real, imag части
    """
    inputs = layers.Input(shape=input_shape)
    
    # Преобразуем 2D сверткой (Conv2D вместо Conv1D)
    # Block 1
    x = layers.Conv2D(32, (5, 5), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x1 = x  # Сохраняем для skip connection
    
    # Block 2
    x = layers.Conv2D(64, (5, 5), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.1)(x)
    x2 = x
    
    # Middle block
    x = layers.Conv2D(128, (5, 5), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Decoder
    # Block 3
    x = layers.Conv2D(64, (5, 5), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.1)(x)
    
    # Добавляем skip connection
    x = layers.add([x, x2])
    
    # Block 4
    x = layers.Conv2D(32, (5, 5), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Добавляем skip connection
    x = layers.add([x, x1])
    
    # Output block - 2 канала (real, imag)
    outputs = layers.Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
    
    model = models.Model(inputs, outputs)
    return model

def build_fast_denoise_model(input_shape):
    """С остаточными связями - лучше сохраняет голос"""
    from tensorflow.keras import layers, models
    
    inputs = layers.Input(shape=input_shape)
    
    # Первый слой
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x_skip1 = x  # Сохраняем для skip connection
    
    # Второй слой
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Третий слой + skip connection
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, x_skip1])  # Добавляем оригинальный сигнал!
    x = layers.ReLU()(x)§
    
    outputs = layers.Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
    
    model = models.Model(inputs, outputs)
    print(f"✅ Fast модель с skip-connections создана")
    return model

def build_minimal_denoise_model(input_shape):
    """
    Минимальная модель - обучается за минуты!
    """
    inputs = layers.Input(shape=input_shape)
    
    # Всего 2 слоя!
    x = layers.Conv2D(8, (5, 5), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Conv2D(2, (5, 5), activation='tanh', padding='same')(x)
    
    model = models.Model(inputs, outputs)
    print(f"✅ Minimal модель создана")
    return model

def complex_mse_loss(y_true, y_pred):
    """MSE loss для комплексных чисел"""
    return tf.reduce_mean(tf.square(y_true - y_pred))


def load_and_preprocess_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio

def audio_to_spectrogram(audio):
    spec = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # Разделяем на реальную и мнимую части
    real = np.real(spec)
    imag = np.imag(spec)
    
    # Объединяем по каналам
    spec_2ch = np.stack([real, imag], axis=-1)
    return spec_2ch

# === утилиты для обработки аудио ===
def audio_to_stft_complex(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Преобразует аудио в комплексный STFT"""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    real = stft.real.T  # (time, freq)
    imag = stft.imag.T  # (time, freq)
    return real, imag

def stft_complex_to_audio(real, imag, hop_length=HOP_LENGTH):
    """Преобразует реальную и мнимую части обратно в аудио"""
    stft = (real + 1j * imag).T  # Транспонируем обратно
    audio = librosa.istft(stft, hop_length=hop_length)
    return audio

def preprocess_audio(audio, target_length=MAX_AUDIO_LENGTH):
    """Предобработка аудио: обрезка/дополнение и нормализация"""
    if len(audio) > target_length:
        # Случайная обрезка
        start = np.random.randint(0, len(audio) - target_length)
        audio = audio[start:start + target_length]
    else:
        # Дополнение нулями
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    
    # Нормализация по пиковому значению
    max_val = np.max(np.abs(audio)) + 1e-8
    audio = audio / max_val
    
    return audio

# === функция для тестирования ===
def test_model(model, test_files, output_dir="test_results"):
    """Тестирует модель на нескольких файлах с визуализацией"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (clean_file, noisy_file) in enumerate(test_files[:3]):
        try:
            print(f"Тестирование файла {i+1}/{min(3, len(test_files))}")
            
            # Загружаем аудио
            clean_audio, _ = librosa.load(clean_file, sr=SR)
            noisy_audio, _ = librosa.load(noisy_file, sr=SR)
            
            # Предобработка
            clean_audio = preprocess_audio(clean_audio)
            noisy_audio = preprocess_audio(noisy_audio)
            
            # Преобразуем в STFT
            noisy_real, noisy_imag = audio_to_stft_complex(noisy_audio)
            
            # Обрабатываем батчами для эффективности
            enhanced_real, enhanced_imag = [], []
            batch_size = 32
            
            for start_idx in range(0, len(noisy_real), batch_size):
                end_idx = min(start_idx + batch_size, len(noisy_real))
                batch_frames = []
                
                for j in range(start_idx, end_idx):
                    input_frame = np.stack([noisy_real[j], noisy_imag[j]], axis=-1)
                    batch_frames.append(input_frame)
                
                batch_frames = np.array(batch_frames)
                predictions = model.predict(batch_frames, verbose=0)
                
                for pred in predictions:
                    enhanced_real.append(pred[:, 0])
                    enhanced_imag.append(pred[:, 1])
            
            # Обратное преобразование
            enhanced_audio = stft_complex_to_audio(np.array(enhanced_real), 
                                                 np.array(enhanced_imag))
            
            # Обрезаем до исходной длины
            min_length = min(len(clean_audio), len(enhanced_audio))
            clean_audio = clean_audio[:min_length]
            enhanced_audio = enhanced_audio[:min_length]
            
            # Сохраняем результаты
            sf.write(os.path.join(output_dir, f"original_{i}.wav"), clean_audio, SR)
            sf.write(os.path.join(output_dir, f"noisy_{i}.wav"), noisy_audio[:min_length], SR)
            sf.write(os.path.join(output_dir, f"enhanced_{i}.wav"), enhanced_audio, SR)
            
            # Визуализация спектрограмм
            plt.figure(figsize=(15, 12))
            
            # Original
            plt.subplot(3, 1, 1)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(clean_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)), ref=np.max)
            librosa.display.specshow(D, sr=SR, hop_length=HOP_LENGTH, 
                                   y_axis='log', x_axis='time')
            plt.title('Original Clean Audio')
            plt.colorbar(format='%+2.0f dB')
            
            # Noisy
            plt.subplot(3, 1, 2)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_audio[:min_length], n_fft=N_FFT, hop_length=HOP_LENGTH)), ref=np.max)
            librosa.display.specshow(D, sr=SR, hop_length=HOP_LENGTH, 
                                   y_axis='log', x_axis='time')
            plt.title('Noisy Audio')
            plt.colorbar(format='%+2.0f dB')
            
            # Enhanced
            plt.subplot(3, 1, 3)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)), ref=np.max)
            librosa.display.specshow(D, sr=SR, hop_length=HOP_LENGTH, 
                                   y_axis='log', x_axis='time')
            plt.title('Enhanced Audio')
            plt.colorbar(format='%+2.0f dB')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"spectrogram_{i}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Сохранены результаты для файла {i}")
            
        except Exception as e:
            print(f"Ошибка при тестировании файла {clean_file}: {e}")
            continue

def prepare_batch_from_tuples(file_tuples, num_samples=None, target_time_frames=256):
    """
    Подготовка данных с ФИКСИРОВАННОЙ длиной
    """
    if num_samples is None:
        num_samples = len(file_tuples)
    
    clean_specs = []
    noisy_specs = []
    
    for clean_path, noisy_path in file_tuples[:num_samples]:
        try:
            # Загружаем аудио
            clean_audio = load_and_preprocess_audio(clean_path)
            noisy_audio = load_and_preprocess_audio(noisy_path)
            
            # Преобразуем в спектрограммы
            clean_spec = audio_to_spectrogram(clean_audio)
            noisy_spec = audio_to_spectrogram(noisy_audio)
            
            # ФУНКЦИЯ ДЛЯ ФИКСАЦИИ РАЗМЕРА
            def fix_size(spec, target_frames=256):
                current_frames = spec.shape[1]
                
                if current_frames >= target_frames:
                    # Обрезаем
                    start = (current_frames - target_frames) // 2
                    return spec[:, start:start + target_frames, :]
                else:
                    # ДОПОЛНЯЕМ НУЛЯМИ
                    padding = target_frames - current_frames
                    # Дополняем с обеих сторон для симметрии
                    left_pad = padding // 2
                    right_pad = padding - left_pad
                    
                    return np.pad(spec, 
                                ((0, 0), (left_pad, right_pad), (0, 0)), 
                                mode='constant', 
                                constant_values=0)
            
            # Приводим к одному размеру
            clean_spec = fix_size(clean_spec, target_time_frames)
            noisy_spec = fix_size(noisy_spec, target_time_frames)
            
            clean_specs.append(clean_spec)
            noisy_specs.append(noisy_spec)
            
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
            continue
    
    if not clean_specs:
        raise ValueError("Нет данных!")
    
    print(f"✅ Данные подготовлены. Форма: {clean_specs[0].shape}")
    return np.array(noisy_specs), np.array(clean_specs)

def prepare_dataset_from_tuples(file_tuples, num_samples=10000, val_split=0.15):
    """
    Полная подготовка данных из кортежей с разделением на train/val
    """
    # Случайно перемешиваем
    np.random.shuffle(file_tuples)
    
    # Ограничиваем количество
    if num_samples is not None:
        file_tuples = file_tuples[:min(num_samples, len(file_tuples))]
    
    # Подготавливаем все данные
    X_all, Y_all = prepare_batch_from_tuples(file_tuples)
    
    # Разделяем на train/val
    split_idx = int(len(X_all) * (1 - val_split))
    
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    Y_train, Y_val = Y_all[:split_idx], Y_all[split_idx:]
    
    print(f"\nРазделение данных:")
    print(f"Всего: {len(X_all)} примеров")
    print(f"Train: {len(X_train)} примеров")
    print(f"Val:   {len(X_val)} примеров")
    print(f"Форма X_train: {X_train.shape}")
    print(f"Форма Y_train: {Y_train.shape}")
    
    return (X_train, Y_train), (X_val, Y_val)

# === основной код ===
if __name__ == "__main__":
    # ... (инициализация как раньше)
    
    # Собираем файлы
    clean_files = glob.glob("data/clean_speech/*.wav")
    noisy_files_dir = "data/noise/"
    
    print(f"Найдено {len(clean_files)} чистых файлов")
    
    # Создаем кортежи (чистый, шумный) из ВСЕХ файлов
    all_tuples = [(f, os.path.join(noisy_files_dir, os.path.basename(f))) 
                  for f in clean_files]
    
    # Проверяем существование файлов
    valid_tuples = []
    for clean_path, noisy_path in all_tuples:
        if os.path.exists(clean_path) and os.path.exists(noisy_path):
            valid_tuples.append((clean_path, noisy_path))
        else:
            print(f"⚠️ Пропущена пара: {clean_path}")
    
    print(f"Доступно {len(valid_tuples)} валидных пар")
    
    # Перемешиваем
    np.random.shuffle(valid_tuples)
    
    # Можно взять все или ограничить количество
    if len(valid_tuples) > 10000:
        print(f"Ограничиваем до 10000 пар для быстрого обучения")
        valid_tuples = valid_tuples[:10000]
    
    # Определяем input shape
    TARGET_TIME_FRAMES = 256  # Выберите удобное число (128, 256, 512 и т.д.)
    FREQ_BINS = 257  # N_FFT//2 + 1
    input_shape = (FREQ_BINS, TARGET_TIME_FRAMES, 2)
    
    print(f"Input shape: {input_shape}")
    
    # Создаем модель
    print("Создание легкой модели для шумоподавления...")
    model = build_fast_denoise_model(input_shape)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=complex_mse_loss,
        metrics=['mae']
    )
    
    model.summary()
    
    # Подготавливаем ВСЕ данные
    print("Подготовка данных...")
    (X_train, Y_train), (X_val, Y_val) = prepare_dataset_from_tuples(
        valid_tuples, 
        num_samples=len(valid_tuples),  # все данные
        val_split=0.15  # 15% на валидацию
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            "models/best_light_model.weights.h5",
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]
    
    # Обучаем модель
    print("Начало обучения...")
    history = model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, Y_val),
        shuffle=True,
        verbose=1,
        callbacks=callbacks
    )
    
    # Сохраняем модель
    model.save("models/final_light_denoise_model.h5")
    model.save_weights("models/final_light_denoise_model.weights.h5")
    print("✅ Легкая модель обучена и сохранена")
    
    print("✅ Обучение завершено! Результаты в test_results/ и models/")