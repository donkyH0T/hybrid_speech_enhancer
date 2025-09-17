import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import random
import soundfile as sf

# === параметры ===
SR = 16000
N_FFT = 512
HOP_LENGTH = 256  # Увеличим hop length для уменьшения количества кадров
EPOCHS = 50
BATCH_SIZE = 64
MAX_AUDIO_LENGTH = 2 * SR  # 2 секунды

# === простая и эффективная модель ===
def build_simple_denoise_model(input_shape):
    """Простая последовательная модель без skip connections"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Первый блок
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Второй блок
        layers.Conv1D(128, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Третий блок
        layers.Conv1D(256, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Четвертый блок
        layers.Conv1D(256, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Пятый блок
        layers.Conv1D(128, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Шестой блок
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        # Выходной слой
        layers.Conv1D(2, 3, activation='tanh', padding='same')
    ])
    return model

# === утилиты для обработки аудио ===
def audio_to_stft_complex(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Преобразует аудио в комплексный STFT"""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    real = stft.real.T  # Транспонируем для (time, freq)
    imag = stft.imag.T  # Транспонируем для (time, freq)
    return real, imag

def stft_complex_to_audio(real, imag, hop_length=HOP_LENGTH):
    """Преобразует реальную и мнимую части обратно в аудио"""
    stft = (real + 1j * imag).T  # Транспонируем обратно
    audio = librosa.istft(stft, hop_length=hop_length)
    return audio

def preprocess_audio(audio):
    """Предобработка аудио: обрезка/дополнение и нормализация"""
    # Обрезаем или дополняем до нужной длины
    if len(audio) > MAX_AUDIO_LENGTH:
        audio = audio[:MAX_AUDIO_LENGTH]
    else:
        padding = MAX_AUDIO_LENGTH - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    
    # Нормализация
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    return audio

# === подготовка данных ===
def prepare_dataset(clean_files, noisy_files_dir, num_samples=30000):
    """Подготавливает dataset из парных clean/noisy frames"""
    X_real, X_imag, Y_real, Y_imag = [], [], [], []
    
    # Создаем список пар файлов
    file_pairs = []
    for clean_file in clean_files:
        filename = os.path.basename(clean_file)
        noisy_file = os.path.join(noisy_files_dir, filename)
        
        if os.path.exists(noisy_file):
            file_pairs.append((clean_file, noisy_file))
    
    print(f"Найдено {len(file_pairs)} парных файлов")
    
    if not file_pairs:
        raise ValueError("Не найдено парных файлов для обучения!")
    
    random.shuffle(file_pairs)
    
    for i, (clean_file, noisy_file) in enumerate(file_pairs):
        if len(X_real) >= num_samples:
            break
            
        if i % 100 == 0:
            print(f"Обработано {i} файлов, собрано {len(X_real)} frames")
            
        try:
            # Загружаем аудио
            clean_audio, _ = librosa.load(clean_file, sr=SR)
            noisy_audio, _ = librosa.load(noisy_file, sr=SR)
            
            # Предобработка
            clean_audio = preprocess_audio(clean_audio)
            noisy_audio = preprocess_audio(noisy_audio)
            
            # Преобразуем в STFT
            clean_real, clean_imag = audio_to_stft_complex(clean_audio)
            noisy_real, noisy_imag = audio_to_stft_complex(noisy_audio)
            
            # Убедимся, что размеры совпадают
            min_frames = min(len(clean_real), len(noisy_real))
            clean_real = clean_real[:min_frames]
            clean_imag = clean_imag[:min_frames]
            noisy_real = noisy_real[:min_frames]
            noisy_imag = noisy_imag[:min_frames]
            
            # Нормализуем каждый кадр
            for cr, ci, nr, ni in zip(clean_real, clean_imag, noisy_real, noisy_imag):
                # Нормализация clean
                clean_mag = np.sqrt(np.sum(cr**2 + ci**2))
                if clean_mag > 0:
                    cr = cr / clean_mag
                    ci = ci / clean_mag
                
                # Нормализация noisy
                noisy_mag = np.sqrt(np.sum(nr**2 + ni**2))
                if noisy_mag > 0:
                    nr = nr / noisy_mag
                    ni = ni / noisy_mag
                
                # Сохраняем как отдельные каналы
                X_real.append(nr)
                X_imag.append(ni)
                Y_real.append(cr)
                Y_imag.append(ci)
                
                if len(X_real) >= num_samples:
                    break
                    
        except Exception as e:
            print(f"Ошибка обработки файлов {clean_file}, {noisy_file}: {e}")
            continue
    
    # Объединяем реальную и мнимую части
    X = np.stack([np.array(X_real), np.array(X_imag)], axis=-1)
    Y = np.stack([np.array(Y_real), np.array(Y_imag)], axis=-1)
    
    return X, Y

# === функция для тестирования ===
def test_model_simple(model, test_files, output_dir="test_results"):
    """Тестирует модель на нескольких файлах"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (clean_file, noisy_file) in enumerate(test_files[:3]):
        try:
            # Загружаем аудио
            clean_audio, _ = librosa.load(clean_file, sr=SR)
            noisy_audio, _ = librosa.load(noisy_file, sr=SR)
            
            clean_audio = preprocess_audio(clean_audio)
            noisy_audio = preprocess_audio(noisy_audio)
            
            # Преобразуем в STFT
            noisy_real, noisy_imag = audio_to_stft_complex(noisy_audio)
            
            # Обрабатываем каждый кадр
            enhanced_real, enhanced_imag = [], []
            for nr, ni in zip(noisy_real, noisy_imag):
                # Подготавливаем вход
                input_frame = np.stack([nr, ni], axis=-1)
                input_frame = np.expand_dims(input_frame, axis=0)
                
                # Предсказываем
                output_frame = model.predict(input_frame, verbose=0)[0]
                
                enhanced_real.append(output_frame[:, 0])
                enhanced_imag.append(output_frame[:, 1])
            
            # Обратное преобразование
            enhanced_audio = stft_complex_to_audio(np.array(enhanced_real), 
                                                 np.array(enhanced_imag))
            
            # Сохраняем результаты
            sf.write(os.path.join(output_dir, f"original_{i}.wav"), clean_audio, SR)
            sf.write(os.path.join(output_dir, f"noisy_{i}.wav"), noisy_audio, SR)
            sf.write(os.path.join(output_dir, f"enhanced_{i}.wav"), enhanced_audio, SR)
            
            print(f"Сохранены тестовые результаты для файла {i}")
            
        except Exception as e:
            print(f"Ошибка при тестировании: {e}")

# === основной код ===
if __name__ == "__main__":
    # Игнорируем предупреждения AVX
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Создаем директории
    os.makedirs("models", exist_ok=True)
    os.makedirs("test_results", exist_ok=True)
    
    # Собираем файлы
    clean_files = glob.glob("data/clean_speech/*.wav")
    noisy_files_dir = "data/noise/"
    
    if not clean_files:
        raise RuntimeError("❌ Не найдено файлов в data/clean_speech/")
    if not os.path.exists(noisy_files_dir):
        raise RuntimeError("❌ Не найдена папка data/noise/")

    print(f"Найдено {len(clean_files)} чистых файлов")

    # Создаем модель
    input_shape = (N_FFT//2 + 1, 2)  # (257, 2)
    model = build_simple_denoise_model(input_shape)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), 
                 loss="mse",
                 metrics=['mae'])
    model.summary()

    # Подготавливаем данные
    print("Подготовка данных...")
    X_train, Y_train = prepare_dataset(clean_files, noisy_files_dir, num_samples=30000)
    
    print(f"Размер dataset: X {X_train.shape}, Y {Y_train.shape}")

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint("models/best_model.h5", monitor='val_loss', save_best_only=True)
    ]

    # Обучаем модель
    print("Начало обучения...")
    history = model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        shuffle=True,
        verbose=1,
        callbacks=callbacks
    )

    # Сохраняем модель
    model.save("models/final_denoise_model.weights.h5")
    print("✅ Модель обучена и сохранена")

    # Тестируем
    print("Тестирование модели...")
    test_files = [(clean, os.path.join(noisy_files_dir, os.path.basename(clean))) 
                 for clean in clean_files[:3]]
    test_model_simple(model, test_files)
    
    print("✅ Обучение завершено! Результаты в test_results/")