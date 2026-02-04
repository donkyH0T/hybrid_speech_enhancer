import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import random
from tensorflow import keras
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

warnings.filterwarnings('ignore')

# === параметры ===
SR = 16000
N_FFT = 512
HOP_LENGTH = 256
EPOCHS = 100
BATCH_SIZE = 8
TARGET_TIME_FRAMES = 256
FREQ_BINS = N_FFT // 2 + 1  # 257
MAX_SAMPLES = 1000
CONTEXT_FRAMES = 5

# Параметры спектрального вычитания (для предобработки)
SPEC_SUBTRACTION_ALPHA = 1.5
SPEC_SUBTRACTION_BETA = 0.1

def setup_colab_environment():
    """Настройка окружения для Google Colab"""
    print("="*60)
    print("НАСТРОЙКА GOOGLE COLAB")
    print("="*60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU доступен: {len(gpus)} устройств")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"⚠️ Ошибка GPU: {e}")
    else:
        print("❌ GPU не найден, используется CPU")
    
    temp_dir = "/tmp/speech_enhancement"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(f"{temp_dir}/data/clean_speech", exist_ok=True)
    os.makedirs(f"{temp_dir}/data/noise", exist_ok=True)
    os.makedirs(f"{temp_dir}/models", exist_ok=True)
    
    print(f"📁 Временная директория: {temp_dir}")
    print("="*60)
    
    return temp_dir

def copy_data_to_tmp(drive_path, temp_dir):
    """Копируем данные с Google Диска на локальный SSD"""
    print("\n📁 Копирование данных на локальный SSD...")
    
    drive_data_path = os.path.join(drive_path, "data")
    
    if os.path.exists(drive_data_path):
        clean_src = os.path.join(drive_data_path, "clean_speech")
        noise_src = os.path.join(drive_data_path, "noise")
        
        clean_dst = os.path.join(temp_dir, "data/clean_speech")
        noise_dst = os.path.join(temp_dir, "data/noise")
        
        max_files = 10000
        
        def copy_files(src_dir, dst_dir, file_type="wav"):
            os.makedirs(dst_dir, exist_ok=True)
            files = glob.glob(os.path.join(src_dir, f"*.{file_type}"))[:max_files]
            
            print(f"Копируем {len(files)} файлов из {src_dir}...")
            
            for i, src_file in enumerate(files):
                if i % 100 == 0:
                    print(f"  Скопировано {i}/{len(files)} файлов")
                
                dst_file = os.path.join(dst_dir, os.path.basename(src_file))
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
            
            return len(files)
        
        clean_count = copy_files(clean_src, clean_dst, "wav")
        noise_count = copy_files(noise_src, noise_dst, "wav")
        
        print(f"✅ Скопировано {clean_count} чистых и {noise_count} шумных файлов")
        return clean_dst, noise_dst
    else:
        print(f"⚠️ Папка {drive_data_path} не найдена!")
        return None, None

def spectral_subtraction(noisy_spec, noise_estimate, alpha=1.5, beta=0.1):
    """Применяет спектральное вычитание для предобработки."""
    mag_noisy = np.abs(noisy_spec)
    phase = np.angle(noisy_spec)
    
    # Защита речи от чрезмерного подавления
    speech_mask = np.maximum(0, 1 - beta * noise_estimate / (mag_noisy + 1e-8))
    speech_mask = np.minimum(1, speech_mask)
    
    # Спектральное вычитание
    mag_clean = np.maximum(0, mag_noisy - alpha * noise_estimate)
    mag_clean = mag_clean * speech_mask
    
    return mag_clean * np.exp(1j * phase)

def wiener_filter(stft_features, noise_estimate):
    """Применяет фильтр Винера для постобработки."""
    mag = np.abs(stft_features)
    phase = np.angle(stft_features)
    
    # Оценка отношения сигнал/шум
    snr_estimate = np.maximum(0, mag**2 / (noise_estimate**2 + 1e-8) - 1)
    wiener_gain = snr_estimate / (snr_estimate + 1)
    
    # Применяем фильтр
    mag_clean = mag * wiener_gain
    
    return mag_clean * np.exp(1j * phase)

def estimate_noise_from_audio(audio, sr=16000, n_fft=512, hop_length=256, win_length=512):
    """Оценивает шумовой спектр из аудио (первые 100 мс)."""
    # Берем первые 100 мс для оценки шума
    noise_samples = int(0.1 * sr)
    if len(audio) > noise_samples:
        noise_segment = audio[:noise_samples]
    else:
        noise_segment = audio
    
    noise_spec = librosa.stft(
        noise_segment,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann'
    )
    
    return np.mean(np.abs(noise_spec), axis=1, keepdims=True)

def process_single_file_with_pipeline(args):
    """Обработка файла с полным каскадом: SS → NN → Wiener."""
    clean_path, noisy_path, target_frames = args
    
    try:
        # 1. Загрузка аудио
        target_samples = target_frames * HOP_LENGTH
        target_seconds = target_samples / SR
        
        clean_audio = librosa.load(clean_path, sr=SR, duration=target_seconds)[0]
        noisy_audio = librosa.load(noisy_path, sr=SR, duration=target_seconds)[0]
        
        # 2. Выравнивание длины
        if len(clean_audio) < target_samples:
            pad_len = target_samples - len(clean_audio)
            clean_audio = np.pad(clean_audio, (0, pad_len), mode='constant')
        clean_audio = clean_audio[:target_samples]
        
        if len(noisy_audio) < target_samples:
            pad_len = target_samples - len(noisy_audio)
            noisy_audio = np.pad(noisy_audio, (0, pad_len), mode='constant')
        noisy_audio = noisy_audio[:target_samples]
        
        # 3. STFT
        clean_spec = librosa.stft(
            clean_audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=N_FFT,
            window='hann',
            center=False
        )
        
        noisy_spec = librosa.stft(
            noisy_audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=N_FFT,
            window='hann',
            center=False
        )
        
        # 4. Оценка шума
        noise_estimate = estimate_noise_from_audio(noisy_audio, SR, N_FFT, HOP_LENGTH, N_FFT)
        
        # 5. СПЕКТРАЛЬНОЕ ВЫЧИТАНИЕ (предобработка)
        preprocessed_spec = spectral_subtraction(
            noisy_spec, 
            noise_estimate,
            alpha=SPEC_SUBTRACTION_ALPHA,
            beta=SPEC_SUBTRACTION_BETA
        )
        
        # 6. Подготовка данных для нейросети
        # Нейросеть будет обучаться улучшать результат спектрального вычитания
        preprocessed_real = np.real(preprocessed_spec)
        preprocessed_imag = np.imag(preprocessed_spec)
        clean_real = np.real(clean_spec)
        clean_imag = np.imag(clean_spec)
        
        # 7. Приведение к правильному размеру
        if clean_spec.shape[1] != target_frames:
            if clean_spec.shape[1] > target_frames:
                clean_spec = clean_spec[:, :target_frames]
                preprocessed_spec = preprocessed_spec[:, :target_frames]
            else:
                pad_width = target_frames - clean_spec.shape[1]
                clean_spec = np.pad(clean_spec, ((0, 0), (0, pad_width)), mode='constant')
                preprocessed_spec = np.pad(preprocessed_spec, ((0, 0), (0, pad_width)), mode='constant')
        
        # 8. Фильтр Винера на идеальном сигнале (цель для обучения)
        # Это то, что нейросеть должна научиться приближать
        wiener_target = wiener_filter(clean_spec, noise_estimate)
        
        # 9. Подготовка финальных данных
        # Вход: спектрально вычтенные данные (2 канала: real, imag)
        # Цель: результат фильтра Винера на чистом сигнале
        input_spec_2ch = np.stack([np.real(preprocessed_spec), np.imag(preprocessed_spec)], axis=-1)
        target_spec_2ch = np.stack([np.real(wiener_target), np.imag(wiener_target)], axis=-1)
        
        # Проверка качества
        filename = os.path.basename(clean_path)
        if np.any(np.isnan(input_spec_2ch)) or np.any(np.isinf(input_spec_2ch)):
            print(f"❌ {filename}: NaN/Inf в входных данных")
            return None
        
        return input_spec_2ch, target_spec_2ch, noise_estimate
        
    except Exception as e:
        print(f"❌ Ошибка в файле {os.path.basename(clean_path)}: {str(e)[:100]}")
        return None

def prepare_data_pipeline_parallel(file_tuples, num_samples=1000, target_time_frames=256):
    """Параллельная подготовка данных с полным каскадом обработки."""
    print(f"\n⚡ ПОДГОТОВКА ДАННЫХ С КАСКАДОМ SS→NN→WIENER")
    print(f"Обработка {min(num_samples, len(file_tuples))} файлов")
    
    start_time = time.time()
    
    file_tuples = file_tuples[:min(num_samples, len(file_tuples))]
    args_list = [(clean, noisy, target_time_frames) for clean, noisy in file_tuples]
    
    input_specs = []
    target_specs = []
    noise_estimates = []
    successful = 0
    
    print(f"Используется {multiprocessing.cpu_count()} ядер CPU")
    
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(process_single_file_with_pipeline, args_list))
    
    for result in results:
        if result is not None:
            input_spec, target_spec, noise_est = result
            input_specs.append(input_spec)
            target_specs.append(target_spec)
            noise_estimates.append(noise_est)
            successful += 1
    
    elapsed = time.time() - start_time
    
    print(f"✅ Успешно обработано {successful}/{len(file_tuples)} файлов")
    print(f"⏱️  Время обработки: {elapsed:.1f} секунд")
    
    if successful == 0:
        raise ValueError("Не удалось обработать ни одного файла!")
    
    print(f"\n📏 ФОРМА ДАННЫХ:")
    print(f"Вход (после спектрального вычитания): {input_specs[0].shape}")
    print(f"Цель (идеальный Wiener): {target_specs[0].shape}")
    
    return np.array(input_specs), np.array(target_specs), np.array(noise_estimates)

def create_context_windows_pipeline(data, context_frames=CONTEXT_FRAMES):
    """Создает окна с временным контекстом для каскадной модели."""
    num_samples, freq, time, channels = data.shape
    windows = []
    
    for i in range(num_samples):
        for t in range(time - context_frames + 1):
            window = data[i, :, t:t+context_frames, :]
            windows.append(window)
    
    return np.array(windows)

def create_center_frames_pipeline(data, context_frames=CONTEXT_FRAMES):
    """Создает центральные кадры для Y в каскадной модели."""
    num_samples, freq, time, channels = data.shape
    center_idx = context_frames // 2
    result = []
    
    for i in range(num_samples):
        for t in range(time - context_frames + 1):
            center_frame = data[i, :, t + center_idx, :]
            result.append(center_frame)
    
    return np.array(result)

def build_cascade_model(input_shape):
    """Создает нейросеть для каскадной обработки."""
    inputs = layers.Input(shape=input_shape)  # (freq, context, 2)
    
    # Conv2D для частотно-временной обработки
    x = layers.Conv2D(32, (5, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(64, (5, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    
    x = layers.Conv2D(128, (5, 1), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Residual block
    residual = x
    x = layers.Conv2D(128, (3, 1), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(128, (3, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(64, (3, 1), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Attention mechanism
    attention = layers.GlobalAveragePooling2D()(x)
    attention = layers.Dense(32, activation='relu')(attention)
    attention = layers.Dense(64, activation='sigmoid')(attention)
    attention = layers.Reshape((1, 1, 64))(attention)
    x = layers.Multiply()([x, attention])
    
    # Выходной слой
    x = layers.Conv2D(32, (3, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(2, (3, 1), padding='same', activation='tanh')(x)
    
    # Убираем временную ось
    output = tf.squeeze(x, axis=2)
    
    model = models.Model(inputs=inputs, outputs=output)
    return model

def complex_mae_loss(y_true, y_pred):
    """MAE loss для комплексных чисел (более устойчивый, чем MSE)."""
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def weighted_frequency_loss(y_true, y_pred):
    """Взвешенная функция потерь с большим весом на низких частотах."""
    # Создаем веса: больше вес на низких частотах (где больше энергии речи)
    freq_weights = 1.0 + tf.linspace(0.0, 1.0, FREQ_BINS)
    freq_weights = tf.reshape(freq_weights, [1, FREQ_BINS, 1])
    
    error = tf.abs(y_true - y_pred)
    weighted_error = error * freq_weights
    
    return tf.reduce_mean(weighted_error)

# === основной код ===
if __name__ == "__main__":
    # Настройка окружения
    TEMP_DIR = setup_colab_environment()
    DRIVE_PROJECT_PATH = "/content/drive/MyDrive/diplom-project"
    
    # 1. Копируем данные
    clean_dir, noise_dir = copy_data_to_tmp(DRIVE_PROJECT_PATH, TEMP_DIR)
    
    if clean_dir is None or noise_dir is None:
        print("❌ Не удалось найти данные. Выход.")
        exit(1)
    
    # 2. Собираем список файлов
    clean_files = glob.glob(os.path.join(clean_dir, "*.wav"))
    print(f"\n📊 Найдено {len(clean_files)} чистых файлов")
    
    file_tuples = []
    for clean_file in clean_files:
        noisy_file = os.path.join(noise_dir, os.path.basename(clean_file))
        if os.path.exists(noisy_file):
            file_tuples.append((clean_file, noisy_file))
        else:
            print(f"⚠️ Шумный файл не найден: {noisy_file}")
    
    print(f"✅ Создано {len(file_tuples)} пар файлов")
    
    if len(file_tuples) == 0:
        print("❌ Нет пар файлов для обучения. Выход.")
        exit(1)
    
    # 3. Ограничиваем количество
    if len(file_tuples) > MAX_SAMPLES:
        print(f"🔧 Ограничиваем до {MAX_SAMPLES} пар для теста")
        file_tuples = file_tuples[:MAX_SAMPLES]
    
    # 4. Подготовка данных с каскадом
    print(f"\n🎯 ПОДГОТОВКА ДАННЫХ С КАСКАДОМ:")
    print(f"1. Спектральное вычитание (alpha={SPEC_SUBTRACTION_ALPHA}, beta={SPEC_SUBTRACTION_BETA})")
    print(f"2. Нейросеть (улучшение)")
    print(f"3. Фильтр Винера (идеальная цель)")
    
    X_input, Y_target, noise_estimates = prepare_data_pipeline_parallel(
        file_tuples,
        num_samples=len(file_tuples),
        target_time_frames=TARGET_TIME_FRAMES
    )
    
    # 5. Разделение данных
    val_split = 0.15
    split_idx = int(len(X_input) * (1 - val_split))
    
    X_train, X_val = X_input[:split_idx], X_input[split_idx:]
    Y_train, Y_val = Y_target[:split_idx], Y_target[split_idx:]
    noise_train, noise_val = noise_estimates[:split_idx], noise_estimates[split_idx:]
    
    print(f"\n📊 РАЗМЕРНОСТИ ДАННЫХ:")
    print(f"X_train: {X_train.shape} (вход после спектрального вычитания)")
    print(f"Y_train: {Y_train.shape} (цель - идеальный Wiener)")
    print(f"X_val: {X_val.shape}")
    print(f"Y_val: {Y_val.shape}")
    
    # 6. Создание контекстных окон
    print(f"\n🔄 СОЗДАНИЕ ОКОН С КОНТЕКСТОМ ({CONTEXT_FRAMES} кадров)...")
    
    X_train_context = create_context_windows_pipeline(X_train, CONTEXT_FRAMES)
    X_val_context = create_context_windows_pipeline(X_val, CONTEXT_FRAMES)
    Y_train_center = create_center_frames_pipeline(Y_train, CONTEXT_FRAMES)
    Y_val_center = create_center_frames_pipeline(Y_val, CONTEXT_FRAMES)
    
    print(f"\n✅ ДАННЫЕ ПОДГОТОВЛЕНЫ:")
    print(f"X_train_context: {X_train_context.shape}")
    print(f"Y_train_center:  {Y_train_center.shape}")
    print(f"X_val_context:   {X_val_context.shape}")
    print(f"Y_val_center:    {Y_val_center.shape}")
    
    # 7. Создание модели
    print("\n🤖 СОЗДАНИЕ КАСКАДНОЙ МОДЕЛИ")
    print("Архитектура: Conv2D → Residual → Attention → Output")
    
    input_shape = (FREQ_BINS, CONTEXT_FRAMES, 2)
    model = build_cascade_model(input_shape)
    
    # 8. Компиляция модели
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=weighted_frequency_loss,  # Используем взвешенную loss
        metrics=['mae', 'mse']
    )
    
    model.summary()
    
    # 9. Callbacks
    model_dir = os.path.join(TEMP_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(model_dir, "best_cascade_model.weights.h5"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(model_dir, "cascade_model_epoch_{epoch:02d}.h5"),
            save_freq='epoch',
            save_weights_only=True
        )
    ]
    
    # 10. Обучение
    print("\n🚀 НАЧАЛО ОБУЧЕНИЯ КАСКАДНОЙ МОДЕЛИ")
    print("="*60)
    
    history = model.fit(
        X_train_context, Y_train_center,
        batch_size=min(BATCH_SIZE, len(X_train_context)),
        epochs=EPOCHS,
        validation_data=(X_val_context, Y_val_center),
        shuffle=True,
        verbose=1,
        callbacks=callbacks
    )
    
    # 11. Сохранение модели
    final_model_path = os.path.join(model_dir, "final_cascade_model.h5")
    model.save(final_model_path)
    print(f"✅ Модель сохранена: {final_model_path}")
    
    # 12. Тестирование на примере
    print("\n🧪 ТЕСТИРОВАНИЕ НА ПРИМЕРЕ")
    test_idx = np.random.randint(0, len(X_val_context))
    test_input = X_val_context[test_idx:test_idx+1]
    test_target = Y_val_center[test_idx:test_idx+1]
    
    prediction = model.predict(test_input, verbose=0)
    
    mae = np.mean(np.abs(prediction - test_target))
    print(f"MAE на тестовом примере: {mae:.4f}")
    
    # 13. Визуализация результатов (опционально)
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 2, 1)
        plt.imshow(test_input[0, :, :, 0], aspect='auto', origin='lower')
        plt.title("Вход (real)")
        plt.colorbar()
        
        plt.subplot(3, 2, 2)
        plt.imshow(test_input[0, :, :, 1], aspect='auto', origin='lower')
        plt.title("Вход (imag)")
        plt.colorbar()
        
        plt.subplot(3, 2, 3)
        plt.imshow(test_target[0, :, 0], aspect='auto', origin='lower')
        plt.title("Цель (real)")
        plt.colorbar()
        
        plt.subplot(3, 2, 4)
        plt.imshow(test_target[0, :, 1], aspect='auto', origin='lower')
        plt.title("Цель (imag)")
        plt.colorbar()
        
        plt.subplot(3, 2, 5)
        plt.imshow(prediction[0, :, 0], aspect='auto', origin='lower')
        plt.title("Предсказание (real)")
        plt.colorbar()
        
        plt.subplot(3, 2, 6)
        plt.imshow(prediction[0, :, 1], aspect='auto', origin='lower')
        plt.title("Предсказание (imag)")
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "cascade_model_test.png"))
        print(f"✅ Визуализация сохранена")
    except:
        print("⚠️  Не удалось создать визуализацию")
    
    # 14. Копирование результатов
    print("\n💾 КОПИРОВАНИЕ РЕЗУЛЬТАТОВ НА GOOGLE ДИСК...")
    drive_models_dir = os.path.join(DRIVE_PROJECT_PATH, "models")
    os.makedirs(drive_models_dir, exist_ok=True)
    
    for file in glob.glob(os.path.join(model_dir, "*")):
        shutil.copy2(file, drive_models_dir)
    
    print(f"✅ Результаты сохранены в: {drive_models_dir}")
    
    # 15. Сохранение истории обучения
    history_path = os.path.join(model_dir, "training_history.npy")
    np.save(history_path, history.history)
    
    print("\n" + "="*60)
    print("🎉 ОБУЧЕНИЕ КАСКАДНОЙ МОДЕЛИ УСПЕШНО ЗАВЕРШЕНО!")
    print(f"Архитектура: Спектральное вычитание → Нейросеть → Фильтр Винера")
    print(f"Контекст: {CONTEXT_FRAMES} кадров")
    print(f"Параметры спектрального вычитания: alpha={SPEC_SUBTRACTION_ALPHA}, beta={SPEC_SUBTRACTION_BETA}")
    print(f"Окончательная точность:")
    print(f"  Train Loss: {history.history['loss'][-1]:.4f}")
    print(f"  Val Loss:   {history.history['val_loss'][-1]:.4f}")
    print("="*60)
    
    # 16. Инструкции по использованию
    print("\n📋 ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ:")
    print("1. Загрузите веса модели: model.load_weights('best_cascade_model.weights.h5')")
    print("2. Для обработки аудио:")
    print("   - Примените спектральное вычитание к шумному сигналу")
    print("   - Подайте результат в нейросеть")
    print("   - (Опционально) Примените фильтр Винера к выходу нейросети")
    print("\nФормат входа для модели: (batch, freq_bins, context_frames, 2)")