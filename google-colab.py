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
import shutil
import time
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# === параметры ===
SR = 16000
N_FFT = 512
HOP_LENGTH = 256
EPOCHS = 100
BATCH_SIZE = 8
TARGET_TIME_FRAMES = 256
FREQ_BINS = N_FFT // 2 + 1  # 257
# MAX_SAMPLES = 100
MAX_SAMPLES = 1000
CONTEXT_FRAMES = 5

def setup_colab_environment():
    """Настройка окружения для Google Colab"""
    print("="*60)
    print("НАСТРОЙКА GOOGLE COLAB")
    print("="*60)
    
    # Проверка GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU доступен: {len(gpus)} устройств")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Memory growth включен")
        except Exception as e:
            print(f"⚠️ Ошибка GPU: {e}")
    else:
        print("❌ GPU не найден, используется CPU")
    
    # Создаем временную директорию для быстрого доступа
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
        # Копируем только если не скопировали ранее
        clean_src = os.path.join(drive_data_path, "clean_speech")
        noise_src = os.path.join(drive_data_path, "noise")
        
        clean_dst = os.path.join(temp_dir, "data/clean_speech")
        noise_dst = os.path.join(temp_dir, "data/noise")
        
        # Копируем только первые N файлов для теста (можно увеличить)
        max_files = 10000
        
        # Функция для копирования файлов с прогресс-баром
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
        
        # Копируем файлы
        clean_count = copy_files(clean_src, clean_dst, "wav")
        noise_count = copy_files(noise_src, noise_dst, "wav")
        
        print(f"✅ Скопировано {clean_count} чистых и {noise_count} шумных файлов")
        return clean_dst, noise_dst
    else:
        print(f"⚠️ Папка {drive_data_path} не найдена!")
        return None, None

def process_single_file(args):
    """Обработка одного аудиофайла с точным контролем размера"""
    clean_path, noisy_path, target_frames = args
    
    try:
        # 1. РАССЧИТЫВАЕМ точное количество секунд для нужного числа кадров
        target_samples = target_frames * HOP_LENGTH  # 256 * 256 = 65536 samples
        target_seconds = target_samples / SR         # 65536 / 16000 = 4.096 секунд
        
        # 2. ЗАГРУЖАЕМ с ПРАВИЛЬНОЙ длительностью
        clean_audio = librosa.load(
            clean_path, 
            sr=SR, 
            duration=target_seconds  # Загружаем РОВНО столько, сколько нужно
        )[0]
        
        noisy_audio = librosa.load(
            noisy_path,
            sr=SR,
            duration=target_seconds
        )[0]
        
        # 3. ТОЧНОЕ ВЫРАВНИВАНИЕ ДЛИНЫ
        # Если аудио короче нужного - дополняем тишиной
        if len(clean_audio) < target_samples:
            pad_len = target_samples - len(clean_audio)
            clean_audio = np.pad(clean_audio, (0, pad_len), mode='constant')
        
        # Если длиннее - обрезаем
        clean_audio = clean_audio[:target_samples]
        
        # То же самое для шумного
        if len(noisy_audio) < target_samples:
            pad_len = target_samples - len(noisy_audio)
            noisy_audio = np.pad(noisy_audio, (0, pad_len), mode='constant')
        
        noisy_audio = noisy_audio[:target_samples]
        
        # 4. STFT с ФИКСИРОВАННЫМИ ПАРАМЕТРАМИ
        clean_spec = librosa.stft(
            clean_audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=N_FFT,  # Важно для стабильности!
            window='hann',
            center=False  # Убираем center для точного контроля кадров
        )
        
        noisy_spec = librosa.stft(
            noisy_audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=N_FFT,
            window='hann',
            center=False
        )
        
        # 5. ПРОВЕРКА РАЗМЕРА (должно быть точно target_frames)
        # При center=False: кадров = ⌈(len(audio) - n_fft) / hop_length⌉ + 1
        # При наших параметрах должно быть 256 кадров точно
        
        if clean_spec.shape[1] != target_frames:
            # Логируем, но корректируем
            print(f"⚠️ {os.path.basename(clean_path)}: "
                  f"ожидали {target_frames}, получили {clean_spec.shape[1]} кадров")
            
            if clean_spec.shape[1] > target_frames:
                clean_spec = clean_spec[:, :target_frames]
                noisy_spec = noisy_spec[:, :target_frames]
            else:
                pad_width = target_frames - clean_spec.shape[1]
                clean_spec = np.pad(clean_spec, ((0, 0), (0, pad_width)), 
                                   mode='constant', constant_values=0)
                noisy_spec = np.pad(noisy_spec, ((0, 0), (0, pad_width)), 
                                   mode='constant', constant_values=0)
        
        # 6. ПРЕОБРАЗОВАНИЕ В 2 КАНАЛА (real, imag)
        clean_spec_2ch = np.stack([np.real(clean_spec), np.imag(clean_spec)], axis=-1)
        noisy_spec_2ch = np.stack([np.real(noisy_spec), np.imag(noisy_spec)], axis=-1)
        
        # 7. ВЕРИФИКАЦИЯ
        filename = os.path.basename(clean_path)
        if clean_spec_2ch.shape[1] != target_frames:  # Проверка времени
            print(f"❌ {filename}: финальная форма {clean_spec_2ch.shape} "
                  f"(кадров: {clean_spec_2ch.shape[1]}, ожидали {target_frames})")
            return None
        
        # Проверка на NaN/Inf
        if np.any(np.isnan(clean_spec_2ch)) or np.any(np.isinf(clean_spec_2ch)):
            print(f"❌ {filename}: обнаружены NaN/Inf значения")
            return None
        
        # Проверка на полную тишину (возможно поврежденный файл)
        if np.max(np.abs(clean_spec_2ch)) < 1e-6:
            print(f"⚠️ {filename}: возможно тихий или поврежденный файл")
            # Можно либо пропустить, либо оставить
        
        return noisy_spec_2ch, clean_spec_2ch
        
    except Exception as e:
        print(f"❌ Ошибка в файле {os.path.basename(clean_path)}: {str(e)[:100]}")
        return None

def prepare_data_parallel(file_tuples, num_samples=1000, target_time_frames=256):
    """Параллельная подготовка данных с использованием всех ядер CPU"""
    print(f"\n⚡ ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА {min(num_samples, len(file_tuples))} ФАЙЛОВ")
    
    start_time = time.time()
    
    # Ограничиваем количество файлов
    file_tuples = file_tuples[:min(num_samples, len(file_tuples))]
    
    # Подготавливаем аргументы для параллельной обработки
    args_list = [(clean, noisy, target_time_frames) for clean, noisy in file_tuples]
    
    # Используем ThreadPoolExecutor для параллельной обработки
    noisy_specs = []
    clean_specs = []
    successful = 0
    
    print(f"Используется {multiprocessing.cpu_count()} ядер CPU")
    
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Маппинг всех файлов
        results = list(executor.map(process_single_file, args_list))
    
    # Собираем результаты
    for result in results:
        if result is not None:
            noisy, clean = result
            noisy_specs.append(noisy)
            clean_specs.append(clean)
            successful += 1
    
    elapsed = time.time() - start_time
    
    print(f"✅ Успешно обработано {successful}/{len(file_tuples)} файлов")
    print(f"⏱️  Время обработки: {elapsed:.1f} секунд")
    print(f"📊 Скорость: {successful/elapsed:.1f} файлов/сек")
    
    if successful == 0:
        raise ValueError("Не удалось обработать ни одного файла!")
    
    # Проверяем форму данных
    print(f"\n📏 ФОРМА ДАННЫХ:")
    print(f"noisy_specs[0] shape: {noisy_specs[0].shape if noisy_specs else 'нет данных'}")
    print(f"clean_specs[0] shape: {clean_specs[0].shape if clean_specs else 'нет данных'}")
    
    return np.array(noisy_specs), np.array(clean_specs)

def build_better_hybrid_model(input_shape, context_frames=5):
    """Улучшенная версия с лучшими skip connections"""
    inputs = layers.Input(shape=input_shape)  # (freq, time, channels)
    
    # Сохраняем центральный кадр входа
    input_center = layers.Lambda(lambda x: x[:, :, context_frames//2, :])(inputs)
    
    # ========== ENCODER ==========
    # Первый блок
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    skip1 = x
    
    # Второй блок
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    skip2 = x
    
    # Третий блок
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # ========== BOTTLENECK ==========
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # ========== DECODER ==========
    # Третий блок декодера + skip connection
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, skip2])  # Skip connection 2
    
    # Второй блок декодера + skip connection
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, skip1])  # Skip connection 1
    
    # ========== OUTPUT ==========
    # Берем центральный кадр
    x = layers.Lambda(lambda x: x[:, :, context_frames//2, :])(x)
    
    # Финальный слой
    x = layers.Conv1D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    outputs = layers.Conv1D(2, 3, padding='same')(x)
    
    # ГЛОБАЛЬНЫЙ SKIP: выход = вход + изменения
    outputs = layers.Add()([input_center, outputs])
    outputs = layers.Activation('tanh')(outputs)
    
    model = models.Model(inputs, outputs)
    
    print(f"✅ Улучшенная hybrid модель создана")
    model.summary()
    
    return model

def complex_mse_loss(y_true, y_pred):
    """MSE loss для комплексных чисел"""
    return tf.reduce_mean(tf.square(y_true - y_pred))

def create_context_windows(data, context_frames=CONTEXT_FRAMES):
        """Создает окна с временным контекстом."""
        num_samples, freq, time, channels = data.shape
        
        # Создаем скользящие окна по временной оси
        windows = []
        for i in range(num_samples):
            sample_windows = []
            for t in range(time - context_frames + 1):
                # Берем context_frames последовательных кадров
                window = data[i, :, t:t+context_frames, :]
                sample_windows.append(window)
            windows.extend(sample_windows)
        
        return np.array(windows)

def create_center_frames(data, context_frames=CONTEXT_FRAMES):
    """Создает центральные кадры для Y."""
    num_samples, freq, time, channels = data.shape
    center_idx = context_frames // 2
    result = []
    for i in range(num_samples):
        for t in range(time - context_frames + 1):
            center_frame = data[i, :, t + center_idx, :]
            result.append(center_frame)
    return np.array(result)

# === основной код ===
if __name__ == "__main__":
    # Настройка окружения Colab
    TEMP_DIR = setup_colab_environment()
    
    # Путь к проекту на Google Диске (измените на свой)
    DRIVE_PROJECT_PATH = "/content/drive/MyDrive/diplom-project"
    
    # 1. Копируем данные на локальный SSD
    clean_dir, noise_dir = copy_data_to_tmp(DRIVE_PROJECT_PATH, TEMP_DIR)
    
    if clean_dir is None or noise_dir is None:
        print("❌ Не удалось найти данные. Выход.")
        exit(1)
    
    # 2. Собираем список файлов
    clean_files = glob.glob(os.path.join(clean_dir, "*.wav"))
    print(f"\n📊 Найдено {len(clean_files)} чистых файлов")
    
    # Создаем пары файлов (чистый, шумный)
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
        
    # 4. Ограничиваем количество для быстрого теста
    if len(file_tuples) > MAX_SAMPLES:
        print(f"🔧 Ограничиваем до {MAX_SAMPLES} пар для быстрого теста")
        file_tuples = file_tuples[:MAX_SAMPLES]
    
    # 5. Параллельная подготовка данных
    print(f"\n🎯 ПОДГОТОВКА ДАННЫХ")
    X_all, Y_all = prepare_data_parallel(
        file_tuples,
        num_samples=len(file_tuples),
        target_time_frames=TARGET_TIME_FRAMES
    )
    val_split = 0.15
    split_idx = int(len(X_all) * (1 - val_split))
    
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    Y_train, Y_val = Y_all[:split_idx], Y_all[split_idx:]

    X_train_context = create_context_windows(X_train, CONTEXT_FRAMES)
    X_val_context = create_context_windows(X_val, CONTEXT_FRAMES)
    Y_train_center = create_center_frames(Y_train, CONTEXT_FRAMES)
    Y_val_center = create_center_frames(Y_val, CONTEXT_FRAMES)
    print(f"\n✅ ДАННЫЕ ПЕРЕФОРМАТИРОВАНЫ:")
    print(f"X_train_context: {X_train_context.shape} (окна х freq х context х channels)")
    print(f"Y_train_center:  {Y_train_center.shape} (окна х freq х channels)")
    print(f"X_val_context:   {X_val_context.shape}")
    print(f"Y_val_center:    {Y_val_center.shape}")
    
    # Проверяем соответствие размеров
    if len(X_train_context) != len(Y_train_center):
        print(f"❌ ОШИБКА: Несоответствие размеров!")
        print(f"X_train_context: {len(X_train_context)} окон")
        print(f"Y_train_center:  {len(Y_train_center)} окон")
        exit(1)
        
    input_shape = (FREQ_BINS, CONTEXT_FRAMES, 2)  
    # 9. Создаем модель
    print("\n🤖 СОЗДАНИЕ МОДЕЛИ")
    model = build_better_hybrid_model(input_shape)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=complex_mse_loss,
        metrics=['mae']
    )
    
    model.summary()
    
    # 11. Callbacks
    model_dir = os.path.join(TEMP_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
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
            os.path.join(model_dir, "best_model.weights.h5"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]
    
    # 12. Обучаем модель (только тестовые 2 эпохи)
    print("\n🚀 НАЧАЛО ОБУЧЕНИЯ")
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
    
    # 13. Сохраняем модель
    final_model_path = os.path.join(model_dir, "final_model.h5")
    model.save(final_model_path)
    print(f"✅ Модель сохранена: {final_model_path}")
    
    # 14. Копируем результаты обратно на Google Диск
    print("\n💾 КОПИРОВАНИЕ РЕЗУЛЬТАТОВ НА GOOGLE ДИСК...")
    drive_models_dir = os.path.join(DRIVE_PROJECT_PATH, "models")
    os.makedirs(drive_models_dir, exist_ok=True)
    
    # Копируем все модели и логи
    for file in glob.glob(os.path.join(model_dir, "*")):
        shutil.copy2(file, drive_models_dir)
    
    print(f"✅ Результаты сохранены в: {drive_models_dir}")
    
    print("\n" + "="*60)
    print("🎉 ТЕСТОВОЕ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
    print(f"Если всё работает, увеличьте:")
    print(f"  - MAX_SAMPLES до 1000-2000")
    print(f"  - TEST_EPOCHS до {EPOCHS}")
    print("="*60)