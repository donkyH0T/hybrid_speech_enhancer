import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

# === ПАРАМЕТРЫ (ДОЛЖНЫ СОВПАДАТЬ С ОБУЧЕНИЕМ!) ===
SR = 16000
N_FFT = 512
HOP_LENGTH = 256
CONTEXT_FRAMES = 5  # ДОЛЖНО БЫТЬ ТАК ЖЕ КАК ПРИ ОБУЧЕНИИ!
FREQ_BINS = N_FFT // 2 + 1  # 257

def load_and_preprocess_audio(file_path, target_samples=None):
    """Загружает и нормализует аудио."""
    audio, _ = librosa.load(file_path, sr=SR)
    
    # Нормализация
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # Обрезаем/дополняем если нужно
    if target_samples:
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
        else:
            audio = audio[:target_samples]
    
    return audio

def create_context_windows(spec, context_frames):
    """Создает окна с временным контекстом для одной спектрограммы."""
    # spec shape: (freq, time, channels)
    time_frames = spec.shape[1]
    windows = []
    
    for t in range(time_frames - context_frames + 1):
        window = spec[:, t:t+context_frames, :]
        windows.append(window)
    
    return np.array(windows)  # (num_windows, freq, context_frames, channels)

def reconstruct_from_windows(windows, original_time_frames):
    """Восстанавливает полную спектрограмму из окон (берет центральные кадры)."""
    context_frames = windows.shape[2]
    center_idx = context_frames // 2
    freq_bins = windows.shape[1]
    
    # Инициализируем пустую спектрограмму
    reconstructed = np.zeros((freq_bins, original_time_frames, 2))
    weight_matrix = np.zeros((freq_bins, original_time_frames))
    
    # Складываем центральные кадры
    for i, window in enumerate(windows):
        center_frame = window[:, center_idx, :]
        reconstructed[:, i, :] += center_frame
        weight_matrix[:, i] += 1
    
    # Нормализуем (делим на количество наложений)
    weight_matrix[weight_matrix == 0] = 1  # избегаем деления на 0
    reconstructed = reconstructed / weight_matrix[:, :, np.newaxis]
    
    return reconstructed

def test_hybrid_model(model, clean_path, noisy_path, output_dir="hybrid_test"):
    """Тестирует гибридную модель с временным контекстом."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import soundfile as sf
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🧪 Тестируем гибридную модель:")
    print(f"Чистый: {os.path.basename(clean_path)}")
    print(f"Шумный: {os.path.basename(noisy_path)}")
    print(f"CONTEXT_FRAMES: {CONTEXT_FRAMES}")
    print("-" * 50)
    
    try:
        # 1. Загружаем аудио
        print("1. Загружаем аудио...")
        clean_audio = load_and_preprocess_audio(clean_path)
        noisy_audio = load_and_preprocess_audio(noisy_path)
        
        # Выравниваем длину
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        
        print(f"   Длина: {min_len} отсчетов ({min_len/SR:.2f} сек)")
        
        # 2. Преобразуем в спектрограммы
        print("\n2. Преобразуем в спектрограммы...")
        
        def audio_to_spec(audio):
            spec = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
            real = np.real(spec)
            imag = np.imag(spec)
            return np.stack([real, imag], axis=-1)  # (freq, time, 2)
        
        clean_spec = audio_to_spec(clean_audio)
        noisy_spec = audio_to_spec(noisy_audio)
        
        # ОБЯЗАТЕЛЬНО: проверяем, что размеры совпадают
        if clean_spec.shape[1] != noisy_spec.shape[1]:
            print(f"⚠️ Разная длина спектрограмм!")
            print(f"   Clean: {clean_spec.shape[1]} кадров")
            print(f"   Noisy: {noisy_spec.shape[1]} кадров")
            # Берем минимальную длину
            min_frames = min(clean_spec.shape[1], noisy_spec.shape[1])
            clean_spec = clean_spec[:, :min_frames, :]
            noisy_spec = noisy_spec[:, :min_frames, :]
        
        print(f"   Форма спектрограммы: {noisy_spec.shape}")
        
        # 3. Создаем контекстные окна
        print("\n3. Создаем контекстные окна...")
        
        # Важно: нужно чтобы было достаточно кадров для контекста
        if noisy_spec.shape[1] < CONTEXT_FRAMES:
            print(f"❌ ОШИБКА: слишком мало кадров для контекста!")
            print(f"   Есть: {noisy_spec.shape[1]}, нужно минимум: {CONTEXT_FRAMES}")
            return None
            
        noisy_windows = create_context_windows(noisy_spec, CONTEXT_FRAMES)
        
        print(f"   Создано {len(noisy_windows)} окон")
        print(f"   Форма окна: {noisy_windows[0].shape} (должна быть {model.input_shape[1:]})")
        
        # 4. Обработка моделью
        print("\n4. Обработка моделью...")
        
        # Обрабатываем батчами для экономии памяти
        batch_size = 32
        processed_windows = []
        
        for i in range(0, len(noisy_windows), batch_size):
            batch = noisy_windows[i:i+batch_size]
            predictions = model.predict(batch, verbose=0)
            processed_windows.append(predictions)
            
            if i == 0:
                print(f"   Первый батч обработан")
                print(f"   Вход: {batch.shape}, Выход: {predictions.shape}")
        
        # Объединяем результаты
        denoised_center_frames = np.concatenate(processed_windows, axis=0)
        print(f"   Обработано {len(denoised_center_frames)} центральных кадров")
        
        # 5. Восстанавливаем полную спектрограмму
        print("\n5. Восстанавливаем полную спектрограмму...")
        
        # КАК МНОГО КАДРОВ МЫ ДОЛЖНЫ ПОЛУЧИТЬ?
        # Если было T кадров и CONTEXT_FRAMES = 5,
        # то после create_context_windows будет T-4 окон
        # И после reconstruct_from_windows будет T-4 кадров
        
        # Правильное количество кадров на выходе:
        expected_frames = noisy_spec.shape[1] - CONTEXT_FRAMES + 1
        
        denoised_windows = []
        for center_frame in denoised_center_frames:
            window = np.zeros((FREQ_BINS, CONTEXT_FRAMES, 2))
            window[:, CONTEXT_FRAMES//2, :] = center_frame
            denoised_windows.append(window)
        
        denoised_windows = np.array(denoised_windows)
        denoised_spec = reconstruct_from_windows(denoised_windows, expected_frames)
        
        print(f"   Восстановленная спектрограмма: {denoised_spec.shape}")
        print(f"   Оригинальная спектрограмма: {noisy_spec.shape}")
        
        # 6. Обратно в аудио
        print("\n6. Конвертируем обратно в аудио...")
        
        def spec_to_audio(spec):
            real = spec[:, :, 0]
            imag = spec[:, :, 1]
            return librosa.istft(real + 1j * imag, hop_length=HOP_LENGTH, length=min_len)
        
        # Используем length=min_len для точного контроля длины
        denoised_audio = spec_to_audio(denoised_spec)
        
        # Теперь clean_audio и denoised_audio должны быть одинаковой длины
        # Но если нет - обрежем до минимальной
        final_len = min(len(clean_audio), len(denoised_audio))
        clean_audio = clean_audio[:final_len]
        noisy_audio = noisy_audio[:final_len]
        denoised_audio = denoised_audio[:final_len]
        
        print(f"   Итоговая длина:")
        print(f"   Clean: {len(clean_audio)}")
        print(f"   Noisy: {len(noisy_audio)}")
        print(f"   Denoised: {len(denoised_audio)}")
        
        # 7. ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА РАЗМЕРОВ
        if len(clean_audio) != len(denoised_audio):
            print(f"⚠️ Предупреждение: разные длины!")
            print(f"   Clean: {len(clean_audio)}")
            print(f"   Denoised: {len(denoised_audio)}")
            # Делаем их одинаковыми
            min_len_final = min(len(clean_audio), len(denoised_audio))
            clean_audio = clean_audio[:min_len_final]
            noisy_audio = noisy_audio[:min_len_final]
            denoised_audio = denoised_audio[:min_len_final]
        
        # 8. Сохраняем результаты
        print("\n7. Сохраняем результаты...")
        base_name = os.path.splitext(os.path.basename(clean_path))[0]
        
        sf.write(os.path.join(output_dir, f"{base_name}_clean.wav"), clean_audio, SR)
        sf.write(os.path.join(output_dir, f"{base_name}_noisy.wav"), noisy_audio, SR)
        sf.write(os.path.join(output_dir, f"{base_name}_denoised.wav"), denoised_audio, SR)
        
        # 9. Вычисляем метрики (ТОЛЬКО ПОСЛЕ ВЫРАВНИВАНИЯ!)
        print("\n8. Вычисляем метрики...")
        
        def calculate_snr(signal, noise):
            # Убедимся, что размеры совпадают
            if len(signal) != len(noise):
                min_len = min(len(signal), len(noise))
                signal = signal[:min_len]
                noise = noise[:min_len]
            
            signal_power = np.mean(signal ** 2)
            noise_power = np.mean(noise ** 2)
            if noise_power < 1e-10:
                return float('inf')
            return 10 * np.log10(signal_power / noise_power)
        
        # Проверка размеров перед вычислением
        print(f"   Проверка размеров для вычисления SNR:")
        print(f"   Clean: {len(clean_audio)}, Noisy: {len(noisy_audio)}")
        
        original_noise = noisy_audio - clean_audio
        residual_noise = denoised_audio - clean_audio
        
        print(f"   Original noise: {len(original_noise)}")
        print(f"   Residual noise: {len(residual_noise)}")
        
        original_snr = calculate_snr(clean_audio, original_noise)
        denoised_snr = calculate_snr(clean_audio, residual_noise)
        
        mse_original = np.mean(original_noise ** 2)
        mse_residual = np.mean(residual_noise ** 2)
        mse_reduction = (mse_original - mse_residual) / mse_original if mse_original > 0 else 0
        
        print(f"\n📊 РЕЗУЛЬТАТЫ:")
        print(f"   Исходный SNR: {original_snr:.2f} dB")
        print(f"   После очистки: {denoised_snr:.2f} dB")
        print(f"   Улучшение: {denoised_snr - original_snr:.2f} dB")
        print(f"   Уменьшение MSE: {mse_reduction*100:.1f}%")
        
        # 10. Визуализация
        print("\n9. Создаем визуализацию...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        time_clean = np.arange(len(clean_audio)) / SR
        time_noisy = np.arange(len(noisy_audio)) / SR
        time_denoised = np.arange(len(denoised_audio)) / SR
        
        # Волновые формы
        axes[0, 0].plot(time_clean, clean_audio, 'g', alpha=0.7, linewidth=0.5)
        axes[0, 0].set_title('Исходный чистый звук')
        axes[0, 0].set_xlabel('Время (с)')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(time_noisy, noisy_audio, 'r', alpha=0.7, linewidth=0.5)
        axes[0, 1].set_title(f'Зашумленный (SNR: {original_snr:.1f} dB)')
        axes[0, 1].set_xlabel('Время (с)')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(time_denoised, denoised_audio, 'b', alpha=0.7, linewidth=0.5)
        axes[1, 0].set_title(f'Очищенный (SNR: {denoised_snr:.1f} dB)')
        axes[1, 0].set_xlabel('Время (с)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Шум (убедимся что одинаковой длины)
        noise_time = np.arange(len(original_noise)) / SR
        axes[1, 1].plot(noise_time, original_noise, 'orange', alpha=0.5, linewidth=0.5, label='Исходный')
        axes[1, 1].plot(noise_time, residual_noise, 'purple', alpha=0.5, linewidth=0.5, label='Оставшийся')
        axes[1, 1].set_title('Сравнение шума')
        axes[1, 1].set_xlabel('Время (с)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(
            f'Гибридная модель: {os.path.basename(clean_path)}\n'
            f'Улучшение SNR: {denoised_snr - original_snr:.2f} dB',
            fontsize=14
        )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_results.png"), dpi=150)
        plt.close()
        
        print(f"\n✅ Тестирование завершено!")
        print(f"📁 Результаты в: {output_dir}/")
        
        return {
            'clean': clean_audio,
            'noisy': noisy_audio,
            'denoised': denoised_audio,
            'snr_improvement': denoised_snr - original_snr,
            'mse_reduction': mse_reduction
        }
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None

def build_better_hybrid_model(input_shape, context_frames=5):
    """Та же модель что и при обучении"""
    inputs = layers.Input(shape=input_shape)
    input_center = layers.Lambda(lambda x: x[:, :, context_frames//2, :])(inputs)
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    skip1 = x
    
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    skip2 = x
    
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Bottleneck
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Decoder
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, skip2])
    
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, skip1])
    
    # Output
    x = layers.Lambda(lambda x: x[:, :, context_frames//2, :])(x)
    x = layers.Conv1D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    outputs = layers.Conv1D(2, 3, padding='same')(x)
    outputs = layers.Add()([input_center, outputs])
    outputs = layers.Activation('tanh')(outputs)
    
    return models.Model(inputs, outputs)

if __name__ == "__main__":
    print("="*60)
    print("ТЕСТ ГИБРИДНОЙ МОДЕЛИ С КОНТЕКСТОМ")
    print("="*60)
    
    # 1. Создаем модель с ТОЙ ЖЕ архитектурой
    input_shape = (FREQ_BINS, CONTEXT_FRAMES, 2)
    print(f"Input shape: {input_shape}")
    
    model = build_better_hybrid_model(input_shape, context_frames=CONTEXT_FRAMES)
    
    # 2. Загружаем веса
    weights_path = "models/best_model.weights.h5"  # Убедитесь что путь правильный!
    print(f"\nЗагружаю веса из: {weights_path}")
    
    if not os.path.exists(weights_path):
        print(f"❌ Файл {weights_path} не найден!")
        print("Сначала обучите модель или укажите правильный путь")
        exit(1)
    
    try:
        model.load_weights(weights_path)
        print("✅ Веса успешно загружены")
    except Exception as e:
        print(f"❌ Ошибка загрузки весов: {e}")
        exit(1)
    
    # 3. Тестируем
    clean_path = "data/clean_speech/p226_007.wav"
    noisy_path = "data/noise/p226_007.wav"
    
    print(f"\nТестируем файлы:")
    print(f"Clean: {clean_path}")
    print(f"Noisy: {noisy_path}")
    
    if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
        print("❌ Файлы не найдены!")
        print("Проверьте пути к данным")
        exit(1)
    
    # 4. Запускаем тест
    results = test_hybrid_model(
        model,
        clean_path,
        clean_path,
        "hybrid_model_test"
    )
    
    if results:
        print("\n" + "="*60)
        print("🎉 ТЕСТ УСПЕШНО ЗАВЕРШЕН!")
        print("="*60)
        print(f"Улучшение SNR: {results['snr_improvement']:.2f} dB")
        print(f"Уменьшение MSE: {results['mse_reduction']*100:.1f}%")
        print("\n🎧 Для прослушивания:")
        print("1. Откройте папку 'hybrid_model_test/'")
        print("2. Воспроизведите .wav файлы")
        print("3. Откройте .png для графиков")