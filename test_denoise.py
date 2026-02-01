import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

N_FFT = 512
HOP_LENGTH = 256

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
    x = layers.ReLU()(x)
    
    outputs = layers.Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
    
    model = models.Model(inputs, outputs)
    print(f"✅ Fast модель с skip-connections создана")
    return model

def test_single_file_simple(model, clean_file_path, noisy_file_path, output_dir="single_test"):
    """
    Тестирует модель на одном файле (без сложной обрезки)
    Упрощенная версия для файлов одинаковой длины
    """
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    import soundfile as sf
    
    # Создаем директорию
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🧪 Тестируем на одном файле:")
    print(f"Чистый: {os.path.basename(clean_file_path)}")
    print(f"Шумный: {os.path.basename(noisy_file_path)}")
    print("-" * 50)
    
    try:
        # 1. Загружаем аудио
        print("1. Загружаем аудио файлы...")
        clean_audio = load_and_preprocess_audio(clean_file_path)
        noisy_audio = load_and_preprocess_audio(noisy_file_path)
        
        # Проверяем длину
        if len(clean_audio) != len(noisy_audio):
            print(f"⚠️ Внимание: файлы разной длины!")
            print(f"   Чистый: {len(clean_audio)} отсчетов")
            print(f"   Шумный: {len(noisy_audio)} отсчетов")
            # Но продолжим, просто обрежем до минимальной
            min_len = min(len(clean_audio), len(noisy_audio))
            clean_audio = clean_audio[:min_len]
            noisy_audio = noisy_audio[:min_len]
        
        print(f"   Аудио: {len(clean_audio)} отсчетов ({len(clean_audio)/16000:.2f} сек)")
        
        # 2. Обработка ВСЕГО аудио (по кускам если длинное)
        print("\n2. Обработка моделью...")
        
        # Разбиваем на сегменты по 4 секунды (как модель обучена)
        segment_duration = 4.0  # секунды
        segment_samples = int(segment_duration * 16000)  # 4 сек × 16000 Гц
        
        denoised_segments = []
        
        for start in range(0, len(clean_audio), segment_samples):
            end = start + segment_samples
            segment_noisy = noisy_audio[start:end]
            
            # Если сегмент слишком короткий, дополняем
            if len(segment_noisy) < segment_samples:
                padding = segment_samples - len(segment_noisy)
                segment_noisy = np.pad(segment_noisy, (0, padding), mode='constant')
            
            # Преобразуем в спектрограмму
            segment_spec = audio_to_spectrogram(segment_noisy)
            
            # Приводим к размеру модели (256 фреймов)
            target_frames = model.input_shape[2]
            current_frames = segment_spec.shape[1]
            
            if current_frames > target_frames:
                # Обрезаем по центру (редко, но на всякий случай)
                start_frame = (current_frames - target_frames) // 2
                segment_spec = segment_spec[:, start_frame:start_frame + target_frames, :]
            elif current_frames < target_frames:
                # Дополняем (тоже редко)
                padding = target_frames - current_frames
                segment_spec = np.pad(segment_spec, ((0, 0), (0, padding), (0, 0)), mode='constant')
            
            # Обработка моделью
            segment_denoised_spec = model.predict(
                np.expand_dims(segment_spec, 0), 
                verbose=0
            )[0]
            
            # Обратно в аудио
            def spec_to_audio(spec):
                real = spec[:, :, 0]
                imag = spec[:, :, 1]
                return librosa.istft(real + 1j * imag, hop_length=HOP_LENGTH)
            
            segment_denoised = spec_to_audio(segment_denoised_spec)
            
            # Обрезаем до исходной длины сегмента (без дополнения)
            segment_denoised = segment_denoised[:min(len(segment_noisy), len(segment_denoised))]
            denoised_segments.append(segment_denoised)
            
            if start == 0:
                print(f"   Обработан первый сегмент: {len(segment_denoised)/16000:.2f} сек")
        
        # Собираем все сегменты
        denoised_audio = np.concatenate(denoised_segments)
        
        # Обрезаем до длины оригинального аудио
        denoised_audio = denoised_audio[:len(clean_audio)]
        
        print(f"   Очищенное аудио: {len(denoised_audio)} отсчетов ({len(denoised_audio)/16000:.2f} сек)")
        
        # 3. Сохраняем аудио файлы
        print("\n3. Сохраняем аудио файлы...")
        sr = 16000
        base_name = os.path.splitext(os.path.basename(clean_file_path))[0]
        
        sf.write(os.path.join(output_dir, f"{base_name}_clean.wav"), clean_audio, sr)
        sf.write(os.path.join(output_dir, f"{base_name}_noisy.wav"), noisy_audio, sr)
        sf.write(os.path.join(output_dir, f"{base_name}_denoised.wav"), denoised_audio, sr)
        
        print(f"   ✅ Аудио сохранены в папке: {output_dir}/")
        
        # 4. Вычисляем метрики
        print("\n4. Вычисляем метрики качества...")
        
        def calculate_snr(signal, noise):
            signal_power = np.mean(signal ** 2)
            noise_power = np.mean(noise ** 2)
            if noise_power < 1e-10:  # избегаем деления на 0
                return float('inf')
            return 10 * np.log10(signal_power / noise_power)
        
        # Оригинальный шум
        original_noise = noisy_audio - clean_audio
        original_snr = calculate_snr(clean_audio, original_noise)
        
        # Оставшийся шум
        residual_noise = denoised_audio - clean_audio
        denoised_snr = calculate_snr(clean_audio, residual_noise)
        
        # MSE
        mse_original = np.mean(original_noise ** 2)
        mse_residual = np.mean(residual_noise ** 2)
        mse_reduction = (mse_original - mse_residual) / mse_original if mse_original > 0 else 0
        
        print(f"\n📊 РЕЗУЛЬТАТЫ:")
        print(f"   SNR исходный: {original_snr:.2f} dB")
        print(f"   SNR после очистки: {denoised_snr:.2f} dB")
        print(f"   Улучшение SNR: {denoised_snr - original_snr:.2f} dB")
        print(f"   Уменьшение MSE: {mse_reduction*100:.1f}%")
        
        # 5. Простая визуализация
        print("\n5. Создаем визуализацию...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        time = np.arange(len(clean_audio)) / sr
        
        # Волновые формы
        axes[0, 0].plot(time, clean_audio, 'g', alpha=0.7, linewidth=0.5, label='Чистый')
        axes[0, 0].set_title('Исходный чистый звук')
        axes[0, 0].set_xlabel('Время (с)')
        axes[0, 0].set_ylabel('Амплитуда')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(time, noisy_audio, 'r', alpha=0.7, linewidth=0.5, label='Зашумленный')
        axes[0, 1].set_title(f'Зашумленный звук (SNR: {original_snr:.1f} dB)')
        axes[0, 1].set_xlabel('Время (с)')
        axes[0, 1].set_ylabel('Амплитуда')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(time, denoised_audio, 'b', alpha=0.7, linewidth=0.5, label='Очищенный')
        axes[1, 0].set_title(f'Очищенный звук (SNR: {denoised_snr:.1f} dB)')
        axes[1, 0].set_xlabel('Время (с)')
        axes[1, 0].set_ylabel('Амплитуда')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Шум
        axes[1, 1].plot(time, original_noise, 'orange', alpha=0.5, linewidth=0.5, label='Исходный шум')
        axes[1, 1].plot(time, residual_noise, 'purple', alpha=0.5, linewidth=0.5, label='Оставшийся шум')
        axes[1, 1].set_title('Сравнение шума')
        axes[1, 1].set_xlabel('Время (с)')
        axes[1, 1].set_ylabel('Амплитуда')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(
            f'Результаты шумоподавления: {os.path.basename(clean_file_path)}\n'
            f'Улучшение SNR: {denoised_snr - original_snr:.2f} dB | '
            f'Уменьшение шума: {mse_reduction*100:.1f}%',
            fontsize=14
        )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_results.png"), dpi=150, bbox_inches='tight')
        # plt.show()
        
        print(f"\n✅ Тестирование завершено!")
        print(f"📁 Результаты в папке: {output_dir}/")
        print(f"🎧 Послушайте: {base_name}_denoised.wav")
        
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


if __name__ == "__main__":
    TARGET_TIME_FRAMES = 256
    FREQ_BINS = 257
    input_shape = (FREQ_BINS, TARGET_TIME_FRAMES, 2)
    
    print(f"Input shape: {input_shape}")
    model = build_fast_denoise_model(input_shape)
    print("Загружаю обученные веса...")
    model.load_weights("models/best_light_model.weights.h5")
    
    # Тестируем конкретный файл
    clean_path = "data/clean_speech/p226_001.wav"
    noisy_path = "data/noise/p226_001.wav"
    
    print("\n" + "="*60)
    print("ТЕСТ КОНКРЕТНОГО ФАЙЛА")
    print("="*60)
    
    # Вариант 1: Полный тест с визуализацией
    results = test_single_file_simple(
        model=model,
        clean_file_path=clean_path,
        noisy_file_path=noisy_path,
        output_dir="test_p226_001"
    )
    
    if results:
        print("\n🎧 Чтобы послушать результат:")
        print("1. Откройте папку test_p226_001/")
        print("2. Воспроизведите файлы:")
        print("   - p226_001_clean.wav - исходный чистый")
        print("   - p226_001_noisy.wav - зашумленный")
        print("   - p226_001_denoised.wav - очищенный моделью")
        print("3. Откройте p226_001_results.png для графиков")