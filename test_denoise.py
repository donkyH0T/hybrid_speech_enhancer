import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
from collections import deque
import argparse

class DenoiseModel(nn.Module):
    def __init__(self, n_fft, context_frames=5, dropout_rate=0.2):
        super().__init__()
        self.freq_dim = n_fft // 2 + 1
        self.context_frames = context_frames
        
        # Encoder
        self.conv1 = nn.Conv2d(2, 32, (3, 3), padding='same')
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding='same')
        
        # Временная обработка
        self.temporal_conv1 = nn.Conv2d(64, 64, (1, 3), padding=(0, 1))
        self.temporal_conv2 = nn.Conv2d(64, 64, (1, 3), padding=(0, 1))
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(64, 32, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (1, 1)),
            nn.Sigmoid()
        )
        
        # Decoder
        self.conv3 = nn.Conv2d(64, 32, (3, 3), padding='same')
        self.conv4 = nn.Conv2d(32, 16, (3, 3), padding='same')
        self.out = nn.Conv2d(16, 2, (3, 3), padding='same')
        
        # Нормализация
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(16)
        
        # Dropout слои
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        # Encoder
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        
        # Temporal processing
        x = torch.relu(self.temporal_conv1(x))
        x = torch.relu(self.temporal_conv2(x))
        
        # Attention
        attn = self.attention(x)
        x = x * attn
        
        # Decoder
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.tanh(self.out(x))
        
        return torch.mean(x, dim=3)

class SimpleSpeechDenoiser:
    def __init__(self, model_path='best_model.pth', n_fft=512, hop_length=160, context_frames=5):
        # Параметры (должны совпадать с обучением)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.context_frames = context_frames
        
        # Устройство
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")
        
        # Создание модели с правильной архитектурой
        self.model = DenoiseModel(n_fft=n_fft, context_frames=context_frames).to(self.device)
        
        # Загрузка весов
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Загружена модель из checkpoint (эпоха {checkpoint.get('epoch', 'unknown')})")
            print(f"Лучшая валидационная ошибка: {checkpoint.get('best_val_loss', 'unknown')}")
        else:
            # Пытаемся загрузить как state_dict
            try:
                self.model.load_state_dict(checkpoint)
                print("Загружена модель (state_dict)")
            except Exception as e:
                print(f"Ошибка загрузки модели: {e}")
                print("Пытаемся загрузить с учетом возможных различий в архитектуре...")
                # Пробуем загрузить с игнорированием лишних параметров
                self.model.load_state_dict(checkpoint, strict=False)
        
        self.model.eval()
        print("Модель переведена в режим inference")
        
        # Буферы для контекста
        self.context_buffer_real = deque(maxlen=self.context_frames)
        self.context_buffer_imag = deque(maxlen=self.context_frames)
        
    def enhance_audio(self, input_file, output_file):
        """
        Основная функция для улучшения качества аудио
        """
        print(f"Обработка: {input_file}")
        
        # Загрузка аудио
        audio, sr = librosa.load(input_file, sr=None)
        print(f"  Частота дискретизации: {sr} Hz")
        print(f"  Длина аудио: {len(audio)/sr:.2f} секунд")
        
        # STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        print(f"  STFT размер: {stft.shape} (частоты × время)")
        
        # Очистка STFT
        print("  Обработка STFT...")
        enhanced_stft = self._enhance_stft(stft)
        
        # Обратное преобразование
        print("  Обратное STFT...")
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length, length=len(audio))
        
        # Нормализация амплитуды
        max_amp = np.max(np.abs(enhanced_audio))
        if max_amp > 0:
            enhanced_audio = enhanced_audio / max_amp * 0.9
        
        # Сохранение
        sf.write(output_file, enhanced_audio, sr)
        print(f"Сохранено: {output_file}")
        print("-" * 50)
        
        return enhanced_audio
    
    def _enhance_stft(self, stft):
        """
        Улучшение STFT с помощью нейросети
        """
        # Разделение на реальную и мнимую части
        real = np.real(stft)
        imag = np.imag(stft)
        
        # Вычисляем коэффициент нормализации
        # Используем глобальную нормализацию для всего файла
        max_val = np.max([np.max(np.abs(real)), np.max(np.abs(imag))]) + 1e-10
        real = real / max_val
        imag = imag / max_val
        
        # Обработка кадров
        result_real = []
        result_imag = []
        
        # Очистка буферов
        self.context_buffer_real.clear()
        self.context_buffer_imag.clear()
        
        num_frames = stft.shape[1]
        
        # Прогресс-бар
        from tqdm import tqdm
        
        for frame_idx in tqdm(range(num_frames), desc="Обработка кадров", unit="кадр"):
            # Добавляем в буфер
            self.context_buffer_real.append(real[:, frame_idx])
            self.context_buffer_imag.append(imag[:, frame_idx])
            
            # Если буфер не заполнен - пропускаем (используем исходный)
            if len(self.context_buffer_real) < self.context_frames:
                result_real.append(real[:, frame_idx])
                result_imag.append(imag[:, frame_idx])
                continue
            
            # Подготовка данных для модели
            context_real = np.stack(list(self.context_buffer_real), axis=1)
            context_imag = np.stack(list(self.context_buffer_imag), axis=1)
            
            input_data = np.stack([context_real, context_imag], axis=0)  # [2, freq, context]
            input_tensor = torch.FloatTensor(input_data[np.newaxis, ...]).to(self.device)
            
            # Обработка
            with torch.no_grad():
                output = self.model(input_tensor)[0]  # [2, freq]
            
            # Восстановление и сохранение
            output_real = output[0].cpu().numpy() * max_val
            output_imag = output[1].cpu().numpy() * max_val
            
            # Сохраняем центральный кадр из контекста
            # (модель возвращает усредненный выход по времени)
            result_real.append(output_real)
            result_imag.append(output_imag)
        
        # Для первых кадров, где не было контекста, дублируем последний обработанный
        if len(result_real) < num_frames:
            last_real = result_real[-1] if result_real else real[:, 0]
            last_imag = result_imag[-1] if result_imag else imag[:, 0]
            while len(result_real) < num_frames:
                result_real.append(last_real)
                result_imag.append(last_imag)
        
        # Сборка результата
        enhanced_stft = np.stack(result_real, axis=1) + 1j * np.stack(result_imag, axis=1)
        
        return enhanced_stft
    
    def process_folder(self, input_folder, output_folder, extension='.wav'):
        """
        Обработка всех аудиофайлов в папке
        """
        import os
        os.makedirs(output_folder, exist_ok=True)
        
        audio_files = [f for f in os.listdir(input_folder) if f.endswith(extension)]
        print(f"Найдено {len(audio_files)} файлов для обработки")
        
        for audio_file in audio_files:
            input_path = os.path.join(input_folder, audio_file)
            output_path = os.path.join(output_folder, f"enhanced_{audio_file}")
            
            try:
                self.enhance_audio(input_path, output_path)
            except Exception as e:
                print(f"✗ Ошибка при обработке {audio_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Speech denoising with neural network')
    parser.add_argument('--input', type=str, required=True, help='Input audio file or folder')
    parser.add_argument('--output', type=str, required=True, help='Output audio file or folder')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to trained model')
    parser.add_argument('--n_fft', type=int, default=512, help='STFT window size')
    parser.add_argument('--hop_length', type=int, default=160, help='STFT hop length')
    parser.add_argument('--context_frames', type=int, default=5, help='Number of context frames')
    
    args = parser.parse_args()
    
    # Создаем денойзер
    denoiser = SimpleSpeechDenoiser(
        model_path=args.model,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        context_frames=args.context_frames
    )
    
    # Проверка, является ли вход папкой
    import os
    if os.path.isdir(args.input):
        denoiser.process_folder(args.input, args.output)
    else:
        denoiser.enhance_audio(args.input, args.output)

# Пример использования
if __name__ == "__main__":
    denoiser = SimpleSpeechDenoiser('best_model.pth')
    denoiser.enhance_audio('data/noisy/p226_001.wav', 'test/output.wav')