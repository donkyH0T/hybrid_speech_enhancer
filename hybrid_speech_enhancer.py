import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
from collections import deque
import webrtcvad
from scipy.signal import medfilt

class OnlineHybridSpeechEnhancer:
    def __init__(self, sr, n_fft=512, hop_length=160, win_length=480,
                 context_frames=5, vad_aggressiveness=2, online_learning=False,
                 model_path='best_model.pth'):
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length
        self.window = np.hanning(self.win)
        self.context_frames = context_frames
        self.online_learning = online_learning

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Buffers
        self.prev_samples = np.zeros(self.win - self.hop, dtype=np.float32)
        self.stft_context = deque(maxlen=context_frames)
        self.istft_buffer = np.zeros(self.win - self.hop, dtype=np.float32)

        # Noise & VAD
        self.noise_estimate = np.zeros(n_fft // 2 + 1)
        self.noise_floor = 0.005
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.speech_history = deque(maxlen=9)
        self.speech_probability = 0.0
        self.consecutive_noise_frames = 0
        self.consecutive_speech_frames = 0

        # Online learning buffers
        self.clean_buffer = deque(maxlen=600)
        self.noisy_buffer = deque(maxlen=600)
        self.frame_counter = 0
        self.learning_interval = 25

        # Load model
        self.model = self._build_denoise_model().to(self.device)
        self.model.eval()
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("✅ Модель загружена")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить модель, используем случайные веса: {e}")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.95)

    def _build_denoise_model(self):
        class DenoiseModel(nn.Module):
            def __init__(self, n_fft, context_frames):
                super().__init__()
                self.freq_dim = n_fft // 2 + 1
                self.context_frames = context_frames

                self.conv1 = nn.Conv2d(2, 32, (3, 3), padding='same')
                self.conv2 = nn.Conv2d(32, 64, (3, 3), padding='same')
                self.temporal_conv1 = nn.Conv2d(64, 64, (1, 3), padding=(0, 1))
                self.temporal_conv2 = nn.Conv2d(64, 64, (1, 3), padding=(0, 1))

                self.attention = nn.Sequential(
                    nn.Conv2d(64, 32, (1, 1)),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, (1, 1)),
                    nn.Sigmoid()
                )

                self.conv3 = nn.Conv2d(64, 32, (3, 3), padding='same')
                self.conv4 = nn.Conv2d(32, 16, (3, 3), padding='same')
                self.out = nn.Conv2d(16, 2, (3, 3), padding='same')

                self.bn1 = nn.BatchNorm2d(32)
                self.bn2 = nn.BatchNorm2d(64)
                self.bn3 = nn.BatchNorm2d(32)
                self.bn4 = nn.BatchNorm2d(16)
                self.dropout1 = nn.Dropout2d(0.2)
                self.dropout2 = nn.Dropout2d(0.2)
                self.dropout3 = nn.Dropout2d(0.2)

            def forward(self, x):
                x = torch.relu(self.bn1(self.conv1(x)))
                x = self.dropout1(x)
                x = torch.relu(self.bn2(self.conv2(x)))
                x = self.dropout2(x)
                x = torch.relu(self.temporal_conv1(x))
                x = torch.relu(self.temporal_conv2(x))
                attn = self.attention(x)
                x = x * attn
                x = torch.relu(self.bn3(self.conv3(x)))
                x = self.dropout3(x)
                x = torch.relu(self.bn4(self.conv4(x)))
                x = torch.tanh(self.out(x))
                return torch.mean(x, dim=3)

        return DenoiseModel(n_fft=self.n_fft, context_frames=self.context_frames)

    def _multi_mode_vad(self, frame):
        pcm = (np.clip(frame[-480:], -1, 1) * 32767).astype(np.int16).tobytes()
        is_speech = self.vad.is_speech(pcm, self.sr)
        self.speech_history.append(is_speech)
        speech_ratio = sum(self.speech_history) / len(self.speech_history)
        self.speech_probability = 0.92 * self.speech_probability + 0.08 * speech_ratio
        return is_speech

    def _update_noise_estimate(self, stft, is_speech):
        mag = np.abs(stft)
        mag_mean = np.mean(mag, axis=-1)
        if np.all(self.noise_estimate == 0):
            self.noise_estimate = mag_mean
            return
        if not is_speech:
            self.consecutive_noise_frames += 1
            alpha = 0.9
            self.noise_estimate = alpha * self.noise_estimate + (1 - alpha) * mag_mean
        else:
            self.consecutive_speech_frames += 1
            alpha = 0.995
            self.noise_estimate = alpha * self.noise_estimate + (1 - alpha) * mag_mean

    def _neural_denoise(self, stft):
        real = np.real(stft)[:, -1]
        imag = np.imag(stft)[:, -1]
        self.stft_context.append(np.stack([real, imag], axis=0))
        if len(self.stft_context) < self.context_frames:
            return real + 1j * imag
        context = np.stack(self.stft_context, axis=-1)
        tensor = torch.from_numpy(context).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            out = self.model(tensor)[0]
        return out[0].cpu().numpy() + 1j * out[1].cpu().numpy()

    def _hybrid_denoise(self, stft, is_speech):
        self._update_noise_estimate(stft, is_speech)
        if is_speech:
            stft_clean = self._neural_denoise(stft)
            mag = np.abs(stft_clean)
            phase = np.angle(stft_clean)
            noise_floor = self.noise_floor if np.isscalar(self.noise_floor) else self.noise_floor[:, np.newaxis]
            mag_clean = np.where(mag > noise_floor * 2.5, mag, 0)
            return mag_clean * np.exp(1j * phase)
        else:
            mag = np.abs(stft)
            phase = np.angle(stft)
            mag_clean = np.maximum(self.noise_floor, mag - 2 * self.noise_estimate[:, np.newaxis])
            return mag_clean * np.exp(1j * phase)

    def process_chunk(self, chunk):
        full_frame = np.concatenate([self.prev_samples, chunk])
        self.prev_samples = full_frame[-(self.win - self.hop):]
        is_speech = self._multi_mode_vad(full_frame)

        stft = librosa.stft(full_frame, n_fft=self.n_fft, hop_length=self.hop,
                            win_length=self.win, window=self.window)
        stft_clean = self._hybrid_denoise(stft)

        audio_clean = librosa.istft(stft_clean, hop_length=self.hop, win_length=self.win,
                                    window=self.window, n_fft=self.n_fft)

        # Overlap-add с буфером
        output = audio_clean[-len(chunk):]
        if len(output) < len(chunk):
            output = np.pad(output, (0, len(chunk) - len(output)))
        elif len(output) > len(chunk):
            output = output[:len(chunk)]

        # Онлайн обучение
        if self.online_learning and self.frame_counter % self.learning_interval == 0:
            self._online_learning_step(stft_clean, is_speech)
        self.frame_counter += 1

        return output

    def _online_learning_step(self, stft_clean, is_speech):
        if not self.online_learning or not is_speech:
            return
        noisy_feat = np.stack([np.real(stft_clean)[:, 0], np.imag(stft_clean)[:, 0]], axis=0)
        clean_feat = noisy_feat  # Тут можно заменить на отдельную цель, если есть clean
        self.noisy_buffer.append(noisy_feat)
        self.clean_buffer.append(clean_feat)
        if len(self.clean_buffer) >= 16:
            self._train_batch()

    def _train_batch(self):
        import random
        batch_size = min(16, len(self.clean_buffer))
        clean_batch = random.sample(list(self.clean_buffer), batch_size)
        noisy_batch = random.sample(list(self.noisy_buffer), batch_size)

        noisy_tensor = torch.from_numpy(np.array(noisy_batch)[:, :, :, np.newaxis]).float().to(self.device)
        clean_tensor = torch.from_numpy(np.array(clean_batch)).float().to(self.device)

        self.optimizer.zero_grad()
        out = self.model(noisy_tensor)
        loss = nn.MSELoss()(out, clean_tensor)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def save_model(self, path='denoise_model_online.pth'):
        torch.save(self.model.state_dict(), path)
        print(f"✅ Модель сохранена: {path}")
