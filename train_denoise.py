import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt


class SpeechDenoiseDataset(Dataset):
    def __init__(
        self,
        clean_dir,
        noisy_dir,
        sr,
        n_fft=512,
        hop_length=160,
        win_length=480,
        context_frames=5,
        max_files=None,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length
        self.context_frames = context_frames
        self.freq_dim = n_fft // 2 + 1

        clean_files = sorted(f for f in os.listdir(clean_dir) if f.endswith(".wav"))
        noisy_files = sorted(f for f in os.listdir(noisy_dir) if f.endswith(".wav"))

        if max_files:
            clean_files = clean_files[:max_files]
            noisy_files = noisy_files[:max_files]

        assert len(clean_files) == len(noisy_files)

        self.samples = []

        print("▶ Preparing STFT training pairs")
        for c, n in tqdm(zip(clean_files, noisy_files), total=len(clean_files)):
            clean, _ = librosa.load(os.path.join(clean_dir, c), sr=sr)
            noisy, _ = librosa.load(os.path.join(noisy_dir, n), sr=sr)

            min_len = min(len(clean), len(noisy))
            clean = clean[:min_len]
            noisy = noisy[:min_len]

            clean_stft = librosa.stft(
                clean, n_fft=n_fft, hop_length=hop_length, win_length=win_length
            )
            noisy_stft = librosa.stft(
                noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length
            )

            for t in range(clean_stft.shape[1] - context_frames):
                noisy_ctx = noisy_stft[:, t : t + context_frames]
                clean_tgt = clean_stft[:, t + context_frames // 2]

                self.samples.append(
                    (
                        np.stack([noisy_ctx.real, noisy_ctx.imag], axis=0),
                        np.stack([clean_tgt.real, clean_tgt.imag], axis=0),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class DenoiseModel(nn.Module):
    def __init__(self, n_fft, context_frames, dropout_rate=0.2):
        super().__init__()
        self.freq_dim = n_fft // 2 + 1
        self.context_frames = context_frames

        self.conv1 = nn.Conv2d(2, 32, 3, padding="same")
        self.conv2 = nn.Conv2d(32, 64, 3, padding="same")

        self.temporal_conv1 = nn.Conv2d(64, 64, (1, 3), padding=(0, 1))
        self.temporal_conv2 = nn.Conv2d(64, 64, (1, 3), padding=(0, 1))

        self.attention = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 1),
            nn.Sigmoid(),
        )

        self.conv3 = nn.Conv2d(64, 32, 3, padding="same")
        self.conv4 = nn.Conv2d(32, 16, 3, padding="same")
        self.out = nn.Conv2d(16, 2, 3, padding="same")

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(16)

        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.dropout3 = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = self.dropout1(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.conv2(x))))

        x = torch.relu(self.temporal_conv1(x))
        x = torch.relu(self.temporal_conv2(x))

        x = x * self.attention(x)

        x = self.dropout3(torch.relu(self.bn3(self.conv3(x))))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.tanh(self.out(x))

        return torch.mean(x, dim=3)

def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs=40,
    lr=1e-3,
):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()

        val_loss /= len(val_loader)
        print(f"✓ Val loss: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                },
                "best_model.pth",
            )
            print("💾 Saved best model")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SpeechDenoiseDataset(
        clean_dir="data/clean",
        noisy_dir="data/noisy",
        sr=16000,
        n_fft=512,
        hop_length=160,
        win_length=480,
        context_frames=5,
        max_files=20,
    )

    train_ds, val_ds = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    model = DenoiseModel(n_fft=512, context_frames=5).to(device)

    train(model, train_loader, val_loader, device)


if __name__ == "__main__":
    main()