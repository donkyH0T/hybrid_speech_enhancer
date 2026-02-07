import queue
import threading
import time
import json
import sys
from pathlib import Path
import numpy as np
import sounddevice as sd
import webrtcvad
from translator import Translator
from vosk import Model, KaldiRecognizer, SetLogLevel
from collections import deque
from hybrid_speech_enhancer import OnlineHybridSpeechEnhancer
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
from scipy import signal
warnings.filterwarnings('ignore')

class RealTimeTranscriber:
    def __init__(self, model_path):
        self.model_path = model_path
        self.sample_rate = 16000
        self.channels = 1
        self.frame_duration_ms = 30
        self.noise_reduction = False
        self.translator = Translator(source_lang='ru', target_lang='en')
        self.stop_event = threading.Event()
        self.audio_q = queue.Queue(maxsize=400)
        self.segment_q = queue.Queue()
        self.hybrid_speech_enhancer = OnlineHybridSpeechEnhancer(
            sr=self.sample_rate,
            vad_aggressiveness=2,
            online_learning=False
        )
        self.vad = webrtcvad.Vad(2)
        self.model = Model(self.model_path)
        self.rec = KaldiRecognizer(self.model, self.sample_rate)
        self.rec.SetWords(False)
        self.energy_buffer = deque(maxlen=120)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("sounddevice status:", status, file=sys.stderr)
        if indata.shape[1] > 1:
            indata = np.mean(indata, axis=1, keepdims=True)
        indata = np.clip(indata, -1.0, 1.0)
        energy = np.sqrt(np.mean(indata**2))
        self.energy_buffer.append(energy)
        try:
            self.audio_q.put_nowait(indata.copy().flatten())
        except queue.Full:
            try:
                self.audio_q.get_nowait()
                self.audio_q.put_nowait(indata.copy().flatten())
            except queue.Empty:
                pass
        if self.audio_q.qsize() > 350:
            time.sleep(0.005)

    def producer_thread(self):
        frame_samples = int(self.sample_rate * self.frame_duration_ms / 1000)
        sample_buf = deque(maxlen=frame_samples * 4)
        while not self.stop_event.is_set():
            try:
                frame = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            sample_buf.extend(frame.tolist())
            while len(sample_buf) >= frame_samples:
                chunk = np.array([sample_buf.popleft() for _ in range(frame_samples)], dtype=np.float32)
                if self.noise_reduction:
                    clean = self.hybrid_speech_enhancer.process_chunk_enhanced(chunk)
                else:
                    clean = chunk

                ts = time.time() - (len(sample_buf) / self.sample_rate)
                seg = {
                    "clear_frame": clean,
                    "timestamp": ts
                }
                try:
                    self.segment_q.put_nowait(seg)
                except queue.Full:
                    try:
                        self.segment_q.get_nowait()
                        self.segment_q.put_nowait(seg)
                    except queue.Empty:
                        pass
        try:
            self.segment_q.put_nowait(None)
        except Exception:
            pass

    def asr_consumer_thread(self):
        SPEECH_HANG_MS = 600
        MIN_SPEECH_MS = 120
        last_speech_time = None
        speech_start_time = None
        consecutive_speech_frames = 0
        while True:
            item = self.segment_q.get()
            if item is None:
                break
            clear_frame = item["clear_frame"]
            ts = item["timestamp"]
            pcm = (np.clip(clear_frame, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            try:
                is_speech = self.vad.is_speech(pcm, self.sample_rate)
            except:
                is_speech = False

            if is_speech:
                consecutive_speech_frames += 1
                if speech_start_time is None:
                    speech_start_time = ts

                if consecutive_speech_frames >= (MIN_SPEECH_MS / self.frame_duration_ms):
                    last_speech_time = ts
                    accepted = self.rec.AcceptWaveform(pcm)
                    if accepted:
                        out = json.loads(self.rec.Result())
                        text = out.get("text", "")
                        if text:
                            print(f"[RESULT] {text}")
                    else:
                        pres = json.loads(self.rec.PartialResult())
                        partial = pres.get("partial", "")
                        if partial:
                            print(f"[PARTIAL] {partial}", end="\r")
            else:
                consecutive_speech_frames = 0
            if last_speech_time is not None and speech_start_time is not None:
                silence_duration = ts - last_speech_time
                if silence_duration > (SPEECH_HANG_MS / 1000.0):
                    speech_duration = last_speech_time - speech_start_time
                    if speech_duration > (MIN_SPEECH_MS / 1000.0):
                        out = json.loads(self.rec.FinalResult())
                        text = out.get("text", "")
                        if text:
                            latency = time.time() - last_speech_time
                            print(f"\n[SUBTITLE] {text} (latency ~{latency:.3f}s)")
                    last_speech_time = None
                    speech_start_time = None

    def start(self):
        SetLogLevel(-1)
        print("Запуск улучшенной системы транскрипции в реальном времени...")
        print("Нажмите Ctrl+C для остановки\n")
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.03),
                dtype='float32'
            ):
                producer = threading.Thread(target=self.producer_thread, daemon=True)
                consumer = threading.Thread(target=self.asr_consumer_thread, daemon=True)
                producer.start()
                consumer.start()
                while not self.stop_event.is_set():
                    time.sleep(0.04)
        except KeyboardInterrupt:
            print("\nОстановка по запросу пользователя...")
            self.stop_event.set()
            time.sleep(0.15)
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            self.stop_event.set()
        finally:
            print("Программа завершена.")