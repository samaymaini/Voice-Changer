import os
import glob
import torchaudio
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms
from torchaudio.functional import resample

class LogMelSpectrogram(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.melspctrogram = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            center=False,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=128,
            mel_scale="slaney",
        )

    def forward(self, wav):
        padding = (1024 - 160) // 2
        wav = F.pad(wav, (padding, padding), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel

def extract_mels(data_path: str, mel_path: str, device: int):
    """
    Converts all wavs in data_path to corresponding mels in mel_path.
    
    Args:
        data_path (str): path with all .wav files to be converted.
        mel_path (str): path where all .npy mel files will be stored.
        device (int): CUDA device to use
    """
    device = f"cuda:{device}"

    for file in tqdm(glob.glob(os.path.join(data_path, '*.wav'))):
        wav, sr = torchaudio.load(file)
        wav = wav.to(device)

        melspectrogram = LogMelSpectrogram().to(device)
        wav = resample(wav, sr, 16000)
        mel = melspectrogram(wav.unsqueeze(0)).squeeze(0).transpose(2, 1).cpu().numpy()

        np.save(os.path.join(mel_path, f"{os.path.basename(file)[:-4]}.npy"), mel)


if __name__ == "__main__":
    extract_mels("../../../LJSpeech-1.1/wavs", "../../../LJSpeech-1.1/mels", 7)