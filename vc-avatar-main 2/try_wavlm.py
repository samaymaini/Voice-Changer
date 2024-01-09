import s3prl.hub as hub
import torch
import soundfile as sf
import librosa

import torchaudio
from torchaudio.models.rnnt import _Predictor
import torchaudio.transforms as transforms
from torchaudio.functional import resample
import torch.nn.functional as F
import time

from soft_vc_wavlm.acoustic_model.acoustic_model import AcousticModel

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

model_name = "wavlm_large"
wavlm_model = getattr(hub, model_name)()

device = "cuda:2"
wavlm_model = wavlm_model.to(device)

wav, sr = torchaudio.load("audios/mngu0_s1_0001.wav")
wav = wav.to(device)
#wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(device)
print(len(wav[0]))

with torch.no_grad():
    states = wavlm_model(wav)["hidden_states"]

feature = states[9].squeeze(0)
feature = feature.long().to(device)
print(feature.shape)

melspectrogram = LogMelSpectrogram()
melspectrogram.to(device)
wav = resample(wav, sr, 16000).to(device)
logmel = melspectrogram(wav).to(device)

wavlm_acoustic = AcousticModel(discrete=True).to(device)
#print(wavlm_acoustic(feature, logmel))

