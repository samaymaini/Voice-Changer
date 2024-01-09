# Voice Conversion and Avatar Visualization via Inversion

Guided by [A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion](https://arxiv.org/pdf/2111.02392.pdf).

## Content Encoder

We use pretrained WavLM and do k-means clustering to get discrete speech units
from a given waveform (in ```soft_vc_wavlm/extract_wavlm.py```).

From this we train a WavLM to mel spectrogram acoustic model using ```soft_vc_wavlm/acoustic_model.py```
and ```soft_vc_wavlm/train_acoustic_model.py```.

Final voice conversion step is to train a Hifi-GAN vocoder using a specific
person's voice (TODO).