import os
import glob
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MelDataset(Dataset):
    def __init__(self, root: str, train: bool = True, discrete: bool = False):
        self.discrete = discrete
        #self.mels_dir = root / "mels"
        #self.units_dir = root / "discrete" if discrete else root / "soft"

        self.mels_dir = os.path.join(root, "mels")
        self.units_dir = os.path.join(root, "wavlm_clustered")

        #pattern = "*.npy" if train else "dev/**/*.npy"
        self.metadata = [
            os.path.basename(path[:-4]) for path in glob.glob(os.path.join(self.mels_dir, "*.npy"))
        ]
        
        self.train = train

        train_split = int(len(self.metadata) * 0.8)
        if train:
            self.metadata = self.metadata[:train_split]
        else:
            self.metadata = self.metadata[train_split:]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]
        mel_path = os.path.join(self.mels_dir, f"{path}.npy")
        units_path = os.path.join(self.units_dir, f"{path}.npy")

        mel = np.load(mel_path)
        units = np.load(units_path)

        length = 2 * units.shape[0]

        mel = torch.from_numpy(mel[:, :length, :])
        #mel = F.pad(mel, (0, 0, 1, 0))
        units = torch.from_numpy(units)
        if self.discrete:
            units = units.long()
        return mel.squeeze(0), units

    def pad_collate(self, batch):
        mels, units = zip(*batch)

        mels, units = list(mels), list(units)
        #print(mels[0].shape, mels[1].shape, mels[2].shape)

        mels_lengths = torch.tensor([x.size(0) - 1 for x in mels])
        units_lengths = torch.tensor([x.size(0) for x in units])

        mels = pad_sequence(mels, batch_first=True)
        units = pad_sequence(
            units, batch_first=True, padding_value=100 if self.discrete else 0
        )

        return mels, mels_lengths, units, units_lengths

if __name__ == "__main__":
    mel_dataset = MelDataset("../../audios/", train=True, discrete=True)
    print(len(mel_dataset))
    print(mel_dataset[0][0].shape)