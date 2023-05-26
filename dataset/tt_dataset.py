import os
import librosa
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

'''
Generate dataset for enhancement
'''

class TestDataset(Dataset):

    def __init__(self,
                 mix_dataset,
                 limit=None,
                 offset=0,
                 ):
        mix_dataset = os.path.abspath(os.path.expanduser(mix_dataset))
        assert os.path.exists(mix_dataset)
        mix_wav = librosa.util.find_files(mix_dataset, ext="wav", limit=limit, offset=offset)
        print(f"\t Initial length: {len(mix_wav)}")

        self.length = len(mix_wav)
        self.mixed_waves = mixed_waves
        print(f"\t Offset: {offset}, Limit: {limit}, Final length: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        mix_path = self.mixed_waves[item]
        name = os.path.splitext(os.path.basename(mix_path))[0]
        mix, sr = sf.read(mix_path, dtype="float32")
        if sr != 16000:
          print(sr)
          mix = librosa.resample(mix, sr, 16000)
          sr = 16000

        n_frames = (len(mix) - 400) // 101
        return mix, 0, n_frames, name
