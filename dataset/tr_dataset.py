import os

import librosa
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

'''
Generate dataset for training and validation
'''

class TrainDataset(Dataset):


    def __init__(self,
                 mix_dataset,
                 clean_dataset,
                 limit=None,
                 offset=0,
                 ):
        
        
        mix_dataset = os.path.abspath(os.path.expanduser(mix_dataset))
        clean_dataset = os.path.abspath(os.path.expanduser(clean_dataset))
        assert os.path.exists(mix_dataset) and os.path.exists(clean_dataset)
        print("Look for datasets...")
        mix_wav_files = librosa.util.find_files(mix_dataset, ext="wav", limit=limit, offset=offset)
        clean_wav_files = librosa.util.find_files(clean_dataset, ext="wav", limit=limit, offset=offset)
        assert len(mix_wav_files) == len(clean_wav_files)
        print(f"\t Original length: {len(mix_wav_files)}")
        self.length = len(mix_wav_files)
        self.mix_wav_files = mix_wav_files
        self.clean_wav_files = clean_wav_files
        print(f"\t Offset: {offset}")
        print(f"\t Limit: {limit}")
        print(f"\t Final length: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, item):
	
        mix_path = self.mix_wav_files[item]
        clean_path = self.clean_wav_files[item]
        name = os.path.splitext(os.path.basename(clean_path))[0]

        mix, sr = sf.read(mix_path, dtype="float32")
        clean, sr = sf.read(clean_path, dtype="float32")

        assert sr == 16000
		
        if (len(clean) < len(mix)):
            mix = mix[0:len(clean)]
        else:
            clean = clean[0:len(mix)]
        assert mix.shape == clean.shape

        n_frames = (len(mix) - 400) // 101 # len - window_len // frame shift+1

        return mix, clean, n_frames, name
