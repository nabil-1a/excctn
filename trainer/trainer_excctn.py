# Created on 2023/09
# Author: M. Nabil Sarwar
# Training file for exCCTN - M. Nabil Sarwar, et al, "AN EXTENDED COMPLEX CONVOLUTIONAL NETWORK FOR SINGLE-CHANNEL SPEECH ENHANCEMENT, ICASSP 2024"


import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as Fn

from trainer.base_trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from utils.conv_stft import ConvSTFT, ConviSTFT
from utils.utils import compute_STOI, compute_PESQ, z_score, reverse_z_score


class Trainer(BaseTrainer):
    def __init__(self,
                 config,
                 resume,
                 model,
                 optimizer,
                 loss_function,
                 train_dataloader,
                 validation_dataloader):
        super(Trainer, self).__init__(config, resume, model, optimizer, loss_function)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
		self.transform = ConvSTFT(400, 100, 512)
        self.inverse = ConviSTFT(400, 100, 512)

    def _train_epoch(self, epoch):
        loss_total = 0.0
        for mixture, clean, n_frames_list, _ in tqdm(self.train_dataloader, desc="Training"):
            self.optimizer.zero_grad()

            
            if (len(clean) < len(mixture)):
                mixture = mixture[0:len(clean)]
            if (len(clean) > len(mixture)):
                clean = clean[0:len(mixture)]

            # Mixture and Clean STFTs
			mixture = mixture.to(self.device)
            clean = clean.to(self.device)
			mixture_D = self.transform(mixture)
            # mixture_D  = self.stft.transform(mixture)
            mixture_D = torch.tensor(, device=self.device)
            clean_D = self.transform(clean)

            enhanced_real, enhanced_imag = self.model(mixture_D)
			
			# just testing the ConviSTFT module
			if (epoch < 2):
                enhanced_D = torch.cat((enhanced_real, enhanced_imag), 1)
                enhanced = self.inverse(enhanced_D)
                print(f"enhance_D {enhanced_D.shape}")
            
            if (len(clean_D[2]) > len(enhanced_real[2])):
                clean_D = clean_D[:, :len(enhanced_real[2]), :]

            # Complex MSE Loss for Real and Imaginary Inputs
			loss = Fn.mse_loss(enhanced_real, clean_D[..., 0]) + Fn.mse_loss(enhanced_imag, clean_D[..., 1])
            
            loss.backward()
            self.optimizer.step()
            loss_total += float(loss)


    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        mixture_mean = None
        mixture_std = None
        stoi_c_n = []
        stoi_c_d = []
        pesq_c_n = []
        pesq_c_d = []

        for mixture, clean, n_frames_list, names in tqdm(self.validation_dataloader):
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            # Mixture and Clean
            mixture_D = self.transform(mixture)
            # mixture_mag = torch.sqrt(mixture_real ** 2 + mixture_imag ** 2)
            # mixture_phase = torch.atan2(mixture_real, mixture_imag)
            clean_D = self.transform(clean)

            enhanced_real, enhanced_imag = self.model(mixture_D)
			enhanced_D = torch.cat([enhanced_real, enhanced_imag], 1)
            enhanced = self.inverse(enhanced_D)
            enhanced_speeches = enhanced.detach().cpu().numpy()
            mixture_speeches = mixture.detach().cpu().numpy()
            clean_speeches = clean.detach().cpu().numpy()

            if (len(clean_D[1]) > len(enhanced_D[1])):
                clean_D = clean_D[:, :len(enhanced_D[1]), :, :]

            loss = Fn.mse_loss(enhanced_D[..., 0], clean_D[..., 0]) + Fn.mse_loss(enhanced_D[..., 1], clean_D[..., 1])
            loss_total += loss

            masks = []
            len_list = []
            for n_frames in n_frames_list:
                masks.append(torch.ones(n_frames, 101, dtype=torch.float32))
                len_list.append((n_frames - 1) * 100 + 400) # hop + win_len

            for i in range(len(n_frames_list)):
                enhanced = enhanced_speeches[i][:len_list[i]]
                mixture = mixture_speeches[i][:len_list[i]]
                clean = clean_speeches[i][:len_list[i]]

                stoi_c_n.append(compute_STOI(clean, mixture, sr=16000))
                stoi_c_d.append(compute_STOI(clean, enhanced, sr=16000))
                pesq_c_n.append(compute_PESQ(clean, mixture, sr=16000))
                pesq_c_d.append(compute_PESQ(clean, enhanced, sr=16000))


        get_ave = lambda metrics: np.sum(metrics) / len(metrics)

        print("STOI of cln-noisy VS cln-enh & PESQ of cln-noisy VS cln-enh")
        print(f"epoch: {epoch}")
        print(get_ave(stoi_c_n))
        print(get_ave(stoi_c_d))
        print(get_ave(pesq_c_n))
        print(get_ave(pesq_c_d))

        #dataloader_len = len(self.validation_dataloader)

        score = (get_ave(stoi_c_d) + self._transform_pesq_range(get_ave(pesq_c_d))) / 2
        return score

