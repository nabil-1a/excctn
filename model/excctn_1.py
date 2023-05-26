# Created on 2020/09
# Author: M. Nabil Sarwar
# Training file for exCCTN - "AN EXTENDED COMPLEX CONVOLUTIONAL NETWORK FOR SINGLE-CHANNEL SPEECH ENHANCEMENT, ICASSP 2021"

import torch.nn as nn
import torch.nn.functional as Fn
import torch


class EXCCTN(nn.Module):
    '''
       ECCTN Encoder - TMN - Decoder
    '''

    def __init__(self):
        super(EXCCTN, self).__init__()

        # Complex Shared Encoder
        self.conv_re = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(2, 5), stride=(1, 3))
        self.conv_im = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(2, 5), stride=(1, 3))
		
		# Tanh activations for GCRs and GISs
        self.a1 = nn.Tanh()
        self.a2 = nn.Tanh()
        self.a3 = nn.Tanh()
        self.a4 = nn.Tanh()
        self.a5 = nn.Tanh()
        self.a6 = nn.Tanh()
        self.a7 = nn.Tanh()
        self.a8 = nn.Tanh()
        self.a9 = nn.Tanh()
        self.a10 = nn.Tanh()
        self.a11 = nn.Tanh()
        self.a12 = nn.Tanh()
        self.a13 = nn.Tanh()
		
		# Pointwise bridge (PWB) between encoder and TMN
        self.tn1x1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.tn1x1i = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # Temporal Masking Network (TMN) - real
        self.tn1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(11, 2), stride=(7, 1), bias=False)
        self.tn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 11), stride=(1, 7), bias=False)
        self.tn3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.tn4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), dilation=1, groups=128, stride=(1, 1), bias=False)
        self.tn5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        
		# Temporal Masking Network (TMN) - imaginary
        self.tn1i = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(11, 2), stride=(7, 1), bias=False)
        self.tn2i = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 11), stride=(1, 7), bias=False)
        self.tn3i = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.tn4i = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), dilation=1, groups=128, stride=(1, 1), bias=False)
        self.tn5i = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # Group Layer Normalization layers for TMN
		self.GN1 = nn.GroupNorm(1, 128)
        self.GN2 = nn.GroupNorm(1, 256)
        self.GN3 = nn.GroupNorm(1, 128)
        self.GN4 = nn.GroupNorm(1, 256)

        # CTanh Masks and GCR
		self.mask_real = nn.Tanh()
        self.mask_imag = nn.Tanh()
        self.GCR1 = nn.Tanh()
        self.GCR1i = nn.Tanh()
		self.GCR2 = nn.Tanh()
        self.GCR2i = nn.Tanh()

        # Pointwise bridge (PWB) between TMN and Decoder
		self.d1x1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.d1x1i = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # Complex Shared Decoder
		self.tconv_re = nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=(2, 5), stride=(1, 3))
        self.tconv_im = nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=(2, 5), stride=(1, 3))


    def forward(self, x):
        
		'''
		Input Shape : (batch, Samples, fdim, 2)
        '''
		
        x.unsqueeze_(1)
        
        e_real = self.conv_re(x[..., 0]) - self.a1(self.conv_im(x[..., 1]))
        e_imag = self.conv_re(x[..., 1]) + self.a2(self.conv_im(x[..., 0]))
		# experimentation with Relayed Complex Convolution (RCC)
        # e_real = e_real - self.a2(e_imag)
		
		# pointwise bridge for real
        tn1x1 = self.GN1(self.tn1x1(e_real))
		
		# TMN - real
        tn1_output = self.tn1(tn1x1)
        tn2_output = self.tn2(tn1_output)
        tn3_output = self.tn3(tn2_output)
        tn4_output = self.tn4(tn3_output)
        tn5_output = self.GN2(self.tn5(tn4_output))
		
        # pointwise bridge for imaginary
        tn1x1i = self.a3(self.GN3(self.tn1x1i(e_imag)))
		
		# TMN - imaginary. Gated Imaginary Stream (GIS) to manage imaginary data flow
        tn1_outputi = self.a4(self.tn1i(tn1x1i))
        tn2_outputi = self.a5(self.tn2i(tn1_outputi))
        tn3_outputi = self.a6(self.tn3i(tn2_outputi))
        tn4_outputi = self.a7(self.tn4i(tn3_outputi))
        tn5_outputi = self.a8(self.GN4(self.tn5i(tn4_outputi)))

        B, F, t_enc, D = e_imag.size()
        B, F_tn, t_tn, D_tn = tn5_output.size()
		
		# padding for fixing dimensional differences
        tn5_output = Fn.pad(tn5_output, (0, D - D_tn, 0, t_enc - t_tn))
        tn5_outputi = Fn.pad(tn5_outputi, (0, D - D_tn, 0, t_enc - t_tn))
		
        # Gated Complex Relay (GCR) + skip-connection for information communication between real and imaginary
        tn5_output = tn5_output - self.GCR1(e_imag)
        tn5_outputi = tn5_outputi + self.GCR1i(e_real)
		
		# Second GCR to compute multiplicative masks
        mask_real = self.mask_real(tn5_output) - self.GCR2(tn5_outputi)
        masked_real = e_real * mask_real
        mask_imag = self.mask_imag(tn5_outputi) + self.GCR2i(tn5_output)
        masked_imag = e_imag * mask_imag
		
		# Concatenate masked with encoder output
        all_re = torch.cat((masked_real, e_real), 1)
        all_im = torch.cat((masked_imag, e_imag), 1)
		
		# Pointwise bridge (PWB)
        all_re = self.d1x1(all_re)
        all_im = self.a9(self.d1x1i(all_im))

        # decoder
        d_real = self.tconv_re(all_re) - self.a10(self.tconv_im(all_im))
        d_imag = self.tconv_re(all_im) + self.a11(self.tconv_im(all_re))
        #d_real = d_real - self.a11(d_imag)
		
        #decoder_output = torch.stack((d_real, d_imag), dim=-1)
		d_real = d_real.permute(0, 1, 3, 2)
		d_imag = d_imag.permute(0, 1, 3, 2)
        

        return d_real.squeeze(), d_imag.squeeze()
