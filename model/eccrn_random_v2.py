import torch.nn as nn
import torch.nn.functional as Fn
import torch


class ECCRN(nn.Module):
    '''
       ECCRN Encoder - TCN - Decoder
    '''

    def __init__(self):
        super(ECCRN, self).__init__()
        # almost done
        # self.conv_pre = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 1), stride=(1, 1)
        self.conv_re = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(2, 5), stride=(1, 3))
        self.conv_im = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(2, 5), stride=(1, 3))
        # self.bn_re = nn.BatchNorm2d(num_features=128)
        # self.bn_im = nn.BatchNorm2d(num_features=128)
        self.a1 = nn.Tanh()
        self.a2 = nn.Tanh()
        self.a3 = nn.Tanh()
        self.a4 = nn.Tanh()
        self.a5 = nn.Tanh()
        self.a6 = nn.Tanh()
        self.a7 = nn.Tanh()
        #self.a8 = nn.Tanh()
        #self.a9 = nn.Tanh()
        self.a10 = nn.Tanh()
        self.a11 = nn.Tanh()

        # write tcn code here
        self.tn1x1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.tn1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 2), stride=(3, 1), bias=False)
        self.tn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 2), stride=(3, 1), bias=False)
        self.tn3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.tn4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), dilation=1, groups=128, stride=(1, 1), bias=False)
        self.tn5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        
        self.tn1x1i = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.tn1i = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 2), stride=(3, 1), bias=False)
        self.tn2i = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 2), stride=(3, 1), bias=False)
        self.tn3i = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.tn4i = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), dilation=1, groups=128, stride=(1, 1), bias=False)
        self.tn5i = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.bn4 = nn.BatchNorm2d(num_features=256)

        self.tn_mask_re = nn.Tanh()
        self.tn_mask_im = nn.Tanh()
        self.tn_re_att = nn.Tanh()
        self.tn_im_att = nn.Tanh()

        self.d1x1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.d1x1i = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.tconv_re = nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=(2, 5), stride=(1, 3))
        self.tconv_im = nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=(2, 5), stride=(1, 3))
        # self.tbn_re = nn.BatchNorm2d(num_features=1)
        # self.tbn_im = nn.BatchNorm2d(num_features=1)

    def forward(self, x):
        # shape of input : (batch, axis1, axis2, 2)
        # encoder
        x.unsqueeze_(1)
        #_, _, size_of_T, _, _ = x.size()
        
        e_real = self.conv_re(x[..., 0])
        e_imag = self.conv_im(x[..., 1]) + self.a1(e_real)
        e_real = e_real + self.a2(e_imag)

        # temporal network block

        tn1x1 = self.bn1(self.tn1x1(e_real))
        tn1_output = self.tn1(tn1x1)
        tn2_output = self.tn2(tn1_output)
        tn3_output = self.tn3(tn2_output)
        tn4_output = self.tn4(tn3_output)
        tn5_output = self.bn2(self.tn5(tn4_output))
        
        tn1x1i = self.a3(self.bn3(self.tn1x1i(e_imag)))
        tn1_outputi = self.a4(self.tn1i(tn1x1i))
        tn2_outputi = self.a5(self.tn2i(tn1_outputi))
        tn3_outputi = self.a6(self.tn3i(tn2_outputi))
        tn4_outputi = self.a7(self.tn4i(tn3_outputi))
        tn5_outputi = self.bn4(self.tn5i(tn4_outputi))

        B, F, t_enc, D = e_imag.size()
        B, F_tn, t_tn, D_tn = tn5_output.size()

        tn5_output = Fn.pad(tn5_output, (0, D - D_tn, 0, t_enc - t_tn))
        tn5_outputi = Fn.pad(tn5_outputi, (0, D - D_tn, 0, t_enc - t_tn))
        # tn1_output = Fn.pad(tn1_output, (0, D_diff, 0, T_enc - T_tn5))
        #print(f"{tn5_output.shape},{tn1x1.shape}")
        tn5_output = tn5_output #* self.a9(tn1x1i)
        tn5_outputi = tn5_outputi #* self.a10(tn1x1)


        tn_mask_re = self.tn_mask_re(tn5_output) - self.tn_im_att(tn5_outputi)
        masked_re = e_real * tn_mask_re
        tn_mask_im = self.tn_mask_im(tn5_outputi) + self.tn_re_att(tn5_output)
        masked_im = e_imag * tn_mask_im

        all_re = torch.cat((masked_re, e_real), 1)
        all_im = torch.cat((masked_im, e_imag), 1)
        all_re = self.d1x1(all_re) #- self.a8(all_im)
        all_im = self.d1x1i(all_im) #+ self.a9(all_re)

        # decoder

        d_real = self.tconv_re(all_re)
        d_imag = self.tconv_im(all_im) + self.a10(d_real)
        d_real = d_real - self.a11(d_imag)
        decoder_output = torch.stack((d_real, d_imag), dim=-1)
        

        return decoder_output.squeeze()
