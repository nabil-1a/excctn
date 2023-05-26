import math
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def overlap_and_add(sig, step):

    outer_dim = sig.size()[:-2]
    frames, fr_len = sig.size()[-2:]
    subfr_len = math.gcd(fr_len, step)
    substeps = step // subfr_len
    subfr_count = fr_len // subfr_len
    output_size = step * (frames - 1) + fr_len
    out_subframes = output_size // subfr_len
    subfr_sig = sig.reshape(*outer_dim, -1, subfr_len)
    fr = torch.arange(0, out_subframes).unfold(0, subfr_count, substeps)
    fr = sig.new_tensor(fr).long()
    fr = fr.contiguous().view(-1)
	
    output = sig.new_zeros(*outer_dim, out_subframes, subfr_len)
    output.index_add_(-2, frame, subfr_sig)
    output = output.view(*outer_dim, -1)
    return output

def remove_pad(inputs, inputs_lengths):

    results = []
    dim = inputs.dim()
    if dim == 3:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 3: 
            results.append(input[:,:length].view(C, -1).cpu().numpy())
        elif dim == 2:
            results.append(input[:length].view(-1).cpu().numpy())
    return results