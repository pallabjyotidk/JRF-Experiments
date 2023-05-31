# python vis_patch_complete.py

import torch.nn as nn
import vis_numpy as np
import torch
import glob
import os
from torchsummary import summary

class BasicBlock_large(nn.Module):
    # BasicBlock places the stride for downsampling at 3x3 convolution for nn.conv3d
    # according to Bottleneck in torchvision.resnet 
    # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    def __init__(self,
                 mode: str,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int=3,
                 stride: int=2,
                 padding: int=1,
                 output_padding: int=1,
                 use_batchnorm: bool=False,
                 leaky: bool=False):
        
        super(BasicBlock_large, self).__init__()

        if mode == 'Encoder':
            self._conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        elif mode == 'Decoder':
            self._conv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        else:
            print ("Wrong mode, please enter 'Encoder' or 'Decoder'.")
            return
        
        self._conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self._conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if leaky:
            self._relu = nn.LeakyReLU(0.2) 
        else:    
            self._relu = nn.ReLU(inplace=True)

        self._use_batchnorm = use_batchnorm

        if self._use_batchnorm:
            self._bn_1 = nn.BatchNorm3d(out_channels)
            self._bn_2 = nn.BatchNorm3d(out_channels)
            self._bn_3 = nn.BatchNorm3d(out_channels)
        else:
            self._conv1 = nn.utils.weight_norm(self._conv1, name='weight')
            self._conv2 = nn.utils.weight_norm(self._conv2, name='weight')
            self._conv3 = nn.utils.weight_norm(self._conv3, name='weight')
           
    def forward(self, x):
        out = None
        identity = None

        if self._use_batchnorm:
            out = self._conv1(x)
            out = self._bn_1(out)
            out = self._relu(out)

            identity = out
            out = self._conv2(out)
            out = self._bn_2(out)
            out = self._relu(out)
            out = self._conv3(out)
            out = self._bn_3(out)
        
        else:
            out = self._conv1(x)
            out = self._relu(out)

            identity = out
            out = self._conv2(out)
            out = self._relu(out)
            out = self._conv2(out)

        out += identity
        out = self._relu(out)

        return out

class prior_encoder(nn.Module):
    def __init__(self, use_batchnorm=True, device='cuda', channel_num=128, patch_res=32, truncation=3, input_res=32):
        super(prior_encoder, self).__init__()

        self._device = device
        self._channel_num = channel_num
        self._patch_res = patch_res
        self._patch_num_edge = int(input_res / self._patch_res)
        # prior path
        self._priors_path = 'D:/Projects/JRF/random/patchComplete/priors'
        self._truncation = truncation

        self._encoder_input = BasicBlock_large('Encoder', in_channels=1, out_channels=int(channel_num/2))

        self._softmax = nn.Softmax(1)
        self._relu = nn.ReLU(inplace=False)

        priors = self._get_data()
        # print("Priors type - ", type(priors))
        priors_np = np.array(priors)
        priors = torch.from_numpy(priors_np)
        priors = priors.unsqueeze(1)
        self._codebook = None
        self._codebook = nn.Parameter(priors.float().to(self._device), requires_grad=True)
        # print(self._codebook)
    

    def _get_data(self):
        priors = []
        for proir_file in glob.iglob(os.path.join(self._priors_path, "*.npy")):
            with open(proir_file, 'rb') as data:
                prior = np.load(data)
                prior[np.where(prior > self._truncation)] = self._truncation
                prior[np.where(prior < -1* self._truncation)] = -1 * self._truncation
                priors.append(prior)
        return priors

model = prior_encoder()
params = model.parameters()
# for p in params:
#     with open('sample.txt', 'a') as file:
#         file.write(str(p))
#         file.write('\n')

for name, param in model.named_parameters():
    print(name, ' -------- ', param.shape, ' -------- ', param.requires_grad)
    print()
    with open('sample.txt', 'a') as file:
        s = name + '  -------  ' + str(param.shape) + '\n'
        file.write(s)
        file.write('\n')