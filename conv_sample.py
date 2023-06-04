import numpy as np
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import json
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    # optimal parameters
    parser.add_argument("--res", help="resolution", default=32, required=True, type=int)
    return parser

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self._relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self._relu(out)
        return out

args = parse_arguments().parse_args()
res = args.res
print('Current resolution is {}'.format(res))
model_name = "pc_sample_res_32.pt"
output_path = "./saved_models"
model_path = os.path.join(output_path, model_name)

encoders = {}
def hook_fn(m, i, o):
  encoders['Encoder2_input_res32'] = o.tolist()
  with open('./encoders_dict/encoders_input.json', 'w') as fp:
      json.dump(encoders, fp)
      fp.close()

class Encoder(nn.Module):
    def __init__(self, res):
        super(Encoder, self).__init__()
        
        self.E_1 = Block(in_channels=1, out_channels=64, kernel_size=3, stride=2)
        self.E_2 = Block(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        if res == 32:
            self.E_3 = Block(in_channels=128, out_channels=256, kernel_size=4, stride=4)
        elif res == 8:
            # self.E_3 = Block(in_channels=128, out_channels=256, kernel_size=3, stride=2)
            self.E_3 = Block(in_channels=256, out_channels=256, kernel_size=3, stride=2)
        else:
            self.E_3 = Block(in_channels=256, out_channels=256, kernel_size=3, stride=1)


        self.E_1_prior = Block(in_channels=1, out_channels=64, kernel_size=3, stride=2)
        self.E_2_prior = Block(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        if res == 32:
            self.E_3_prior = Block(in_channels=128, out_channels=256, kernel_size=4, stride=4)
        elif res == 8:
            # self.E_3 = Block(in_channels=128, out_channels=256, kernel_size=3, stride=2)
            self.E_3_prior = Block(in_channels=256, out_channels=256, kernel_size=3, stride=2)
        else:
            self.E_3_prior = Block(in_channels=256, out_channels=256, kernel_size=3, stride=1)
    
    def forward(self, input, prior, res):
        E_1_output = self.E_1(input)
        print("E1 output - ", E_1_output.shape)

        E_2_output = self.E_2(E_1_output)
        print("E2 output - ", E_2_output.shape)

        if res == 32:
            E_3_output = self.E_3(E_2_output)
            print("E3 output - ", E_3_output.shape)

        if res == 8:
            prev_E2_output = torch.rand(6, 128, 8, 8, 8)
            print("prev_E2_output - ", prev_E2_output.shape)

            E2_cat = torch.cat((prev_E2_output, E_2_output), dim=1)
            print("E2_cat - ", E2_cat.shape)
            
            E_3_output = self.E_3(E2_cat)
            print("E3_output  - ", E_3_output.shape, "\n")
        
        if res == 4:
            prev_E2_output = torch.rand(6, 128, 8, 8, 8)
            print("prev_E2_output - ", prev_E2_output.shape)

            E2_cat = torch.cat((prev_E2_output, E_2_output), dim=1)
            print('E2_cat shape ', E2_cat.shape)

            E_3_output = self.E_3(E2_cat)
            print("E3 output - ", E_3_output.shape)


        print('---------------')


        E_1_prior_output = self.E_1_prior(prior)
        print("E1 prior output - ", E_1_prior_output.shape)

        E_2_prior_output = self.E_2_prior(E_1_prior_output)
        print("E2 prior output - ", E_2_prior_output.shape)

        if res == 32:
            E_3_prior_output = self.E_3_prior(E_2_prior_output)
            print("E3 prior output - ", E_3_prior_output.shape)

        if res == 8:
            prev_E2_output = torch.rand(3, 128, 8, 8, 8)
            print("prev_E2_output - ", prev_E2_output.shape)

            E2_prior_cat = torch.cat((prev_E2_output, E_2_prior_output), dim=1)
            print("E2_prior_cat - ", E2_prior_cat.shape)
            
            E_3_prior_output = self.E_3_prior(E2_prior_cat)
            print("E3 prior output  - ", E_3_prior_output.shape, "\n")
        
        if res == 4:
            prev_E2_output = torch.rand(3, 128, 8, 8, 8)
            print("prev_E2_output - ", prev_E2_output.shape)

            E2_prior_cat = torch.cat((prev_E2_output, E_2_prior_output), dim=1)
            print('E2_prior_cat shape ', E2_prior_cat.shape)

            E_3_prior_output = self.E_3_prior(E2_prior_cat)
            print("E3 prior output - ", E_3_prior_output.shape)

        return "Done"

model = Encoder(res)
if res == 32:
    model.E_2.register_forward_hook(hook_fn)
# print(model)

input = torch.rand(6, 1, 32, 32, 32)
prior = torch.rand(3, 1, 32, 32, 32)
a = model(input, prior, res)
print('-------')

