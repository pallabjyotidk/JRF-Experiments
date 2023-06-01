import numpy as np
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import json

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self._relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self._relu(out)
        return out

res = 8
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
        
        self.E_2 = Block(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        if res == 32:
            self.E_3 = Block(in_channels=128, out_channels=256, kernel_size=4, stride=4, padding=1)
        else:
            # self.E_3 = Block(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
            self.E_3 = Block(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)


        self.E_2_prior = Block(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        if res == 32:
            self.E_3_prior = Block(in_channels=128, out_channels=256, kernel_size=4, stride=4, padding=1)
        else:
            # self.E_3 = Block(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
            self.E_3_prior = Block(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
    
    def forward(self, input, prior, res):
        E_2_output = self.E_2(input)
        print("E2 output - ", E_2_output.shape)

        if res == 32:
            E_3_output = self.E_3(E_2_output)
            print("E3 output - ", E_3_output.shape)

        if res == 8:
            prev_E2_output = torch.rand(2, 256, 4, 4, 4)
            print("prev_E2_output - ", prev_E2_output.shape)
            print("E_2_output - ", E_2_output.shape)

            E2_cat = torch.cat((prev_E2_output, E_2_output), dim=1)
            
            E_3_output = self.E_3(E2_cat)
            print("E3_output  - ", E_3_output.shape, "\n")

        return "Done"


model = Encoder(res)
if res == 32:
    model.E_2.register_forward_hook(hook_fn)
# print(model)

input = torch.rand(2, 64, 16, 16, 16)
prior = torch.rand(4, 64, 16, 16, 16)
a = model(input, prior, res)

# torch.save(model.state_dict(), model_path)
# torch.onxx.export(model, input, os.path.join(output_path, "res_32.onxx"), export_params=True)


# x = torch.load(model_path)
# # print(x)
# for key in x:
#     print(key)
#     print(x[key].shape)
    # E_2_res32 = x["E_2.conv1.weight"]
    # print(E_2_res32.shape)

