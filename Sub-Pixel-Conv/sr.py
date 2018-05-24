from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np

img = Image.open('./input/image1.jpeg').convert('YCbCr')
y, cb, cr = img.split()

model = torch.load('./checkpoint/model_epoch_30.pth')
ip_to_tensor = ToTensor()
input = ip_to_tensor(y).view(1, -1, y.size[1], y.size[0])

out = model(input)
out_img_y = out[0].detach().numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
out_img.save('./output/out1.jpg')