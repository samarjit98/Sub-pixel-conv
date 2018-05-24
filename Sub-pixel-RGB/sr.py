from __future__ import print_function
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np

img = Image.open('./input/image1.jpeg').convert('RGB')

model = torch.load('./checkpoint/model_epoch_30.pth')
ip_to_tensor = ToTensor()
input = ip_to_tensor(img).view(1, -1, img.size[1], img.size[0])

out = model(input)
out_img_y = out[0].detach().numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
out_img_y.save('./output/out1.jpg')

out_img_b = img.resize(out_img_y.size, Image.BICUBIC)
out_img_b.save('./output/out2.jpg')
