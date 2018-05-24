import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

class ConvSubPixel(nn.Module):
	def __init__(self, upscale_factor):
		super(ConvSubPixel, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, (5,5), (1,1), (2,2))
		self.conv2 = nn.Conv2d(64, 32, (3,3), (1,1), (1,1))
		self.conv3 = nn.Conv2d(32, (upscale_factor**2)*3, (3,3), (1,1), (1,1))
		self.ps = nn.PixelShuffle(upscale_factor)
		self.init_weights()

	def forward(self, x):
		act1 = F.relu(self.conv1(Variable(x, requires_grad=True)))
		act2 = F.relu(self.conv2(act1))
		noact = self.ps(self.conv3(act2))
		return noact

	def init_weights(self):
		init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.conv3.weight)
		
		