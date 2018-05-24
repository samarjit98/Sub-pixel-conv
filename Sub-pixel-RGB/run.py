from __future__ import print_function
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ConvSubPixel
from data import get_training_set, get_test_set
import os

torch.manual_seed(123)

print('Loading datasets...')
train_set = get_training_set(8)
test_set = get_test_set(8)
training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=32, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=32, shuffle=False)

print('Building model...')
model = ConvSubPixel(upscale_factor=8)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-2)


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, truth = batch[0], batch[1]

        optimizer.zero_grad()
        loss = criterion(model(input), truth)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0], batch[1]

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")
    model_out_path = "./checkpoint/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, 30 + 1):
    train(epoch)
    test()
    checkpoint(epoch)