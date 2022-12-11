import torch
from torch import nn
from api.models import *
from api.models.pytorch import *


class AlexNet(nn.Module):
  def __init__(self, output_dim):
    super(AlexNet, self).__init__()
    self._to_linear = None
    self.x = torch.randn(3, IMG_WIDTH, IMG_HEIGHT).view(-1, 3, IMG_WIDTH, IMG_HEIGHT)
    self.features = nn.Sequential(
        nn.Conv2d(3, 64, 3, 2, 1), # in_channels, out_channels, kernel_size, stride, padding
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 192, 3, padding=1),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True), 
        nn.Conv2d(192, 384, 3, padding=1),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 512, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 256, 3, padding=1),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True)
  )
    self.conv(self.x)
    self.classifier = nn.Sequential(
        nn.Dropout(.5),
        nn.Linear(self._to_linear, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, output_dim),
    )

  def conv(self, x):
    x = self.features(x)
    if self._to_linear is None:
      self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]
    return x

  def forward(self, x):
    x = self.conv(x)
    h = x.view(x.shape[0], -1)
    x = self.classifier(h)
    return x, h
     
print(" ✅ LOADING PYTORCH AIR MODEL!\n") 
# Hyper params
OUTPUT_DIM = len(classes)

# Model instance
air_model = AlexNet(OUTPUT_DIM).to(device)

air_model.load_state_dict(torch.load(PYTORCH_AIR_MODEL_PATH, 
                                     map_location=device))
print(" ✅ LOADING PYTORCH AIR MODEL DONE!\n")


