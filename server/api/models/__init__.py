import torch
import os
from torchvision import transforms
from torch.nn import functional as F
import numpy as np
from api.types import *

# torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# allowed images 
allowed_extensions = ['.jpg', '.JPG', '.png' ,'.PNG' ,'.jpeg' ,'.JPEG']

# Model names
MODEL_NAME = "animal-image-recognition.pt"

# Model paths
PYTORCH_AIR_MODEL_PATH = os.path.join(os.getcwd(),
                                      f"api/models/pytorch/static/{MODEL_NAME}"
                                      )
    
# Classes
classes = ['Cat', 'Cow', 'Dog', 'Elephant', 'Gorilla', 'Hippo', 'Monkey', 'Panda', 'Tiger', 'Zebra']
means = [0.5059, 0.4904, 0.4246]
stds= [0.2292, 0.2269, 0.2292]

IMG_WIDTH = IMG_HEIGHT = 224
  
image_transforms = {
    "training": transforms.Compose([
       transforms.Resize([IMG_WIDTH, IMG_HEIGHT]),
       transforms.RandomRotation(5),
       transforms.RandomHorizontalFlip(.5),
       transforms.ToTensor(),
       transforms.Normalize(mean=means, std=stds, inplace=False)                         
    ]),
    "validation": transforms.Compose([
        transforms.Resize([IMG_WIDTH, IMG_HEIGHT]),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds, inplace=False)
    ]),
    "testing": transforms.Compose([
        transforms.Resize([IMG_WIDTH, IMG_HEIGHT]),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds, inplace=False)
    ])
}

def preprocess_img(img):
    """
    takes in a pillow image and pre process it
    """
    img = image_transforms['testing'](img)
    return img

def predict(model, image, device):
    image = image.unsqueeze(dim=0).to(device)
    preds, _ = model(image)
    preds = F.softmax(preds, dim=1).detach().cpu().numpy().squeeze()
    predicted_label = np.argmax(preds)
    predictions = [
        Prediction(
            label = i,
            class_name = classes[i],
            probability = np.round(preds[i], 2)
        ) for i, _ in enumerate(preds)
    ]
    predicted = Prediction(
        label = predicted_label,
        class_name = classes[predicted_label],
        probability = np.round(preds[predicted_label], 2)
    )
    return Response(
        top_prediction = predicted,
        predictions = predictions
    )