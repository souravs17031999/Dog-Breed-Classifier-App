# Importing all packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch.utils import data
import torch
from torch import nn
from torch import optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image, ImageFile
import json
from torch.optim import lr_scheduler
import random
import os
import sys
from testjson import *

print('Imported packages')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_transfer = models.resnet50(pretrained=False)
num_ftrs = model_transfer.fc.in_features
out_ftrs = 133
model_transfer.fc = nn.Sequential(nn.Linear(num_ftrs, 512),nn.ReLU(),nn.Linear(512,out_ftrs))
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = torch.optim.Adam(filter(lambda p:p.requires_grad,model_transfer.parameters()) , lr = 0.0001)
model_transfer.to(device);

for name,child in model_transfer.named_children():
  if name in ['layer1', 'layer2', 'layer3', 'layer4', 'fc']:
    #print(name + 'is unfrozen')
    for param in child.parameters():
      param.requires_grad = True
  else:
    #print(name + 'is frozen')
    for param in child.parameters():
      param.requires_grad = False

def load_model(path):
    model_transfer.load_state_dict(torch.load(path, map_location = "cpu"))
    return model_transfer

def inference(model, file, transform):
    file_mod = Image.open(file).convert('RGB')
    img = transform(file_mod).unsqueeze(0)
    print('Transforming your image...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_transfer.eval()
    with torch.no_grad():
        print('Passing your image to the model....')
        out = model_transfer(img.to(device))
        ps = torch.exp(out)
        top_p, top_class = ps.topk(1, dim=1)
        value = top_class.item()
        print(value)
        predicted_class = class_name(value)
        display_breed(file, predicted_class)

def display_breed(file, predicted_class):
    file = Image.open(file).convert('RGB')
    plt.imshow(file)
    matplotlib.pyplot.text(5, -50, "Hello and welcome to Face Detector App !",
    color='black', fontsize=15)
    plt.title(f'We think this is : {predicted_class}')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
        model_transfer = load_model('model_transfer.pt')
        print("Model loaded Succesfully")
        test_transforms = torchvision.transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        inference(model_transfer, path, test_transforms)
        print("Thanks for using our system !")
    else:
        print('Path of image not supplied , please supply in form of actual path from terminal !')
