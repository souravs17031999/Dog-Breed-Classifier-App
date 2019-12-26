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
import cv2
from testjson import *

print('Imported packages')


def face_detector(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def resnet_predict(img, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    file = img
    # opening file and convert to RGB
    file = Image.fromarray(img).convert('RGB')
    # defining transforms
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # print('Transforming your image...')
    # transforming the image
    img = transform(file).unsqueeze(0)
    # setting the device if cuda enabled
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = None
    # switching off all dropouts and batchnorm
    model.eval()
    # deactivate autograd engine
    with torch.no_grad():
      # print('Passing your image to the model....')
      out = model(img.to(device))
      ps = torch.exp(out)
      top_p, top_class = ps.topk(1, dim=1)
      index = top_class.item()
    return index # predicted class index

def dog_detector(img_path, model):
    index = resnet_predict(img_path, model)
    return 151 <= index <=268

def load_model(path):
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
    model_transfer.load_state_dict(torch.load(path, map_location = "cpu"))
    return model_transfer

def predict_breed_transfer(model_transfer, file, transform):
    file_mod = Image.fromarray(file).convert('RGB')
    img = transform(file_mod).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_transfer.eval()
    with torch.no_grad():
        out = model_transfer(img.to(device))
        ps = torch.exp(out)
        top_p, top_class = ps.topk(1, dim=1)
        value = top_class.item()
        print(value)
        return class_name(value)


# def display_breed(file, predicted_class):
#     file = Image.open(file).convert('RGB')
#     plt.imshow(file)
#     matplotlib.pyplot.text(5, -50, "Hello and welcome to Face Detector App !",
#     color='black', fontsize=15)
#     plt.title(f'We think this is : {predicted_class}')
#     plt.show()

def run_app(img, model_transfer, model, test_transforms):
    ## handle cases for a human face, dog, and neither
    if (dog_detector(img, model)):
      print("We think you are a dog !")
      # breed_name = predict_breed_transfer(model_transfer, path, test_transforms)
      # display_breed(img_path, breed_name)
      # print("Predicted breed is :", breed_name)
      breed_name = predict_breed_transfer(model_transfer, img, test_transforms)
      print(breed_name)
      return breed_name
    elif (face_detector(img)):
      print("We think you are human !")
      # breed_name_human = predict_breed_transfer(model_transfer, path, test_transforms)
      # display_breed(img_path, breed_name_human)
      # print("You most closely resemble with :", breed_name_human)
      breed_name = predict_breed_transfer(model_transfer, img, test_transforms)
      print(breed_name)
      return breed_name
    else:
      print("We think you are neither human nor dog , might be a alien !")
      # display_breed(img_path, "Alien")
      return "Alien"

def detectAndDisplay():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    cap = cv2.VideoCapture(0)
    model_transfer = load_model('model_transfer.pt')
    print("Model transfer loaded Succesfully")
    test_transforms = torchvision.transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    model = models.resnet50(pretrained=True)
    print("Model normal loaded Succesfully")
    p = 0
    while p < 10:
        ret, img = cap.read()
        name = run_app(img, model_transfer, model, test_transforms)
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # now apply haar detector
        faces = face_cascade.detectMultiScale(gray)
        print(f"faces detected : {faces}")
        # for every detected face
        for (x, y, w, h) in faces:
            # draw a bounding box for every face detected
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            # now crop that part of face where image is detected
            faceROI = gray[y:y+h,x:x+w]
            # detect eyes in the face area
            eyes = eyes_cascade.detectMultiScale(faceROI)
            # for evry eyes
            for (x2,y2,w2,h2) in eyes:
                # calculate the center of detected eye
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                # draw circle with the calculated center
                radius = int(round((w2 + h2)*0.25))
                # associate the circle with the same frame as the rectangle
                cv2.circle(img, eye_center, radius, (255, 0, 0 ), 4)

            cv2.putText(img=img, text=name, org=(00, 185), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (0, 0, 255), thickness = 2)
            cv2.imshow('img',img)
            # Wait for Esc key to stop
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        p += 1
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    detectAndDisplay()
    # if len(sys.argv) > 1:
    #     path = sys.argv[1]
    #     model_transfer = load_model('model_transfer.pt')
    #     print("Model loaded Succesfully")
    #     test_transforms = torchvision.transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    #     ])
    #     run_app(path, model_transfer, path, test_transforms)
    #     print("Thanks for using our system !")
    # else:
    #     print('Path of image not supplied , please supply in form of actual path from terminal !')
