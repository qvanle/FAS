import os
import cv2
from numpy import array 
from datetime import datetime
import numpy as np 
import matplotlib.pyplot as pyplot 
from face_detector import YOLOv8_face
from PIL import Image

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as torchfunc
from torch.utils.data import DataLoader 
from transformers import ViTForImageClassification
from transformers import ViTFeatureExtractor
from torchvision import transforms
from tqdm import tqdm 
import timm

class FAS: 
    def __init__(self, faceDectector, maskDetector, confThreshold, nmsThreshold):  
        
        self.faceConf = confThreshold

        self.face_detector = YOLOv8_face(faceDectector, confThreshold, nmsThreshold)
        
        self.maskDetector = None
        self.optimizer = None 
        self.criterion = None 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = 3

        if maskDetector == "none": 
            self.maskDetector = timm.create_model("vit_base_patch16_224_dino")
            self.maskDetector.to(self.device)
            self.optimizer = optim.Adam(self.maskDetector.parameters(), lr=3e-5)
            self.criterion = nn.CrossEntropyLoss()

            self.exportModel()
        else: 
            # load from maskDetector, which is a path 
            self.maskDetector = timm.create_model("vit_base_patch16_224_dino")
            state_dick = torch.load(maskDetector)
            self.maskDetector.load_state_dict(state_dick)
            self.maskDetector.to(self.device)
            self.optimizer = optim.Adam(self.maskDetector.parameters(), lr=3e-5)
            self.criterion = nn.CrossEntropyLoss()
        self.q = []
    
    def exportModel(self, path = "src/weights/mask_detector.pth"): 
        if(path == "none"): 
            path = "src/weights/mask_detector.pth"
        torch.save(self.maskDetector.state_dict(), path)

    def validating(self, dts): 

        timeStart = datetime.now()

        print("Starting validation: " + str(timeStart)) 

        self.maskDetector.eval() 
        
        classCorrect = {cls : 0 for cls in dts.classes}
        total_predictions = 0
        correct_predictions = 0
        error_predictions = 0

        for i in tqdm(range(len(dts)), total = len(dts), desc = "Validation: "): 
            srcimg, label, path = dts[i]
            blobs = self.face_detector.detect(srcimg)
            boxes, scores, classids, kpts = blobs

            total_predictions += 1
            
            predictedList = [] 
                
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                face = srcimg[y1:y2, x1:x2]
                face = cv2.resize(face, (224, 224))
                face = Image.fromarray(face)
                face = transforms.ToTensor()(face).unsqueeze(0)
                face = face.to(self.device)

                outputs = self.maskDetector(face)
                _, predicted = torch.max(outputs, 1)

                if(predicted == label):
                    correct_predictions += 1
                    predictedList.append(1)
                else:
                    predictedList.append(0)

            if(sum(predictedList) == len(boxes)):
                classCorrect[dts.classes[label]] += 1
            elif len(predictedList) == 0: 
                error_predictions += 1

        print("Validation done: " + str(datetime.now())) 
        print("Correct predictions: " + str(correct_predictions) + " Total predictions: " + str(total_predictions)) 
        print("Accuracy: " + str(correct_predictions/total_predictions)) 
        print("Error predictions: " + str(error_predictions)) 
        print(str(classCorrect))

    def training(self, dts):
        
        timeStart = datetime.now() 

        print("Starting training: " + str(timeStart)) 

        for epoch in range(self.epoch): 
            print("Epoch: " + str(epoch)) 

            self.maskDetector.train() 
            running_loss = 0.0
            correct_predictions = 0 
            total_predictions = 0
            
            NoFaceInImage = 0
            MultiFaceInImage = 0

            for i in tqdm(range(len(dts)), total = len(dts), desc = "Epoch " + str(epoch) + " "): 
                
                srcimg, label, path = dts[i]

                blobs = self.face_detector.detect(srcimg)
                boxes, scores, classids, kpts = blobs
                
                faceCount = 0

                for i in range(len(boxes)): 
                    
                    if (scores[i] < self.faceConf):
                        continue 

                    x1, y1, x2, y2 = boxes[i]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    face = srcimg[y1:y2, x1:x2]
                    
                    if(face.shape[0] <= 1 or face.shape[1] <= 1):
                        continue
                    
                    faceCount += 1

                    face = cv2.resize(face, (224, 224))
                    face = Image.fromarray(face)
                    face = transforms.ToTensor()(face).unsqueeze(0)
                    face = face.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.maskDetector(face)
                    loss = self.criterion(outputs, torch.tensor([label]).to(self.device))
                    loss.backward()
                    self.optimizer.step()

                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)

                    total_predictions += 1
                    if(predicted == label):
                        correct_predictions += 1
                    
                if(faceCount == 0): 
                    NoFaceInImage += 1 
                elif(faceCount > 1):
                    MultiFaceInImage += 1
                
            print("Epoch: " + str(epoch) + " Loss: " + str(running_loss))
            print("Total predictions: " + str(total_predictions) + " Correct predictions: " + str(correct_predictions))
            print("Accuracy: " + str(correct_predictions/total_predictions))
            print("No face in image: " + str(NoFaceInImage) + " Multiple faces in image: " + str(MultiFaceInImage))
            
        
        timeEnd = datetime.now()
        print("Training done: " + str(timeEnd))
        print("Time taken: " + str(timeEnd - timeStart))
