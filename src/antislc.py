import os
import cv2
from numpy import array 
from datetime import datetime
import numpy as np 
import matplotlib.pyplot as pyplot 
from face_detector import YOLOv8_face

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as torchfunc
from torch.utils.data import DataLoader 
from transformers import ViTForImageClassification
from transformers import ViTFeatureExtractor
from torchvision import transforms
from tqdm import tqdm 

class FAS: 
    def __init__(self, faceDectector, maskDetector, confThreshold, nmsThreshold):  

        self.face_detector = YOLOv8_face(faceDectector, confThreshold, nmsThreshold)
        
        self.maskDetector = None
        self.optimizer = None 
        self.criterion = None 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = 10

        if maskDetector == "none": 
            self.maskDetector = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
            self.maskDetector.to(self.device)
            self.optimizer = optim.Adam(self.maskDetector.parameters(), lr=3e-5)
            self.criterion = nn.CrossEntropyLoss()
        else: 
            # load from maskDetector, which is a path 
            self.maskDetector = ViTForImageClassification.from_pretrained(maskDetector)
            self.maskDetector.to(self.device)
            self.optimizer = optim.Adam(self.maskDetector.parameters(), lr=3e-5)
            self.criterion = nn.CrossEntropyLoss()
        self.q = []
        self.log = {}
    
    def exportLog(self, path = "log.txt"): 
        with open(path, 'w') as f: 
            for i in self.log: 
                f.write(str(i)) 
                f.write(": ")
                f.write(str(self.log[i]))
                f.write("\n")
    def exportModel(self, path = "src/weights/mask_detector.onnx"): 
        # save as format onnx 
        torch.onnx.export(self.maskDetector, torch.randn(1, 3, 224, 224), path) 

    def validating(self, dts): 
        self.log = {}

        timeStart = datetime.now()

        print("Starting validation: " + str(timeStart)) 

        for epoch in range(self.epoch): 
            print("Epoch: " + str(epoch)) 
            self.maskDetector.train() 
        pass
    def training(self, dts):
        self.log = {}
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        timeStart = datetime.now() 

        print("Starting training: " + str(timeStart)) 

        for epoch in range(self.epoch): 
            print("Epoch: " + str(epoch)) 
            self.maskDetector.train() 
            running_loss = 0.0
            correct_predictions = 0 
            total_predictions = 0
            
            faceFailed = 0

            for i in range(len(dts)): 
                srcimg, label, path = dts[i]
                print("--> Training: " + path) 

                blobs = self.face_detector.detect(srcimg)
                boxes, scores, classids, kpts = blobs
                #print("Found " + str(len(boxes)) + " faces")
                for i in range(len(boxes)): 
                    x1, y1, x2, y2 = boxes[i]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    face = srcimg[y1:y2, x1:x2]
                    if(face.shape[0] <= 1 or face.shape[1] <= 1):
                        faceFailed += 1
                        continue
                    
                    face = torch.from_numpy(face.transpose(2, 0, 1)).float().to(self.device)   
                    self.optimizer.zero_grad() 
                    outputs = self.maskDetector(face)
                    loss = self.criterion(outputs, torch.tensor([label])) 
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total_predictions += 1
                    if(predicted == label):
                        correct_predictions += 1
                print("")
            print("Epoch: " + str(epoch) + " Loss: " + str(running_loss) + " Accuracy: " + str(correct_predictions/total_predictions))
            print("Failed: " + str(faceFailed) + " faces")
            
        
        timeEnd = datetime.now()
        print("Training done: " + str(timeEnd))
        print("Time taken: " + str(timeEnd - timeStart))

        self.log['timeStart'] = timeStart
        self.log['elapse'] = timeEnd - timeStart
        self.log['total'] = len(dts)
        
        exportModel()



    def testing(self, dts):
        self.log = {}

        timeStart = datetime.now()

        print("Starting testing: " + str(timeStart))

        for i in range(len(dts)): 
            srcimg, label, path = dts[i]
            print("--> Testing: " + path)

            blobs = self.face_detector.detect(srcimg)
            boxes, scores, classids, kpts = blobs
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                face = srcimg[y1:y2, x1:x2]
                face = cv2.resize(face, (224, 224))

        '''
            dstimg = self.face_detector.draw_detections(srcimg, boxes, scores, kpts)
            winName = 'Deep learning face detection use OpenCV'
            cv2.namedWindow(winName, 0)
            cv2.imshow(winName, dstimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        '''

        timeEnd = datetime.now()
        print("Testing done: " + str(timeEnd))
        print("Time taken: " + str(timeEnd - timeStart))

        self.log['timeStart'] = timeStart
        self.log['elapse'] = timeEnd - timeStart
        self.log['total'] = len(dts)
    
