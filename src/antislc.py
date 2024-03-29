import os
import cv2
from numpy import array 
from datetime import datetime
from face_detector import YOLOv8_face

class FAS: 
    def __init__(self, faceDectector, confThreshold, nmsThreshold):  
        self.face_detector = YOLOv8_face(faceDectector, confThreshold, nmsThreshold)
        
        self.q = []
        self.blobs = []
        self.log = {}
    
    def validation(self, dts): 
        pass

    def testing(self, dts):
        self.blobs = []
        self.log = {}

        timeStart = datetime.now()

        print("Starting testing: " + str(timeStart))

        for i in range(len(dts)): 
            srcimg, label, path = dts[i]
            boxes, scores, classids, kpts = self.face_detector.detect(srcimg)
            self.blobs.append((path, boxes, scores, classids, kpts))
            
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
    
