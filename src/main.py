import argparse

from antislc import FAS
from dataset import slcset

def initParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, required=True)
    
    parser.add_argument('--img', type=str, required=True)
    
    parser.add_argument('--face_detector', type=str, required=True)
    # src/weights/yolov8n-face.onnx

    parser.add_argument('--mask_detector', type=str, required=True)
    # src/weights/mask_detector.pth  
    # none
    parser.add_argument('--confThreshold', type=float, required=True)
    # 0.7 
    parser.add_argument('--nmsThreshold', type=float, required=True)
    # 0.5 
    return parser.parse_args()

if __name__ == '__main__':
    args = initParse()
    
    model = FAS(args.face_detector, args.mask_detector, args.confThreshold, args.nmsThreshold)

    if (args.action == 'train'): 
        dt = slcset(args.img)
        model.training(dt)
        model.exportModel(args.mask_detector)

    if (args.action == 'validate'):
        dt = slcset(args.img)
        model.validating(dt)

    if (args.action == 'format'):
        pass
