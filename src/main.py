import argparse

from antislc import FAS
from dataset import slcset

def initParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='test', help="test, validate, train, format")
    
    parser.add_argument('--img', type=str, default='training/', help="image path")
    
    parser.add_argument('--face_detector', type=str, default='src/weights/yolov8n-face.onnx')
    
    parser.add_argument('--mask_detector', type=str, default='none')

    parser.add_argument('--confThreshold', default=0.7, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
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
