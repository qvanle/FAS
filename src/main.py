import argparse

from antislc import FAS
from dataset import slcset

def initParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='test', help="test, validate, train, format")
    parser.add_argument('--img', type=str, default='training/', help="image path")
    parser.add_argument('--face_detector', type=str, default='src/weights/yolov8n-face.onnx',
                        help="onnx filepath")
    parser.add_argument('--mask_detector', type=str, default='none')

    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    return parser.parse_args()

if __name__ == '__main__':
    args = initParse()
    
    if (args.action == 'test'):
        test = FAS(args.face_detector, args.mask_detector, args.confThreshold, args.nmsThreshold)
        dt = slcset(args.img, nolabel=True)
        test.testing(dt)
        test.exportLog()

    if (args.action == 'train'): 
        train = FAS(args.face_detector, args.mask_detector, args.confThreshold, args.nmsThreshold)
        dt = slcset(args.img)
        train.training(dt)
        print("Training done") 
        print("Enter the path to save the model: ", end="") 
        train.exportModel(args.mask_detector)
    if (args.action == 'validate'):
        validation = FAS(args.face_detector, "none", args.confThreshold, args.nmsThreshold)
        dt = slcset(args.img)
        validation.validating(dt)

    if (args.action == 'format'):
        pass
