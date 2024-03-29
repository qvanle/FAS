import argparse

from antislc import FAS
from dataset import slcset

def initParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='test', help="test, validate, train")
    parser.add_argument('--img', type=str, default='testing/', help="image path")
    parser.add_argument('--face_detector', type=str, default='src/weights/yolov8n-face.onnx',
                        help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    return parser.parse_args()

if __name__ == '__main__':
    args = initParse()
    
    if (args.action == 'test'):
        test = FAS(args.face_detector, args.confThreshold, args.nmsThreshold)
        dt = slcset(args.img, nolabel=True)
        test.testing(dt)
