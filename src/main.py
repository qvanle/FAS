import argparse
import cv2

from face_detector import YOLOv8_face

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='images/2.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='weights/yolov8n-face.onnx',
                        help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.45, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    args = parser.parse_args()

    # Initialize YOLOv8_face object detector
    YOLOv8_face_detector = YOLOv8_face(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)
    
    # load image 
    srcimg = cv2.imread(args.imgpath)

    # Detect Objects
    boxes, scores, classids, kpts = YOLOv8_face_detector.detect(srcimg)

    # Draw detections
    dstimg = YOLOv8_face_detector.draw_detections(srcimg, boxes, scores, kpts)
    winName = 'Deep learning face detection use OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

