from human_detection.detector import YoloDetector
from PIL import Image

if __name__ == '__main__':
    detector = YoloDetector()
    img_path = '/home/muhammadaly/dataset/random/1.jpg'
    img = Image.open(img_path)
    m = detector.detect(img)
    print(m)