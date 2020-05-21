from face_functions.face_detection import FaceDetector
from cv2 import imread, rectangle, imshow, waitKey, circle

def test_bbox():
    img = imread('C:\\Users\\Muhammad\\workspace\\data\\1.0.png')
    bbox, landmark = FaceDetector.detect(img)
    print(bbox)

    color = (255, 0, 0)
    c_color = (0, 255, 0)
    thickness = 2
    radius = 2
    for fbbox in bbox:
        img = rectangle(img, (int(fbbox[0]), int(fbbox[1])), (int(fbbox[2]), int(fbbox[3])), color, thickness)
    for face in landmark:
        for f_landmark in face:
            img = circle(img, (int(f_landmark[0]), int(f_landmark[1])), radius, color, thickness)
    imshow('result', img)
    waitKey()

def test_image():
    img = imread('C:\\Users\\Muhammad\\workspace\\data\\0.0.jpg')
    img = FaceDetector.detect(img)
    imshow('result', img)
    waitKey()

if __name__ == '__main__':
    test_image()



