from face_functions.face_recognition import FaceRecognizer
from face_functions.face_detection import FaceDetector

from cv2 import imread, resize, imshow, waitKey
from numpy import square, sum, dot

if __name__ == '__main__':
    FaceRecognizer = FaceRecognizer()
    img0 = imread('C:\\Users\\Muhammad\\workspace\\data\\0.0.png')
    img1 = imread('C:\\Users\\Muhammad\\workspace\\data\\0.1.jpg')
    img2 = imread('C:\\Users\\Muhammad\\workspace\\data\\1.0.png')
    img0 = FaceDetector.detect(img0)
    img1 = FaceDetector.detect(img1)
    img2 = FaceDetector.detect(img2)

    # f0 = FaceRecognizer.recognize(img0).ravel()
    # f1 = FaceRecognizer.recognize(img1).ravel()
    # f2 = FaceRecognizer.recognize(img2).ravel()
    #
    # dist = sum(square(f0 - f0))
    # sim = dot(f0, f0.T)
    # print('f0-f0 ', dist)
    # print('f0-f0 ', sim)
    #
    # dist = sum(square(f0 - f1))
    # sim = dot(f0, f1.T)
    # print('f0-f1 ', dist)
    # print('f0-f1 ', sim)
    #
    # dist = sum(square(f0 - f2))
    # sim = dot(f0, f2.T)
    # print('f0-f2 ', dist)
    # print('f0-f2 ', sim)

    sim = FaceRecognizer.compute_sim(img0, img0)
    print('f0-f0 ', sim)
    sim = FaceRecognizer.compute_sim(img0, img1)
    print('f0-f1 ', sim)
    sim = FaceRecognizer.compute_sim(img0, img2)
    print('f0-f2 ', sim)



