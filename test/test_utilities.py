from cv2 import imread, imwrite, imshow, waitKey, resize
from face_functions.face_detection import FaceDetector


if __name__ == '__main__':
    i_img = imread('C:\\Users\\Muhammad\\workspace\\data\\brad_bit.png')
    faces, _ = FaceDetector.detect(i_img)
    for face in faces.astype(int):
        img = i_img[face[1]:face[3], face[0]:face[2]]
        # if img.shape != (112, 112):
        #     img = resize(img, (112, 112))
        print(img.shape)
        imshow('result', img)
        waitKey()
        imwrite('C:\\Users\\Muhammad\\workspace\\data\\1.0.png', img)

