from face_functions.face_detection import FaceDetector
from face_functions.face_recognition import FaceRecognizer
from cv2 import imread, resize


if __name__ == '__main__':
    tom_hanks_base_image = imread('C:\\Users\\Muhammad\\workspace\\data\\0.0.png')
    tom_hanks_face_emb = FaceRecognizer.recognize(tom_hanks_base_image).ravel()
    tom_hanks_base_image = imread('C:\\Users\\Muhammad\\workspace\\data\\0.0.png')
    tom_hanks_face_emb = FaceRecognizer.recognize(tom_hanks_base_image).ravel()




