from face_functions.face_detection import FaceDetector
from face_functions.face_recognition import FaceRecognizer
from cv2 import resize


class FaceAuthenticator:
    def authenticate(self, i_img):
        face_boxes_list, _ = FaceDetector.detect(i_img)
        for face_box in face_boxes_list:
            face_img = i_img[face_box[0]:face_box[1], face_box[2]:face_box[3]]
            if face_img.shape != (112, 112):
                face_img = resize(face_img, (112, 112))
                emb0 = FaceRecognizer.recognize(face_img).ravel()



