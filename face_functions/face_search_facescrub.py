import numpy as np
import argparse

from face_functions.face_model import FaceModel
import cv2


class FaceAuthenticator:
    def __init__(self):
        self.n_feature_lst = 3519
        text_file = open(r"D:\dataset\data\facescrub_feature_lst_processed.txt", "r")
        self.feature_class_names = [item.split("/")[0] for item in text_file.readlines()]
        self.features_lst = np.loadtxt(r"D:\dataset\data\facescrub_feature_lst.csv", delimiter=',')
        self.features_lst = self.features_lst[:self.n_feature_lst, :]
        parser = argparse.ArgumentParser(description='face model test')
        parser.add_argument('--image-size', default='112,112', help='')
        parser.add_argument('--model',
                            default=r"C:\Users\Muhammad\workspace\Perceptron\face_functions\arcface-model\model-r100-ii\model,0",
                            help='path to load model.')
        parser.add_argument('--ga-model', default='', help='path to load model.')
        parser.add_argument('--gpu', default=-1, type=int, help='gpu id')
        parser.add_argument('--det', default=0, type=int,
                            help='mtcnn option, 1 means using R+O, 0 means detect from begining')
        parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
        parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
        args = parser.parse_args()
        self.model = FaceModel(args)

        self. color = (255, 0, 0)
        self.txt_color = (36, 255, 12)
        self.thickness = 2

    def search(self, i_img):
        faces_bboxes, faces_points = self.model.detect_faces(i_img)
        current_face_bbox = faces_bboxes[0, 0:4]
        current_face_points = faces_points[0, :].reshape((2, 5)).T
        aligned_img = self.model.get_input(i_img, current_face_bbox, current_face_points)
        i_feature = self.model.get_feature(aligned_img).reshape(1, -1)
        # i_n_feature = np.repeat(i_feature, [self.n_feature_lst], axis=0)
        # dist = np.sum(np.square(i_n_feature - self.features_lst), axis=0)
        sim = np.dot(self.features_lst, i_feature.T).ravel()
        idx = np.argsort(sim)

        x = int(current_face_bbox[0])
        y = int(current_face_bbox[1])
        x_end = int(current_face_bbox[2])
        y_end = int(current_face_bbox[3])
        out_img = cv2.rectangle(i_img, (x, y), (x_end, y_end), self.color, self.thickness)
        cv2.putText(out_img, self.feature_class_names[idx[-1]], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=self.txt_color, thickness=self.thickness)
        # cv2.imshow('result', out_img)
        # cv2.waitKey()
        return out_img

    def search_mutiple(self, i_img):
        faces_bboxes, faces_points = self.model.detect_faces(i_img)
        for current_ind in range(faces_bboxes.shape[0]):
            current_face_bbox = faces_bboxes[current_ind]
            current_face_points = faces_points[current_ind]
            aligned_img = self.model.get_input(i_img, current_face_bbox, current_face_points)
            i_feature = self.model.get_feature(aligned_img).reshape(1, -1)
            # i_n_feature = np.repeat(i_feature, [self.n_feature_lst], axis=0)
            # dist = np.sum(np.square(i_n_feature - self.features_lst), axis=0)
            sim = np.dot(self.features_lst, i_feature.T).ravel()
            idx = np.argsort(sim)

            x = int(current_face_bbox[0])
            y = int(current_face_bbox[1])
            x_end = int(current_face_bbox[2])
            y_end = int(current_face_bbox[3])
            out_img = cv2.rectangle(i_img, (x, y), (x_end, y_end), self.color, self.thickness)
            cv2.putText(out_img, self.feature_class_names[idx[-1]], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=self.txt_color, thickness=self.thickness)
        # cv2.imshow('result', out_img)
        # cv2.waitKey()
        return out_img
