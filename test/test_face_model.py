from face_functions.face_model import FaceModel
import argparse
import cv2
import sys
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='C:\\Users\\Muhammad/.insightface/models\\arcface_r100_v1\\model,0', help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=-1, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    args = parser.parse_args()

    model = FaceModel(args)
    img0_0 = cv2.imread('C:\\Users\\Muhammad\\workspace\\data\\0.0.jpg')
    img0_0 = model.get_input(img0_0)
    f0_0 = model.get_feature(img0_0)

    img0_1 = cv2.imread('C:\\Users\\Muhammad\\workspace\\data\\0.1.png')
    img0_1 = model.get_input(img0_1)
    f0_1 = model.get_feature(img0_1)

    img1_0 = cv2.imread('C:\\Users\\Muhammad\\workspace\\data\\1.0.png')
    img1_0 = model.get_input(img1_0)
    f1_0 = model.get_feature(img1_0)

    img1_2 = cv2.imread('C:\\Users\\Muhammad\\workspace\\data\\1.2.jpg')
    img1_2 = model.get_input(img1_2)
    f1_2 = model.get_feature(img1_2)

    img2_0 = cv2.imread('C:\\Users\\Muhammad\\workspace\\data\\2.0.jpg')
    img2_0 = model.get_input(img2_0)
    f2_0 = model.get_feature(img2_0)

    dist = np.sum(np.square(f0_0-f0_1))
    print('tom hanks 0 and tom hanks 1 dist ', dist)
    sim = np.dot(f0_0, f0_1.T)
    print('tom hanks 0 and tom hanks 1 sim ', sim)

    dist = np.sum(np.square(f0_0-f1_0))
    print('tom hanks 0 and brad pitt 0 dist ', dist)
    sim = np.dot(f0_0, f1_0.T)
    print('tom hanks 0 and brad pitt 0 sim ', sim)

    dist = np.sum(np.square(f1_0-f1_2))
    print('brad pitt 0 and brad pitt 2 dist ', dist)
    sim = np.dot(f1_0, f1_2.T)
    print('brad pitt 0 and brad pitt 2 sim ', sim)

    dist = np.sum(np.square(f0_0-f2_0))
    print('tom hanks 0 and will smith 0 dist ', dist)
    sim = np.dot(f0_0, f2_0.T)
    print('tom hanks 0 and will smith 0 sim ', sim)
