from cv2 import imread, imshow, waitKey
import os
from face_functions.face_model import FaceModel
import argparse
from progress.bar import IncrementalBar
import numpy as np
import mxnet

def init(args):
    image_list_file = r"D:\dataset\data\facescrub_lst"
    folder_root = r"D:\dataset\data\facescrub_images"

    with open(image_list_file) as f:
        lines = f.read().splitlines()
        model = FaceModel(args)
        feature_list = np.zeros(shape=(len(lines), 512), dtype=np.float32)
        text_file = open(r"D:\dataset\data\facescrub_feature_lst_processed.txt", "w")
        ind = 0
        for img_name in lines:
            img = imread(os.path.abspath(folder_root + '\\' + img_name))
            try:
                img = model.get_input(img)
                feature_list[ind, :] = model.get_feature(img)
                ind += 1
                print(ind)
                text_file.write(img_name)
            except mxnet.base.MXNetError:
                print("error with ", img_name)

        # np.save(r"D:\dataset\data\facescrub_feature_lst", feature_list)
        np.savetxt(r"D:\dataset\data\facescrub_feature_lst.csv", feature_list, delimiter=",")
        text_file.close()


def read_features():
    # features = np.fromfile(r"D:\dataset\data\facescrub_feature_lst.npy", dtype=np.float32).reshape()
    text_file = open(r"D:\dataset\data\facescrub_feature_lst_processed.txt", "r")
    class_names = text_file.readlines()
    print(len(class_names))
    features = np.loadtxt(r"D:\dataset\data\facescrub_feature_lst.csv", delimiter=',')
    features = features[:3519, :]
    print(features.shape)


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
    # init(args)
    read_features()
