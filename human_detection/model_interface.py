from .config import *
from os.path import expanduser
from keras.models import load_model
from .model import yolo_eval, yolo_body, tiny_yolo_body
from keras.layers import Input
from numpy import random, asarray
import colorsys
from keras import backend as K
from keras.utils import multi_gpu_model
from timeit import default_timer as timer
from PIL import Image


class ModelInterface:
    @staticmethod
    def load_model():
        model_path = expanduser(c_model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        is_tiny_version = (c_model_anchors == c_tiny_yolo_number_of_anchors)
        try:
            yolo_model = load_model(model_path, compile=False)
        except:
            yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), c_num_anchors//2, c_num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), c_num_anchors//3, c_num_classes)
            yolo_model.load_weights(model_path)
        else:
            assert yolo_model.layers[-1].output_shape[-1] == \
                c_num_anchors/len(yolo_model.output) * (c_num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(c_class_names), 1., 1.)
                      for x in range(len(c_class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        input_image_shape = K.placeholder(shape=(2, ))
        if c_gpu_num >= 2:
            yolo_model = multi_gpu_model(yolo_model, gpus=c_gpu_num)
        boxes, scores, classes = yolo_eval(yolo_model.output, c_model_anchors, len(c_class_names),
                                           input_image_shape, score_threshold=c_score,
                                           iou_threshold=c_iou)
        return boxes, scores, classes

    @staticmethod
    def detect_video(yolo, video_path, output_path=""):
        import cv2
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != "" else False
        if isOutput:
            # print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            return_value, frame = vid.read()
            image = Image.fromarray(frame)
            image = yolo.detect_image(image)
            result = asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        yolo.close_session()
