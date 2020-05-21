from numpy import array
# model
c_model_path = "C:\\Users\\Muhammad\\workspace\\Perceptron\\human_detection\\model\\yolo-tiny.h5"

c_yolo_input_size = (416, 416)

tiny_model_anchors = array([[10, 14],
                      [23, 27],
                      [37, 58],
                      [81, 82],
                      [135, 169],
                      [344, 319]])

c_class_names = ['aeroplane',
                 'bicycle',
                 'bird',
                 'boat',
                 'bottle',
                 'bus',
                 'car',
                 'cat',
                 'chair',
                 'cow',
                 'diningtable',
                 'dog',
                 'horse',
                 'motorbike',
                 'person',
                 'pottedplant',
                 'sheep',
                 'sofa',
                 'train',
                 'tvmonitor']

c_model_anchors = tiny_model_anchors
c_tiny_yolo_number_of_anchors = 6
c_num_anchors = c_model_anchors.shape[0]
c_num_classes = len(c_class_names)
c_gpu_num = 1
c_score = 0.3
c_iou =  0.45
