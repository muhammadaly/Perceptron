from human_detection.config import c_yolo_input_size
from common.utils import letterbox_image
from numpy import array, expand_dims
from keras import backend as K
from .measurement_controller import MeasurementController
from .model_interface import ModelInterface


class YoloDetector:
    def __init__(self):
        self.input_size = c_yolo_input_size
        self.sess = K.get_session()
        self.yolo_model , self.boxes, self.scores, self.classes = \
            ModelInterface.load_model()

    def resize_image(self, image):
        """

        :param image:
        :return:
        """
        # if input size of the detector is not initialized
        if self.input_size != (None, None):
            # check if the input size is not divisible bu 32
            assert self.input_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.input_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.input_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        return boxed_image

    def pre_processing(self, image):
        boxed_image = self.resize_image(image)
        image_data = array(boxed_image, dtype='float32')
        image_data /= 255.  # normalize image
        image_data = expand_dims(image_data, 0)  # Add batch dimension.
        return image_data

    # function will input an image and will output bb with class.
    def detect_image(self, input_image):
        processed_image = self.pre_processing(input_image)
        input_image_shape = K.placeholder(shape=(2, ))
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: processed_image,
                input_image_shape: [input_image.size[1], input_image.size[0]],
                K.learning_phase(): 0
            })

        list_measurements = []
        # for object_ind, class_ind in reversed(list(enumerate(out_classes))):
        #     new_measurement = MeasurementController.create_measurement_from_yolo(out_boxes[object_ind],
        #                                                                          class_ind,
        #                                                                          out_scores[object_ind],
        #                                                                          input_image.size)
        #     list_measurements.append(new_measurement)
        return list_measurements
