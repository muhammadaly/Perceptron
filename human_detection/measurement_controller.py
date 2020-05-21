# from human_tracking.measurement import  ClassificationTypes, Measurement
# from human_tracking.config import S_X, S_Y
from numpy import floor


class MeasurementController:
    # @staticmethod
    # def _map_from_yolo_to_platform_classification(yolo_class):
    #     if yolo_class == 0:
    #         return ClassificationTypes.Human
    #     elif yolo_class == 1:
    #         return ClassificationTypes.Animal
    #     else:
    #         return ClassificationTypes.Unknown

    @staticmethod
    def _get_yolo_box_center_point(yolo_box, image_size):
        top, left, bottom, right = yolo_box
        top = max(0, floor(top + 0.5).astype('int32'))
        left = max(0, floor(left + 0.5).astype('int32'))
        bottom = min(image_size[0], floor(bottom + 0.5).astype('int32'))
        right = min(image_size[0], floor(right + 0.5).astype('int32'))

        center_point_x = bottom + int((top - bottom) / 2.)
        center_point_y = left + int((right - left) / 2.)
        return center_point_x, center_point_y

    # @staticmethod
    # def create_measurement_from_yolo(yolo_box, yolo_class, yolo_score, image_size):
    #     new_measurement = Measurement()
    #     new_measurement.existence_probability = yolo_score
    #     new_measurement.classification = MeasurementController._map_from_yolo_to_platform_classification(yolo_class)
    #     new_measurement.state.mean[S_X], new_measurement.state.mean[S_Y] = \
    #         MeasurementController._get_yolo_box_center_point(yolo_box, image_size)
    #     return new_measurement
