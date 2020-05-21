from face_functions.face_preprocess import preprocess
from insightface.model_zoo import get_model
from cv2 import cvtColor, COLOR_BGR2RGB
from numpy import transpose

class FaceDetector:
    @staticmethod
    def detect(img):
        model = get_model('retinaface_r50_v1')
        model.prepare(ctx_id=-1, nms=0.4)
        bbox, feature_pnts = model.detect(img, threshold=0.5, scale=1.0)
        if bbox.shape[0] == 0:
            return None
        bbox = bbox[0, 0:4]
        feature_pnts = feature_pnts[0, :].reshape((2, 5)).T
        nimg = preprocess(img, bbox, feature_pnts, image_size='112,112')
        nimg = cvtColor(nimg, COLOR_BGR2RGB)
        aligned = transpose(nimg, (2, 0, 1))
        return aligned
