from  insightface.model_zoo import get_model
from sklearn.preprocessing import normalize


class FaceRecognizer:
    def __init__(self):
        self.model = get_model('arcface_r100_v1')
        self.model.prepare(ctx_id=-1)

    def recognize(self, img):
        embeddings = self.model.get_embedding(img)
        embeddings = normalize(embeddings).flatten()
        return embeddings

    def compute_sim(self, img1, img2):
        return self.model.compute_sim(img1, img2)

