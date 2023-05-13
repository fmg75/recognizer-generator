import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle


class FaceNetModels:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained="vggface2").eval()
        # self.model = torch.jit.load('model_resnet.pt')
        self.mtcnn = MTCNN(min_face_size=50, keep_all=False)
        with open("./caracteristicas/caracteristicas_ID.pkl", "rb") as f:
            self.caracteristicas = pickle.load(f)

    def embedding(self, img_tensor):
        img_embedding = self.model(img_tensor.unsqueeze(0))
        return img_embedding

    def Distancia(self, img_embedding):
        distances = [
            (label, torch.dist(emb, img_embedding))
            for label, emb in self.caracteristicas.items()
        ]
        sorted_distances = sorted(distances, key=lambda x: x[1])
        return sorted_distances[0][0], sorted_distances[0][1].item()
