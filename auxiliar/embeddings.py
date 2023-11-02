import torch
from torchvision.models import Inception_V3_Weights

class EmbeddingSearch:

    def __init__(self) -> None:
        self.model = self._build_model()
        self.features = {}
        self._load_embeddings_store()

    def _build_model(self):            
        model = torch.hub.load(
            'pytorch/vision:v0.10.0',
            'inception_v3',
            weights=Inception_V3_Weights.IMAGENET1K_V1,
        )
        model.eval()
        return model

    def get_features(self, name):
        def hook(model, input, output):
            self.features[name] = output.detach()
        return hook

    def embed(self, img):
        self.model.avgpool.register_forward_hook(self.get_features('avgpool'))
        _ = self.model(img)
        embedding = self.features['avgpool'].squeeze()
        embedding = embedding / embedding.norm()
        return embedding

    def _load_embeddings_store(self):
        self._embedding_store = torch.load('embeddings.pt')

    def look_up(self, img):
        img_embedding = self.embed(img.expand(1, -1, -1, -1)).squeeze()
        dot_products = self._embedding_store.mv(img_embedding)
        return dot_products.argmax().squeeze().item()
