import sys

sys.path.append(".")

from pathlib import Path

import torch
import pandas as pd
from src.embeddings import EmbeddingSearch
from src.utils import open_img

IMG_DIR = Path("data/img")

emb = EmbeddingSearch()

metadata = pd.read_csv(IMG_DIR / "metadata.csv")
embeddings = []

for i, row in metadata.iterrows():
    img = open_img(IMG_DIR / row["img_path"])
    embedding = emb.embed(img.expand(1, -1, -1, -1)).squeeze()
    embeddings.append(embedding)

embeddings = torch.stack(embeddings)
print(embeddings.shape)

torch.save(embeddings, "embeddings.pt")
