import torch
from sentence_transformers import SentenceTransformer


class SBERTEmbedder:

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(model_name, device=self.device)

        self.model.eval()


    def encode(self, texts, batch_size=16):

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        return embeddings