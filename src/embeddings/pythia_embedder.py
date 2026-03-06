import torch
from transformers import AutoTokenizer, AutoModel


class PythiaEmbedder:

    def __init__(self, model_name="EleutherAI/pythia-160m"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()


    def encode(self, texts, batch_size=16):

        embeddings = []

        for i in range(0, len(texts), batch_size):

            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            hidden_states = outputs.last_hidden_state

            batch_embeddings = hidden_states.mean(dim=1)

            embeddings.append(batch_embeddings.cpu())

        embeddings = torch.cat(embeddings)

        return embeddings.numpy()