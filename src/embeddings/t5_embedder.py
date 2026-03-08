import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


class T5Embedder:

    def __init__(self, model_name="t5-small"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

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

                encoder_outputs = self.model.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )

                encoder_hidden = encoder_outputs.last_hidden_state

                # decoder start token
                decoder_input_ids = torch.full(
                    (encoder_hidden.size(0), 1),
                    self.model.config.decoder_start_token_id,
                    dtype=torch.long,
                    device=self.device
                )

                decoder_outputs = self.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=inputs["attention_mask"]
                )

                decoder_hidden = decoder_outputs.last_hidden_state

            encoder_embed = encoder_hidden.mean(dim=1)
            decoder_embed = decoder_hidden.mean(dim=1)

            batch_embeddings = torch.cat([encoder_embed, decoder_embed], dim=1)

            embeddings.append(batch_embeddings.cpu())

        embeddings = torch.cat(embeddings)

        return embeddings.numpy()