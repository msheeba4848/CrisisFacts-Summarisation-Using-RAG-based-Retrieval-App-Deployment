from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np

if torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)


def compute_query_embedding(texts, batch_size=32):
    """Compute dense embeddings in batches on CUDA to avoid memory issues."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Move to CPU before converting to numpy
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)
