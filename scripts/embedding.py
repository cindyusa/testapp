from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class EmbeddingModel:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings.cpu().detach().numpy()

    def initialize_faiss_index(self, dimension):
        index = faiss.IndexFlatL2(dimension)
        return index

    def add_to_index(self, index, embeddings):
        index.add(embeddings)