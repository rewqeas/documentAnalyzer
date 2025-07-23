import faiss #pip install faiss-cpu
import numpy as np
import os
import pickle



INDEX_PATH = "vector_store.index"
METADATA_PATH = "vector_store_metadata.pkl"


def save_vector_store(index, metadata):
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

def load_vector_store():
    if not os.path.exists(INDEX_PATH):
        return None, []
    
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata  

def build_vector_store(embeddings, texts):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    embedding_matrix = np.array(embeddings).astype('float32')
    index.add(embedding_matrix)

    metadata = [{"text": text} for text in texts]

    return index, metadata
