import google.generativeai as genai
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def embed_chunks(chunks):
    embeddings = []

    for i, chunk in enumerate(chunks):
        #skip empty chunks
        if not chunk.strip():
            embeddings.append(np.zeros(768))  # Assuming 768 is the embedding size
            continue

        chunk = chunk[:3000]

        try:
            res = genai.generate_embeddings(
                model = "models/text-embedding-3",
                text = chunk,
                task_type = "retrieval_document"
            )
            embeddings.append(res['embedding'])
        except Exception as e:
            print(f"Error generating embedding for chunk {i}: {e}")
            embeddings.append(np.zeros(768)) #fallback zero vector

        return np.array(embeddings)

def embed_query(query):
    if not query.strip():
        return np.zeros(768)
    
    try:
        res = genai.generate_embeddings(
            model = "models/text-embedding-3",
            text = query,
            task_type = "retrieval_query"
        )
        return np.array(res['embedding'])
    except Exception as e:
        print(f"Error generating embedding for query: {e}")
        return np.zeros(768)

def retrive_similar_chunks(query,chunks, chunk_embeddings, top_k = 5):
    query_embed = embed_query(query)
    similarities = cosine_similarity([query_embed], chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]#comes in ascending order, so we reverve it
    return top_indices, similarities[top_indices]