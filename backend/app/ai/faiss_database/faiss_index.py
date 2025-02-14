import faiss
import numpy as np
dimension = 1024  # CLIP embeddings are 1024-dimensional for laion/CLIP-ViT-H-14-laion2B-s32B-b79K




image_index = faiss.IndexFlatIP(dimension)  # L2 distance for image embeddings
text_index = faiss.IndexFlatIP(dimension)  # L2 distance for text embeddings
tag_index = faiss.IndexFlatIP(dimension)
# Store image embeddings and their IDs
image_embeddings = []
image_ids = []
text_embeddings = []
text_ids = []
image_descriptions = {}

def normalize_embedding(embedding):
    """ Normalize embedding to unit length for cosine similarity """
    return embedding / np.linalg.norm(embedding)

def store_image_embedding(image_id, embedding, description):

    image_embeddings.append(embedding)
    image_ids.append(image_id)
    embedding=normalize_embedding(embedding)
    image_descriptions[image_id] = description  # Store description
    image_index.add(embedding)  # Add to FAISS

def store_text_embedding(text_id, embedding):

    text_embeddings.append(embedding)
    text_ids.append(text_id)
    embedding=normalize_embedding(embedding)
    text_index.add(embedding)  # Add to FAISS

print("faiss")


