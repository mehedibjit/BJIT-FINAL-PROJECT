import sys
import os

# Get the current directory of this file (image_embedding.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to "data/upload images/"
Image_databse_path = os.path.join(current_dir,  "database","image_faiss_index.bin")
Text_databse_path = os.path.join(current_dir,  "database","text_faiss_index.bin")
Tag_databse_path = os.path.join(current_dir,  "database","tag_faiss_index.bin")
print(Image_databse_path)

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
tag_embeddings = []
tag_ids = []
def normalize_embedding(embedding):
    """ Normalize embedding to unit length for cosine similarity """
    return embedding / np.linalg.norm(embedding)

def store_image_embedding(image_id, embedding, description):

    image_embeddings.append(embedding)
    image_ids.append(image_id)
    embedding=normalize_embedding(embedding)
    image_descriptions[image_id] = description  # Store description
    image_index.add(embedding)  # Add to FAISS 
    os.makedirs(os.path.dirname(Image_databse_path), exist_ok=True)
    faiss.write_index(image_index, Image_databse_path)


def store_text_embedding(image_id, embedding):

    text_embeddings.append(embedding)
    text_ids.append(image_id)
    embedding=normalize_embedding(embedding)
    text_index.add(embedding)  # Add to FAISS
    os.makedirs(os.path.dirname(Text_databse_path), exist_ok=True)
    faiss.write_index(text_index, Text_databse_path)


def store_tag_embedding(image_id, embedding):

    tag_embeddings.append(embedding)
    tag_ids.append(image_id)
    embedding=normalize_embedding(embedding)
    tag_index.add(embedding)  # Add to FAISS
    os.makedirs(os.path.dirname(Tag_databse_path), exist_ok=True)
    faiss.write_index(tag_index, Tag_databse_path)

print("faiss")


