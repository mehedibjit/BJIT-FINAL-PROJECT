#multimodal Search
import sys
import os

# Adjust this path as needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))) 
import numpy as np
import faiss
import torch
from torch.nn.functional import cosine_similarity

from ai.embeddings.image_embedding import get_image_embedding
from ai.embeddings.text_embedding import get_text_embedding
from ai.faiss_database.faiss_index import normalize_embedding
from ai.faiss_database.faiss_index import image_index, text_index, image_ids, text_ids, image_descriptions,normalize_embedding

def multimodal_search(image_query, text_query,image_index,text_index, alpha=0.5):
    """
    Perform late fusion search combining image and text similarity scores.
    :param image_query: Image input for similarity search.
    :param text_query: Text input for similarity search.
    :param alpha: Weight for combining similarities.
    :return: List of tuples (image_id, similarity_score, description)
    """

    # Ensure FAISS indexes are not empty
    if image_index.ntotal == 0 or text_index.ntotal == 0:
        print("Warning: FAISS index is empty! Make sure embeddings are added before searching.")
        return []

    # Get embeddings for query and normalize them for cosine similarity
    image_embedding = normalize_embedding(get_image_embedding(image_query))
    text_embedding = normalize_embedding(get_text_embedding(text_query))

    # Perform FAISS search
    image_similarities, image_indices = image_index.search(image_embedding, k=3)  # Top-3 results
    text_similarities, text_indices = text_index.search(text_embedding, k=3)

    # Retrieve actual image IDs from FAISS indices
    retrieved_image_ids = [image_ids[idx] for idx in image_indices[0] if idx < len(image_ids)]
    retrieved_text_ids = [text_ids[idx] for idx in text_indices[0] if idx < len(text_ids)]

    # Merge image and text search results (Use UNION | instead of INTERSECTION &)
    all_ids = list(set(retrieved_image_ids) | set(retrieved_text_ids))

    # Store similarity scores (since FAISS already returns cosine similarity, no need for extra transformation)
    image_scores = {image_ids[idx]: max(0, image_similarities[0][i]) for i, idx in enumerate(image_indices[0]) if idx < len(image_ids)}
    text_scores = {text_ids[idx]: max(0, text_similarities[0][i]) for i, idx in enumerate(text_indices[0]) if idx < len(text_ids)}

    # Perform weighted fusion for all retrieved IDs
    fused_scores = {}
    for image_id in all_ids:
        image_sim = image_scores.get(image_id, 0)  # Default to 0 if missing
        text_sim = text_scores.get(image_id, 0)  # Default to 0 if missing
        fused_scores[image_id] = alpha * image_sim + (1 - alpha) * text_sim

    # Rank final results by highest similarity score
    sorted_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    #return sorted_results  # Returns (image_id, similarity_score)
    # Retrieve descriptions for the images
    final_results = [(image_id, score, image_descriptions.get(image_id, "No description available")) for image_id, score in sorted_results]
    print("multimodal image")
    return final_results  # Returns (image_id, similarity_score, description)




#search image with text query



def cosine_to_percentage(cosine_sim):
    # Convert cosine similarity to percentage
    return (cosine_sim + 1) / 2 * 100

def search_using_text(query, faiss_index, image_ids, threshold=0.85, top_k=10):
    # Step 1: Generate embedding for the query
    query_embedding = get_text_embedding(query)  # Get the embedding as a PyTorch tensor
    query_embedding = normalize_embedding(query_embedding)  # Normalize the embedding

    # Ensure query_embedding is a PyTorch tensor
    if not isinstance(query_embedding, torch.Tensor):
        query_embedding = torch.tensor(query_embedding)

    # Step 2: Retrieve top N similar images from the index
    D, I = faiss_index.search(query_embedding.detach().numpy(), k=top_k)  # Get top_k results

    # Step 3: Calculate cosine similarity for each image and filter based on threshold
    similar_images = []
    for idx in range(len(I[0])):
        # Get the actual image path using the returned index from FAISS
        image_path = image_ids[I[0][idx]]
        image_embedding = get_image_embedding(image_path)  # You should store image embeddings somewhere

        # Ensure image_embedding is a PyTorch tensor
        if not isinstance(image_embedding, torch.Tensor):
            image_embedding = torch.tensor(image_embedding)

        # Calculate cosine similarity between the query and the image
        similarity_score = cosine_similarity(query_embedding, image_embedding.unsqueeze(0)).mean().item()

        # Convert similarity score to percentage
        similarity_percentage = cosine_to_percentage(similarity_score)

        # If similarity is above the threshold, add the image to the result list
        if similarity_percentage >= threshold * 100:  # Threshold converted to percentage
            similar_images.append((image_path, similarity_percentage))

    # Sort results by similarity (descending order)
    similar_images.sort(key=lambda x: x[1], reverse=True)
    return similar_images



def search_using_image(image_path, faiss_index, image_ids, threshold=0.4, top_k=4):
    # Step 1: Generate embedding for the query
    image_embedding = normalize_embedding(get_image_embedding(image_path))
    
    # Step 2: Retrieve top N similar images from the index
    D, I = faiss_index.search(image_embedding, k=top_k)  # Get top_k results

    # Step 3: Calculate cosine similarity for each image and filter based on threshold
    similar_images = []
    for idx in range(len(I[0])):
        # Get the actual image path using the returned index from FAISS
        retrieved_image_path = image_ids[I[0][idx]]
        retrieved_image_embedding = get_image_embedding(retrieved_image_path)  # You should store image embeddings somewhere

        # Calculate cosine similarity between the query and the image
        similarity_score = cosine_similarity(torch.tensor(image_embedding), torch.tensor(retrieved_image_embedding).unsqueeze(0)).mean().item()

        # Convert similarity score to percentage
        similarity_percentage = cosine_to_percentage(similarity_score)
        print(similarity_percentage)

        # If similarity is above the threshold, add the image to the result list
        if similarity_percentage >= threshold * 100:  # Threshold converted to percentage
            similar_images.append((retrieved_image_path, similarity_percentage)) # Append image path instead of index

    # Sort results by similarity (descending order)
    similar_images.sort(key=lambda x: x[1], reverse=True)
    return similar_images

