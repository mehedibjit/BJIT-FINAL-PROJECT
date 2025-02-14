import sys
import os

# Adjust this path as needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))) 

import torch

import numpy as np
from PIL import Image
from ai.models.CLIP_model import model, processor

def get_text_embedding(text):
    # Truncate the text to the maximum sequence length of the model
    max_length = model.config.text_config.max_position_embeddings  # Get max sequence length
    inputs = processor(text=text[:max_length], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    return text_embedding.numpy().astype("float32")  # FAISS requires float32
print("Image embedding")