import sys
import os

# Adjust this path as needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))) 

import torch

import numpy as np
from PIL import Image
from ai.models.CLIP_model import model, processor

def get_image_embedding(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)
        
    return image_embedding.numpy().astype("float32")  # FAISS requires float32
print("Image embedding")