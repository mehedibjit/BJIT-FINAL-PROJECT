import torch
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model
model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

# Load the corresponding processor
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
