import sys
import os
from PIL import Image

# Set the environment variable to allow multiple OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Add the parent directory of 'ai' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Get the current directory of this file (image_embedding.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Move up TWO levels to reach the project root
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))


# Construct the path to "data/upload images/"
data_path = os.path.join(project_root, "Data", "upload_images")




from ai.embeddings.image_embedding import get_image_embedding
from ai.embeddings.text_embedding import get_text_embedding
from ai.faiss_database.faiss_index import normalize_embedding
from ai.faiss_database.faiss_index import store_image_embedding
from ai.faiss_database.faiss_index import store_text_embedding
from ai.faiss_database.faiss_index import store_tag_embedding
from ai.faiss_database.faiss_index import Image_databse_path
from ai.faiss_database.faiss_index import Text_databse_path
from ai.faiss_database.faiss_index import Tag_databse_path
from ai.faiss_database.faiss_index import image_ids     # List of image IDs
from ai.faiss_database.search_in_faiss import search_using_image
from ai.faiss_database.search_in_faiss import search_using_text
from ai.faiss_database.search_in_faiss import multimodal_search
from ai.utills.generate_description import generate_description_with_retry
from ai.utills.generate_image_tag import generate_image_tag_with_retry
import faiss


des = "its a cat image"

image_files = [f for f in os.listdir(data_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]
for image_name in image_files:
    image_path = os.path.join(data_path, image_name)
    try:
        # Generate embedding for the image
        embedding = get_image_embedding(image_path)
        description = generate_description_with_retry(image_path)
        tag=generate_image_tag_with_retry(image_path)
        print(image_path)
        print(description)
        print("tags are : ",tag)


        # Store the embedding with the image ID (filename with extension)
        store_image_embedding(image_path, embedding, description)
        embedding=get_text_embedding(description)
        store_text_embedding(image_path, embedding)
        tag_embedding=get_text_embedding(tag)
        store_tag_embedding(image_path, tag_embedding)
        
        #print(f"Stored embedding for {image_path}")
        print("\n")
        print("\n")
        print("\n")
    except FileNotFoundError:
        print(f"File not found: {image_path}")



def load_faiss_index(index_path):
    """Load FAISS index"""
    return faiss.read_index(index_path)

image_index = load_faiss_index(Image_databse_path)
text_index = load_faiss_index(Text_databse_path)

print(image_index.ntotal)  # Number of image embeddings stored






