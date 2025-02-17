import google.generativeai as genai
import os
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
# Get the current directory of this file (image_embedding.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Move up TWO levels to reach the project root
project_root = os.path.abspath(os.path.join(current_dir, ".."))


# Construct the path to "data/upload images/"
data_path = os.path.join(project_root, "config", "api_key.txt")


# Function to load API key from config file
def load_api_key():
    config_path = data_path  # Adjust if needed
    try:
        with open(config_path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key file not found at {config_path}. Please create it.")

# Load API Key
API_KEY = load_api_key()

# Configure Gemini API
genai.configure(api_key=API_KEY)

# Initialize the model
modelg = genai.GenerativeModel('gemini-2.0-flash')

def generate_description(image_path):
    try:
        # Open the image
        img = Image.open(image_path)

        # Generate description using Gemini
        response = modelg.generate_content(["Describe this image in 50 words.", img])

        # Return the generated description
        return response.text if response else "No response from API."
    
    except FileNotFoundError:
        return "Error: Image file not found."
    except Exception as e:
        return f"Error: {str(e)}"

# ✅ Retry mechanism with exponential backoff
@retry(wait=wait_exponential(multiplier=1, min=1, max=60), stop=stop_after_attempt(5))
def generate_description_with_retry(image_path):
    return generate_description(image_path)  # Calls original function with retry




