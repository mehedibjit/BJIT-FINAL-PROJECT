
import time
import google.generativeai as genai
import os,sys
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential



# Adjust this path as needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))) 
from ai.utills.generate_description import data_path



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
modeltag = genai.GenerativeModel('gemini-2.0-flash')


def generate_image_tag(image_path):
    try:
        # Open the image
        img = Image.open(image_path)

         # Generate Image Caption & Objects using Gemini
        response = modeltag.generate_content(
            [
                "Analyze this image and provide a detailed list of objects, scene , dominant colors, year (if possible), and location (if possible).",
                img  # Pass the PIL Image object directly
            ]
            )

        # Check for response and extract tags
        if response and hasattr(response, "text"):
            return response.text.split("\n")  # Convert response into a tag list
        else:
            return ["Unknown Tags"]

    
    except FileNotFoundError:
        return "Error: Image file not found."
    except Exception as e:
        return f"Error: {str(e)}"

# ✅ Retry mechanism with exponential backoff






def generate_image_tag_with_retry(image_path, retries=5, backoff_factor=1):
    for attempt in range(retries):
        try:
            return generate_image_tag(image_path)
        except Exception as e:
            if "429" in str(e):
                wait_time = backoff_factor * (2 ** attempt)
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Failed to generate image tag after multiple retries")