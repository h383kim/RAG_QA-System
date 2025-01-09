import os
from dotenv import load_dotenv

# If you use a .env file, load it
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "replace-with-your-key")
PERSIST_DIRECTORY = "db"
ARTICLES_DIRECTORY = "data/articles"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
