from openai import OpenAI
import os
from utils.text_utils import truncate_text
from dotenv import load_dotenv
import tiktoken

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ENCODING = tiktoken.encoding_for_model("gpt-3.5-turbo")
MAX_TOKENS = 8000  # Max tokens for text-embedding-3-small

def chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    tokens = ENCODING.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(ENCODING.decode(chunk_tokens))
    return chunks

def generate_embedding(text: str) -> list[list[float]]:
    chunks = chunk_text(text, MAX_TOKENS)
    embeddings = []
    
    for chunk in chunks:
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        embeddings.append(response.data[0].embedding)
    
    return embeddings