from openai import OpenAI
import os
from utils.text_utils import truncate_text
from dotenv import load_dotenv
import tiktoken

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ENCODING = tiktoken.encoding_for_model("gpt-3.5-turbo")
MAX_TOKENS = 3000

def chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    tokens = ENCODING.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(ENCODING.decode(chunk_tokens))
    return chunks

def generate_summary(text: str) -> str:
    chunks = chunk_text(text, MAX_TOKENS)
    summaries = []
    
    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "Summarize the content in a very short sentence starting with 'This document contains' using as few words as possible."
            }, {
                "role": "user",
                "content": chunk
            }]
        )
        summaries.append(response.choices[0].message.content.strip())
    if len(summaries) > 1:
        combined = " ".join(summaries)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "Combine these into one very short sentence starting with 'This document contains' using as few words as possible."
            }, {
                "role": "user",
                "content": combined
            }]
        )
        return response.choices[0].message.content.strip()
    return summaries[0]