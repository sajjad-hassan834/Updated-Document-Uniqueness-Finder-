from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.file_processor import process_file
from services.embeddings import generate_embedding
from services.uniqueness_checker import check_uniqueness
from services.summary_generator import generate_summary
import uuid
from chromadb import PersistentClient
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    chroma_client = PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("documents")
    logger.info("ChromaDB initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {str(e)}")
    raise

@app.get("/")
async def root():
    return {"message": "Welcome to the File Analysis API. Use POST /analyze/ to upload a file."}

@app.post("/analyze/")
async def analyze_file(file: UploadFile = File(...)):
    try:
        text = await process_file(file)
        logger.info(f"File processed: {file.filename}, size: {len(text)} characters")
        
        embeddings = generate_embedding(text)
        logger.info(f"Generated {len(embeddings)} embeddings")
        score, message = await check_uniqueness(embeddings[0], chroma_collection)
        summary = generate_summary(text)
        for embedding in embeddings:
            chroma_collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                documents=[text[:3000]]
            )
        logger.info(f"Stored {len(embeddings)} chunks in ChromaDB")
        
        return {
            "score": score,
            "status": message,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error in /analyze/: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)