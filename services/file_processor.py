from fastapi import UploadFile
from PyPDF2 import PdfReader
import pandas as pd
import json
import io

async def process_file(file: UploadFile) -> str:
    content = await file.read()
    
    if file.content_type == "application/pdf":
        reader = PdfReader(io.BytesIO(content))  # Use BytesIO for in-memory bytes
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    elif file.content_type in ["text/csv", "application/vnd.ms-excel"]:
        return pd.read_csv(io.BytesIO(content)).to_string(index=False)
    
    elif file.content_type == "application/json":
        return json.dumps(json.loads(content))
    
    elif file.content_type == "text/plain":
        return content.decode("utf-8")
    
    raise ValueError(f"Unsupported file type: {file.content_type}")