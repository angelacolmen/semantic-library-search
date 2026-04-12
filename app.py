from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Start the app
app = FastAPI()

# Allow the webpage to talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load everything
print("Loading search engine...")
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("library.index")
df = pd.read_csv("catalog_processed.csv")
print("Ready!")

# Serve the webpage
@app.get("/")
def serve_homepage():
    return FileResponse("D:/semantic-library-search/index.html")

# Define what a search request looks like
class SearchRequest(BaseModel):
    query: str
    num_results: int = 3

# Create the search endpoint
@app.post("/search")
def search(request: SearchRequest):
    query_embedding = model.encode([request.query])
    distances, indices = index.search(np.array(query_embedding), request.num_results)
    
    results = []
    for idx in indices[0]:
        book = df.iloc[idx]
        results.append({
            "title": book["title"],
            "author": book["author"],
            "subject": book["subject"],
            "description": book["description"]
        })
    
    return {"query": request.query, "results": results}