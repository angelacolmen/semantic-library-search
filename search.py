import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load everything we built
print("Loading search engine...")
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("library.index")
df = pd.read_csv("catalog_processed.csv")

print("Ready! Type your search question below.")
print("Type 'quit' to exit\n")

# Search loop
while True:
    query = input("Search: ")
    if query.lower() == "quit":
        break

    # Turn your search query into a meaning fingerprint
    query_embedding = model.encode([query])

    # Find the 3 most similar books
    distances, indices = index.search(np.array(query_embedding), 3)

    print("\nTop results:")
    for i, idx in enumerate(indices[0]):
        book = df.iloc[idx]
        print(f"{i+1}. {book['title']} by {book['author']}")
        print(f"   {book['description']}\n")