import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Step 1: Load your catalog
print("Loading catalog...")
df = pd.read_csv("catalog.csv")

# Step 2: Combine the important fields into one sentence per book
df["combined"] = df["title"] + " by " + df["author"] + ". " + df["description"]

# Step 3: Load the AI model
print("Loading AI model (this may take a minute first time)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Turn each book description into a vector (list of numbers)
print("Creating embeddings...")
embeddings = model.encode(df["combined"].tolist())

# Step 5: Build the search index
print("Building search index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Step 6: Save everything for later
faiss.write_index(index, "library.index")
df.to_csv("catalog_processed.csv", index=False)
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Done! Your library search index is ready.")