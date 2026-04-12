import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# Load everything
print("Loading search engine...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Build index on startup
import subprocess
subprocess.run(["python", "build_index.py"])

index = faiss.read_index("library.index")
df = pd.read_csv("catalog_processed.csv")
print("Ready!")

def search(query, num_results=3):
    if not query:
        return "Please enter a search query."
    
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), num_results)
    
    results = ""
    for i, idx in enumerate(indices[0]):
        book = df.iloc[idx]
        results += f"### {i+1}. {book['title']}\n"
        results += f"**Author:** {book['author']}\n"
        results += f"**Subject:** {book['subject']}\n"
        results += f"{book['description']}\n\n"
    
    return results

# Create Gradio interface
demo = gr.Interface(
    fn=search,
    inputs=[
        gr.Textbox(
            label="Search Query",
            placeholder="e.g. books about space and the universe...",
            lines=2
        ),
        gr.Slider(
            minimum=1,
            maximum=5,
            value=3,
            step=1,
            label="Number of Results"
        )
    ],
    outputs=gr.Markdown(label="Search Results"),
    title="📚 Semantic Library Search",
    description="Search by meaning, not just keywords. Try searching for 'books about space and the universe' or 'stories about race and justice in America'.",
    examples=[
        ["books about space and the universe", 3],
        ["stories about race and justice in America", 3],
        ["women who made a difference in science", 3],
        ["how governments control people", 3],
        ["survival against the odds", 3]
    ]
)

if __name__ == "__main__":
    demo.launch()