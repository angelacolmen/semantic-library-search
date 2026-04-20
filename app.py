import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import anthropic
import os

# Load everything
print("Loading search engine...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Build index on startup
import subprocess
subprocess.run(["python", "build_index.py"])

index = faiss.read_index("library.index")
df = pd.read_csv("catalog_processed.csv")
print("Ready!")


def get_claude_explanation(query, books):
    """Ask Claude to explain why these books matched the query."""
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        client = anthropic.Anthropic(api_key=api_key)

        book_list = "\n".join([
            f"- {b['title']} by {b['author']} ({b['subject']}): {b['description']}"
            for b in books
        ])

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": f"""A library patron searched for: "{query}"

These books were returned as semantic matches:
{book_list}

In 2-3 sentences, explain why these books are relevant to the patron's search.
Focus on the themes and ideas that connect the search to these results.
Write in a warm, helpful librarian voice. Do not use bullet points."""
                }
            ]
        )
        return message.content[0].text
    except Exception as e:
        print(f"Claude error: {e}")
        return None


def search(query, num_results, api_key):
    if not query:
        return "Please enter a search query."

    if api_key and api_key.strip().startswith("sk-"):
        os.environ["ANTHROPIC_API_KEY"] = api_key.strip()

    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), int(num_results))

    books = []
    for idx in indices[0]:
        book = df.iloc[idx]
        books.append({
            "title": book["title"],
            "author": book["author"],
            "subject": book["subject"],
            "description": book["description"]
        })

    explanation = get_claude_explanation(query, books)

    results = ""

    if explanation:
        results += f"**Why these results?**\n\n{explanation}\n\n---\n\n"

    for i, book in enumerate(books):
        results += f"**{i+1}. {book['title']}**\n\n"
        results += f"Author: {book['author']}  \n"
        results += f"Subject: {book['subject']}  \n"
        results += f"{book['description']}\n\n---\n\n"

    return results


css = """
body, .gradio-container {
    background-color: #f0f4f8 !important;
    color: #1a1a1a !important;
}
.gradio-container * {
    color: #1a1a1a !important;
}
h1, h2, h3, p, label, span, div {
    color: #1a1a1a !important;
}
.prose, .prose * {
    color: #1a1a1a !important;
}
textarea, input {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    border: 1.5px solid #cbd5e1 !important;
}
button.primary {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
}
button.primary:hover {
    background-color: #1d4ed8 !important;
}
.markdown-body, .md {
    color: #1a1a1a !important;
    background: #ffffff !important;
    padding: 1rem !important;
    border-radius: 8px !important;
}
footer { display: none !important; }
"""

with gr.Blocks(theme=gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter")
)) as demo:

    gr.Markdown("# 📚 Semantic Library Search")
    gr.Markdown("Search by meaning, not just keywords — powered by AI")

    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                label="What are you looking for?",
                placeholder="e.g. books about artificial intelligence and society...",
                lines=2
            )
        with gr.Column(scale=1):
            num_results = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Results"
            )

    with gr.Row():
        api_key_input = gr.Textbox(
            label="🔑 Anthropic API Key (optional — enables AI explanations)",
            placeholder="sk-ant-...",
            type="password",
            scale=4
        )

    search_btn = gr.Button("Search →", variant="primary")

    results_output = gr.Markdown()

    gr.Examples(
        examples=[
            ["books about artificial intelligence and society", 3, ""],
            ["stories about race and justice in America", 3, ""],
            ["women who made a difference in science", 3, ""],
            ["how governments control people", 3, ""],
            ["climate change and the future of the planet", 3, ""],
            ["libraries and the organization of knowledge", 3, ""],
            ["survival against the odds", 3, ""],
        ],
        inputs=[query_input, num_results, api_key_input],
        label="Try these searches"
    )

    search_btn.click(
        fn=search,
        inputs=[query_input, num_results, api_key_input],
        outputs=results_output
    )

    query_input.submit(
        fn=search,
        inputs=[query_input, num_results, api_key_input],
        outputs=results_output
    )

demo.launch()
