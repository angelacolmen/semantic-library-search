import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import anthropic
import os

print("Loading search engine...")
model = SentenceTransformer("all-MiniLM-L6-v2")

import subprocess
subprocess.run(["python", "build_index.py"])

index = faiss.read_index("library.index")
df = pd.read_csv("catalog_processed.csv")
print("Ready!")


def get_claude_explanation(query, books):
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
        results += f"#### Librarian's Note\n\n{explanation}\n\n---\n\n"

    for i, book in enumerate(books):
        results += f"**{i+1}. {book['title']}**\n\n"
        results += f"*{book['author']}* &nbsp;·&nbsp; {book['subject']}\n\n"
        results += f"{book['description']}\n\n---\n\n"

    return results


css = """
:root { color-scheme: light; }
html { color-scheme: light; }

@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500&family=Source+Sans+3:ital,wght@0,300;0,400;0,600;1,400&display=swap');

:root {
    --maroon:      #5C1A1A;
    --maroon-mid:  #7A2424;
    --maroon-rule: #8B3A3A;
    --maroon-tint: #F9F2F2;
    --ink:         #1A1A1A;
    --ink-2:       #3D3D3D;
    --ink-3:       #767676;
    --surface:     #FFFFFF;
    --bg:          #F5F3EF;
    --rule:        #DDD9D0;
    --rule-light:  #EAE7E0;
}

html, body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Source Sans 3', Georgia, sans-serif !important;
    color: var(--ink) !important;
}

footer, .built-with, .svelte-1rjryqp { display: none !important; }

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

.inst-bar {
    background: var(--maroon) !important;
    padding: 0.55rem 2.5rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    border-bottom: 3px solid var(--maroon-rule) !important;
}

.inst-bar-left {
    display: flex !important;
    align-items: center !important;
    gap: 0.75rem !important;
}

.inst-bar-icon {
    width: 26px !important;
    height: 26px !important;
    opacity: 0.9 !important;
}

.inst-bar-name {
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #F0E8E8 !important;
}

.inst-bar-badge {
    font-size: 0.65rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #C89A9A !important;
    border: 1px solid #8B5555 !important;
    padding: 0.2rem 0.55rem !important;
    border-radius: 2px !important;
}

.content-wrap {
    padding: 2.5rem 2.5rem 4rem !important;
}

.page-title-block {
    margin-bottom: 2rem !important;
    padding-bottom: 1.5rem !important;
    border-bottom: 1px solid var(--rule) !important;
}

.page-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 2.1rem !important;
    font-weight: 400 !important;
    color: var(--maroon) !important;
    letter-spacing: -0.01em !important;
    line-height: 1.15 !important;
    margin-bottom: 0.4rem !important;
}

.page-desc {
    font-size: 0.9rem !important;
    color: var(--ink-3) !important;
    line-height: 1.6 !important;
    max-width: 560px !important;
    font-weight: 300 !important;
}

label span, .label-wrap span {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--ink-3) !important;
}

textarea, input[type="text"], input[type="password"] {
    background: var(--surface) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 3px !important;
    color: var(--ink) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 300 !important;
    padding: 0.7rem 0.9rem !important;
    box-shadow: none !important;
    transition: border-color 0.15s !important;
}

textarea:focus, input:focus {
    border-color: var(--maroon-mid) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(92,26,26,0.07) !important;
}

input[type="range"] { accent-color: var(--maroon) !important; }

button.primary, button[variant="primary"] {
    background: var(--maroon) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.72rem 1.6rem !important;
    transition: background 0.15s, transform 0.1s !important;
    box-shadow: none !important;
}

button.primary:hover, button[variant="primary"]:hover {
    background: var(--maroon-mid) !important;
    transform: translateY(-1px) !important;
}

button.primary:active { transform: translateY(0) !important; }

.examples button, button.secondary {
    background: var(--surface) !important;
    color: var(--ink-2) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 3px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 300 !important;
    padding: 0.35rem 0.8rem !important;
    transition: border-color 0.15s, color 0.15s !important;
}

.examples button:hover {
    border-color: var(--maroon) !important;
    color: var(--maroon) !important;
}

.prose, .md, .markdown-body, [data-testid="markdown"] {
    background: transparent !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.8 !important;
    font-weight: 300 !important;
}

.prose p, .md p, [data-testid="markdown"] p,
.prose li, .md li, [data-testid="markdown"] li {
    color: #1A1A1A !important;
}

.inst-bar-name { color: #F0E8E8 !important; }
.inst-bar-badge { color: #C89A9A !important; }

[data-testid="markdown"] strong, .prose strong, .md strong {
    font-family: 'Playfair Display', serif !important;
    font-weight: 400 !important;
    font-size: 1.05rem !important;
    color: #1A1A1A !important;
}

[data-testid="markdown"] em, .prose em, .md em {
    font-style: normal !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.03em !important;
    color: #767676 !important;
    font-weight: 400 !important;
}

[data-testid="markdown"] h4, .prose h4, .md h4 {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #5C1A1A !important;
    margin-bottom: 0.5rem !important;
    font-family: 'Source Sans 3', sans-serif !important;
}

[data-testid="markdown"] hr, .prose hr, .md hr {
    border: none !important;
    border-top: 1px solid #EAE7E0 !important;
    margin: 1.2rem 0 !important;
}

.poc-banner {
    background: var(--maroon-tint) !important;
    border: 1px solid #D4AAAA !important;
    border-left: 3px solid var(--maroon) !important;
    border-radius: 0 3px 3px 0 !important;
    padding: 0.6rem 1rem !important;
    margin-bottom: 2rem !important;
    font-size: 0.8rem !important;
    color: var(--maroon) !important;
    font-weight: 400 !important;
    line-height: 1.5 !important;
}
"""

with gr.Blocks(
    css=css,
    theme=gr.themes.Base(
        font=gr.themes.GoogleFont("Source Sans 3"),
    ),
    title="Research Collections Search"
) as demo:

    gr.HTML("""
    <div class="inst-bar">
        <div class="inst-bar-left">
            <svg class="inst-bar-icon" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="3" y="16" width="3" height="7" fill="#F0E8E8" opacity="0.8"/>
                <rect x="7.5" y="12" width="3" height="11" fill="#F0E8E8" opacity="0.8"/>
                <rect x="12" y="8" width="3" height="15" fill="#F0E8E8"/>
                <rect x="16.5" y="12" width="3" height="11" fill="#F0E8E8" opacity="0.8"/>
                <rect x="21" y="16" width="3" height="7" fill="#F0E8E8" opacity="0.6"/>
                <rect x="2" y="23" width="22" height="1.5" fill="#F0E8E8" opacity="0.5"/>
            </svg>
            <span class="inst-bar-name">Research Collections &amp; Services</span>
        </div>
        <span class="inst-bar-badge">Proof of Concept</span>
    </div>
    """)

    with gr.Column(elem_classes=["content-wrap"]):

        gr.HTML("""
        <div class="page-title-block">
            <div class="page-title">Semantic Search</div>
            <div class="page-desc">
                Discover collections through meaning rather than exact keywords. Describe what you're looking for
                in plain language — subjects, themes, questions, or ideas.
            </div>
        </div>
        """)

        gr.HTML("""
        <div class="poc-banner">
            This tool is an early-stage prototype demonstrating AI-powered semantic search across a 200-title catalog.
            Results are ranked by conceptual relevance using sentence-transformer embeddings.
        </div>
        """)

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                query_input = gr.Textbox(
                    label="Search query",
                    placeholder="e.g. the social consequences of artificial intelligence...",
                    lines=2,
                )
            with gr.Column(scale=1, min_width=110):
                num_results = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="Results",
                )

        with gr.Row(equal_height=True):
            api_key_input = gr.Textbox(
                label="Anthropic API Key — optional, enables AI-generated annotations",
                placeholder="sk-ant-...",
                type="password",
                scale=5,
            )
            with gr.Column(scale=1, min_width=130):
                gr.HTML("<div style='height:1.55rem'></div>")
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
            label="Example queries",
        )

    search_btn.click(
        fn=search,
        inputs=[query_input, num_results, api_key_input],
        outputs=results_output,
    )
    query_input.submit(
        fn=search,
        inputs=[query_input, num_results, api_key_input],
        outputs=results_output,
    )

demo.launch()
