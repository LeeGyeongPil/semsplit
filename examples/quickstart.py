"""Quickstart: the only line you change when moving off langchain-experimental.

    - from langchain_experimental.text_splitter import SemanticChunker
    + from semsplit import SemanticChunker

HashEmbeddings is an offline practice mock (word overlap, not meaning). Pass any LangChain
Embeddings object (OpenAI, Ollama, HuggingFace ...) for real documents.
"""

from pathlib import Path

from semsplit import HashEmbeddings, SemanticChunker

text = (Path(__file__).parent / "data" / "three_topics.en.txt").read_text(encoding="utf-8")

# Same arguments as the original; 50 makes the short sample document split into several topics.
chunker = SemanticChunker(HashEmbeddings(), breakpoint_threshold_amount=50)
chunks = chunker.split_text(text)
print(f"chunks: {len(chunks)}")
for number, chunk in enumerate(chunks, start=1):
    print(f"  {number}. {chunk}")

# Opt-in improvements: chunks are exact slices of the source and never longer than 120 characters.
better = SemanticChunker(
    HashEmbeddings(),
    breakpoint_threshold_amount=50,
    text_mode="original",
    max_chunk_size=120,
    add_start_index=True,
)
for doc in better.create_documents([text]):
    start = doc.metadata["start_index"]
    assert text[start : start + len(doc.page_content)] == doc.page_content
    print(f"  start_index={start:<3} len={len(doc.page_content):<3} {doc.page_content[:40]}...")
