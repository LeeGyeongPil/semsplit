"""Opt-in options, one at a time. Run: uv run python examples/options.py"""

from pathlib import Path

from semsplit import HashEmbeddings, SemanticChunker, analyze

data = Path(__file__).parent / "data"
en = (data / "three_topics.en.txt").read_text(encoding="utf-8")
ja = (data / "ja.txt").read_text(encoding="utf-8")

# 1. Hard size limit: every chunk is at most 60 characters (no extra embedding calls).
sized = SemanticChunker(HashEmbeddings(), breakpoint_threshold_amount=50, max_chunk_size=60)
print("max_chunk_size=60 ->", [len(c) for c in sized.split_text(en)])

# 2. Japanese has no spaces between sentences: sentence_split="cjk" splits at 。！？
default = SemanticChunker(HashEmbeddings(), breakpoint_threshold_amount=50)
cjk = SemanticChunker(HashEmbeddings(), breakpoint_threshold_amount=50, sentence_split="cjk")
print("japanese, default ->", len(default.split_text(ja)), "chunk(s)")
print("japanese, cjk     ->", len(cjk.split_text(ja)), "chunk(s)")

# 3. Look inside: distances, threshold and the fragile cuts.
info = analyze(en, HashEmbeddings(), breakpoint_threshold_amount=50)
print(f"threshold={info.threshold:.4f} cuts={info.chunk_count - 1} fragile={len(info.fragile)}")
