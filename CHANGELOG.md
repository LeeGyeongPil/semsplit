# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-19

### Added
- `SemanticChunker`: drop-in replacement for `langchain_experimental.text_splitter.SemanticChunker` 0.4.2
  (same constructor arguments, methods and default output, including the legacy `start_index` values),
  depending only on `langchain-core` and `numpy`.
- Opt-in fixes: `max_chunk_size` / `length_function`, `text_mode="original"` (exact source slices and
  `start_index`, no empty chunks, `min_chunk_size` on the last chunk), `sentence_split="cjk"` or a callback,
  `embed_batch_size`.
- `HashEmbeddings` (offline practice mock), `ReplayEmbeddings`, `OllamaEmbeddings`.
- `analyze()` for distances, threshold, cuts and fragile cuts; `check_parity()` / `record_golden()`.
- `semsplit` CLI with `split`, `explain` and `parity`; English and Korean screens (`--lang`, `SEMSPLIT_LANG`).
- 29 golden cases recorded from langchain-experimental 0.4.2.
