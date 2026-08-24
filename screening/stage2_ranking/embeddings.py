"""Semantic similarity via BAAI/bge-m3, self-hosted through sentence-transformers.

Free, no API account, no token limits, strong French/English coverage. The model is
~2 GB and is downloaded from HuggingFace on first use, then cached locally
(~/.cache/huggingface). The first call in a fresh environment is slow; every call
after that is local.
"""

from sentence_transformers import SentenceTransformer

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-m3")  # load once, reuse across the run
    return _model


def semantic_scores(jd_text: str, candidate_texts: list[str]) -> list[float]:
    # Guard added beyond CLAUDE.md: encoding an empty batch is a needless model load,
    # and some backends error on a zero-row input.
    if not candidate_texts:
        return []
    model = get_embedding_model()
    jd_vec = model.encode(jd_text, normalize_embeddings=True)
    candidate_vecs = model.encode(candidate_texts, normalize_embeddings=True)
    return (candidate_vecs @ jd_vec).tolist()  # normalized vectors → dot product == cosine similarity
