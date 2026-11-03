"""The only place that turns text into vectors. Swap this file to change backend.

An Ollama embedding model (for example nomic-embed-text through
POST /api/embed) could replace it; it must return one normalised vector
per input, in order.
"""

MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # pinned commit on the Hub
# From the model card's model list: add this to queries, never to passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def embed(texts, is_query=False, prefix=QUERY_PREFIX, batch_size=32):
    """Return a numpy array (len(texts), 384) of unit-length float32 vectors."""
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(2)  # shared machine: keep CPU use modest
        _model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    if is_query and prefix:
        texts = [prefix + t for t in texts]
    return _model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
