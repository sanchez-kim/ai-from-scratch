import sys
import types

import pytest

import hybrid.embed as embed_module

# The bge-small-en-v1.5 model card's query instruction, typed out by hand.
CARD_PREFIX = "Represent this sentence for searching relevant passages: "


class FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer: no download,
    no model. It records how it was built and what encode() received."""

    built = []

    def __init__(self, name, **kwargs):
        self.built.append((name, kwargs))
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return f"vectors for {len(texts)} texts"


@pytest.fixture
def fake_model(monkeypatch):
    # Fake torch and sentence_transformers modules, and an empty model slot,
    # so embed() takes its real loading path against the stand-ins.
    FakeSentenceTransformer.built = []
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(set_num_threads=lambda n: None))
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    monkeypatch.setattr(embed_module, "_model", None)
    return FakeSentenceTransformer


def test_prefix_goes_on_queries_only(fake_model):
    embed_module.embed(["a passage about bm25"])
    embed_module.embed(["what is bm25"], is_query=True)
    embed_module.embed(["what is bm25"], is_query=True, prefix="")
    model = embed_module._model
    texts = [call[0] for call in model.calls]
    assert texts == [
        ["a passage about bm25"],             # passages: never prefixed
        [CARD_PREFIX + "what is bm25"],       # queries: the model card's prefix
        ["what is bm25"],                     # prefix="" turns it off
    ]


def test_encode_asks_for_unit_length_numpy_vectors(fake_model):
    embed_module.embed(["one", "two"], batch_size=8)
    (_, kwargs), = embed_module._model.calls
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True
    assert kwargs["batch_size"] == 8


def test_model_is_the_pinned_bge_small_on_cpu_loaded_once(fake_model):
    embed_module.embed(["one"])
    embed_module.embed(["two"], is_query=True)
    assert len(fake_model.built) == 1
    name, kwargs = fake_model.built[0]
    assert name == "BAAI/bge-small-en-v1.5"
    assert kwargs["revision"] == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    assert kwargs["device"] == "cpu"
