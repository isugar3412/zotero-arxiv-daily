"""Tests for ApiReranker — uses stub OpenAI client via monkeypatch."""

from types import SimpleNamespace

from zotero_arxiv_daily.reranker.api import ApiReranker


def test_api_reranker_similarity_shape(config, patch_openai):
    reranker = ApiReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)


def test_api_reranker_batching(config, patch_openai):
    reranker = ApiReranker(config)
    s1 = [f"text {i}" for i in range(5)]
    s2 = [f"corpus {i}" for i in range(3)]
    score = reranker.get_similarity_score(s1, s2)
    assert score.shape == (5, 3)


def test_api_reranker_retries_transient_error(config, monkeypatch):
    reranker = ApiReranker(config)
    config.reranker.api.max_retries = 3
    config.reranker.api.retry_delay_seconds = 0
    attempts = {"count": 0}

    class TransientError(Exception):
        status_code = 503

    def flaky_embeddings_create(**kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TransientError("temporary failure")
        inputs = kwargs.get("input", [])
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3], index=i, object="embedding") for i, _ in enumerate(inputs)],
            model="text-embedding-3-large",
            object="list",
        )

    stub_client = SimpleNamespace(embeddings=SimpleNamespace(create=flaky_embeddings_create))
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kwargs: stub_client)
    score = reranker.get_similarity_score(["hello"], ["world"])

    assert attempts["count"] == 3
    assert score.shape == (1, 1)
