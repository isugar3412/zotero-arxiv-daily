import time

from loguru import logger

from .base import BaseReranker, register_reranker
from openai import OpenAI
import numpy as np
@register_reranker("api")
class ApiReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        client = OpenAI(api_key=self.config.reranker.api.key, base_url=self.config.reranker.api.base_url)
        batch_size = self.config.reranker.api.get("batch_size") or 64
        max_retries = self.config.reranker.api.get("max_retries") or 3
        retry_delay_seconds = self.config.reranker.api.get("retry_delay_seconds") or 1
        all_texts = s1 + s2
        all_embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            for attempt in range(1, max_retries + 1):
                try:
                    response = client.embeddings.create(
                        input=batch,
                        model=self.config.reranker.api.model
                    )
                    break
                except Exception as e:
                    status_code = getattr(e, "status_code", None)
                    should_retry = status_code is None or status_code >= 500
                    if attempt >= max_retries or not should_retry:
                        raise
                    logger.warning(
                        "Embedding request failed (attempt {}/{}): {}. Retrying...",
                        attempt,
                        max_retries,
                        e,
                    )
                    time.sleep(retry_delay_seconds * attempt)
            all_embeddings.extend([r.embedding for r in response.data])
        s1_embeddings = np.array(all_embeddings[:len(s1)])           # [n_s1, d]
        s2_embeddings = np.array(all_embeddings[len(s1):])           # [n_s2, d]
        s1_embeddings_normalized = s1_embeddings / np.linalg.norm(s1_embeddings, axis=1, keepdims=True)
        s2_embeddings_normalized = s2_embeddings / np.linalg.norm(s2_embeddings, axis=1, keepdims=True)
        sim = np.dot(s1_embeddings_normalized, s2_embeddings_normalized.T) # [n_s1, n_s2]
        return sim
