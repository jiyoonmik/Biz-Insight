"""A lightweight local vector store for the legacy Biz-Insight datasets."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.document_loader import RagDocument


INDEX_FILENAME = "tfidf_vectorstore.pkl"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class SearchResult:
    score: float
    doc_id: str
    text: str
    metadata: dict


class LocalTfidfVectorStore:
    """Persisted TF-IDF vector store.

    Character n-grams work reasonably for mixed Korean/English financial labels
    without requiring an external embedding model download.
    """

    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        matrix,
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.texts = texts
        self.metadatas = metadatas

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[RagDocument],
        max_features: int = 200_000,
    ) -> "LocalTfidfVectorStore":
        docs = list(documents)
        texts = [doc.text for doc in docs]
        metadatas = [
            {k: v for k, v in doc.to_record().items() if k != "text"}
            for doc in docs
        ]

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(texts)
        return cls(vectorizer, matrix, texts, metadatas)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / INDEX_FILENAME).open("wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "matrix": self.matrix,
                    "texts": self.texts,
                    "metadatas": self.metadatas,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        manifest = {
            "backend": "local_tfidf",
            "index_file": INDEX_FILENAME,
            "document_count": len(self.texts),
            "feature_count": int(self.matrix.shape[1]),
            "source_files": sorted({m["source_file"] for m in self.metadatas}),
        }
        (out_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path) -> "LocalTfidfVectorStore":
        with (index_dir / INDEX_FILENAME).open("rb") as f:
            payload = pickle.load(f)
        return cls(
            vectorizer=payload["vectorizer"],
            matrix=payload["matrix"],
            texts=payload["texts"],
            metadatas=payload["metadatas"],
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_file: str | None = None,
        entity: str | None = None,
        entity_match: str = "exact",
    ) -> list[SearchResult]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()

        candidate_indices = np.argsort(scores)[::-1]
        results: list[SearchResult] = []
        for idx in candidate_indices:
            metadata = self.metadatas[int(idx)]
            if source_file and metadata.get("source_file") != source_file:
                continue
            if entity and not _entity_matches(str(metadata.get("entity", "")), entity, entity_match):
                continue
            score = float(scores[int(idx)])
            if score <= 0:
                break
            results.append(
                SearchResult(
                    score=score,
                    doc_id=metadata["doc_id"],
                    text=self.texts[int(idx)],
                    metadata=metadata,
                )
            )
            if len(results) >= top_k:
                break
        return results


def _entity_matches(candidate: str, query: str, mode: str) -> bool:
    if mode == "contains":
        return query in candidate
    return candidate == query


def format_rag_context(results: Iterable[SearchResult], max_chars: int = 12_000) -> str:
    """Format search results as compact context for an LLM prompt."""
    chunks: list[str] = []
    total = 0
    for result in results:
        header = (
            f"[{result.doc_id}] score={result.score:.3f} "
            f"source={result.metadata.get('source_file')} "
            f"entity={result.metadata.get('entity')}"
        )
        block = f"{header}\n{result.text}"
        if total + len(block) > max_chars:
            break
        chunks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(chunks)
