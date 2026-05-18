"""CLI for building and querying the local legacy-data vector store."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rag.document_loader import DEFAULT_DATASETS, build_documents
from src.rag.vectorstore import LocalTfidfVectorStore, format_rag_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/query Biz-Insight legacy CSV vector store.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build vector store from CSV data.")
    build_parser.add_argument("--data-dir", default="data", type=Path)
    build_parser.add_argument("--out-dir", default="data/vectorstore", type=Path)
    build_parser.add_argument("--max-rows-per-doc", default=80, type=int)
    build_parser.add_argument("--max-features", default=200_000, type=int)
    build_parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="CSV filenames to include. Defaults to curated legacy datasets.",
    )

    query_parser = subparsers.add_parser("query", help="Query an existing vector store.")
    query_parser.add_argument("query")
    query_parser.add_argument("--index-dir", default="data/vectorstore", type=Path)
    query_parser.add_argument("--top-k", default=5, type=int)
    query_parser.add_argument("--source-file")
    query_parser.add_argument("--entity")
    query_parser.add_argument("--entity-match", choices=["exact", "contains"], default="exact")
    query_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.command == "build":
        documents = build_documents(
            data_dir=args.data_dir,
            dataset_files=args.datasets,
            max_rows_per_doc=args.max_rows_per_doc,
        )
        store = LocalTfidfVectorStore.from_documents(
            documents,
            max_features=args.max_features,
        )
        store.save(args.out_dir)
        print(
            json.dumps(
                {
                    "out_dir": str(args.out_dir),
                    "documents": len(documents),
                    "features": int(store.matrix.shape[1]),
                    "datasets": len(set(doc.source_file for doc in documents)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "query":
        store = LocalTfidfVectorStore.load(args.index_dir)
        results = store.search(
            args.query,
            top_k=args.top_k,
            source_file=args.source_file,
            entity=args.entity,
            entity_match=args.entity_match,
        )
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "score": result.score,
                            "doc_id": result.doc_id,
                            "metadata": result.metadata,
                            "text": result.text,
                        }
                        for result in results
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(format_rag_context(results))


if __name__ == "__main__":
    main()
