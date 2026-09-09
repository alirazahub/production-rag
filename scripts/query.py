import argparse

from rag.retrieval.embeddings import EmbeddingService
from rag.retrieval.retriever import Retriever


def main() -> None:
    """
    Development CLI for testing semantic retrieval.
    """

    parser = argparse.ArgumentParser(
        description="Search ingested documents using semantic similarity."
    )

    parser.add_argument(
        "query",
        help="Question or search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve.",
    )

    args = parser.parse_args()

    embedding_service = EmbeddingService()

    retriever = Retriever(
        embedding_service=embedding_service,
    )

    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
    )

    print()
    print(f"Query: {args.query}")
    print(f"Retrieved chunks: {len(results)}")

    for rank, result in enumerate(results, start=1):
        print()
        print("=" * 70)
        print(f"Rank: {rank}")
        print(f"Score: {result.score:.4f}")
        print(f"File: {result.filename}")
        print(f"Page: {result.page_number}")
        print(f"Chunk: {result.chunk_index}")
        print("-" * 70)
        print(result.text)


if __name__ == "__main__":
    main()