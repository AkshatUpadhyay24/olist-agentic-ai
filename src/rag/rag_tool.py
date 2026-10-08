import os
import pickle

import faiss

from rag.embeddings import create_query_embedding


RAG_DIRECTORY = os.path.dirname(__file__)

VECTOR_INDEX_PATH = os.path.join(
    RAG_DIRECTORY,
    "knowledge_base.index"
)

CHUNKS_PATH = os.path.join(
    RAG_DIRECTORY,
    "knowledge_chunks.pkl"
)


# Minimum similarity required
SIMILARITY_THRESHOLD = 0.45


def load_vector_store():

    if not os.path.exists(
        VECTOR_INDEX_PATH
    ):
        raise FileNotFoundError(
            "FAISS index not found. "
            "Please build the vector store first."
        )

    if not os.path.exists(
        CHUNKS_PATH
    ):
        raise FileNotFoundError(
            "Knowledge chunks file not found. "
            "Please build the vector store first."
        )

    index = faiss.read_index(
        VECTOR_INDEX_PATH
    )

    with open(
        CHUNKS_PATH,
        "rb"
    ) as file:

        chunks = pickle.load(file)

    return index, chunks


def search_knowledge(
    query,
    top_k=3
):
    """
    Semantic knowledge search with:

    - FAISS similarity search
    - similarity threshold
    - source tracking
    - section tracking
    - rank tracking
    - similarity score tracking
    """

    try:

        index, chunks = load_vector_store()

        query_embedding = create_query_embedding(
            query
        )

        scores, indices = index.search(
            query_embedding,
            top_k
        )

        retrieved_chunks = []

        print("\nRAG SEARCH RESULTS")
        print("-" * 70)

        for rank, (score, index_position) in enumerate(
            zip(
                scores[0],
                indices[0]
            ),
            start=1
        ):

            if index_position == -1:
                continue

            score = float(score)

            chunk = chunks[index_position]

            print(
                f"Rank       : {rank}"
            )

            print(
                f"Similarity : {score:.4f}"
            )

            print(
                f"Source     : {chunk['source']}"
            )

            print(
                f"Section    : {chunk['section']}"
            )

            print("-" * 70)

            # Apply threshold
            if score < SIMILARITY_THRESHOLD:

                print(
                    "Result rejected "
                    "(below threshold)"
                )

                print("-" * 70)

                continue

            retrieved_chunks.append({
                "rank": rank,
                "score": score,
                "source": chunk["source"],
                "section": chunk["section"],
                "content": chunk["content"]
            })

        # No relevant results
        if not retrieved_chunks:

            return {
                "success": True,
                "found": False,
                "context": "",
                "results": [],
                "message": (
                    "No sufficiently relevant "
                    "information was found "
                    "in the knowledge base."
                )
            }

        # Build context for LLM
        context_parts = []

        for result in retrieved_chunks:

            context_parts.append(
                f"SOURCE: {result['source']}\n"
                f"SECTION: {result['section']}\n"
                f"SIMILARITY: {result['score']:.4f}\n"
                f"CONTENT:\n{result['content']}"
            )

        context = "\n\n".join(
            context_parts
        )

        return {
            "success": True,
            "found": True,
            "context": context,
            "results": retrieved_chunks,
            "message": (
                f"Retrieved "
                f"{len(retrieved_chunks)} "
                f"relevant knowledge section(s)."
            )
        }

    except Exception as e:

        return {
            "success": False,
            "found": False,
            "context": "",
            "results": [],
            "error": str(e)
        }