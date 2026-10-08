import os
import pickle

import faiss

from rag.embeddings import create_embeddings


RAG_DIRECTORY = os.path.dirname(__file__)

KNOWLEDGE_BASE_PATH = os.path.join(
    RAG_DIRECTORY,
    "knowledge_base.txt"
)

VECTOR_INDEX_PATH = os.path.join(
    RAG_DIRECTORY,
    "knowledge_base.index"
)

CHUNKS_PATH = os.path.join(
    RAG_DIRECTORY,
    "knowledge_chunks.pkl"
)


def load_and_chunk_document():
    """
    Load the knowledge base and create
    section-based chunks.

    Each policy heading stays together
    with its corresponding content.
    """

    with open(
        KNOWLEDGE_BASE_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        content = file.read()

    lines = content.splitlines()

    chunks = []

    current_section = None
    current_content = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Detect section headings
        if line.isupper():

            # Save previous section
            if current_section is not None:

                chunk_text = (
                    current_section
                    + "\n"
                    + "\n".join(current_content)
                )

                chunks.append({
                    "source": "knowledge_base.txt",
                    "section": current_section,
                    "content": chunk_text
                })

            current_section = line
            current_content = []

        else:

            current_content.append(line)

    # Save final section
    if current_section is not None:

        chunk_text = (
            current_section
            + "\n"
            + "\n".join(current_content)
        )

        chunks.append({
            "source": "knowledge_base.txt",
            "section": current_section,
            "content": chunk_text
        })

    return chunks


def build_vector_store():

    print("\nLoading knowledge base...")

    chunks = load_and_chunk_document()

    if not chunks:
        raise ValueError(
            "No document chunks were found."
        )

    print(
        f"Total chunks created: {len(chunks)}"
    )

    print("\nCreated chunks:")

    for i, chunk in enumerate(chunks):

        print("\n" + "=" * 60)

        print(
            f"CHUNK {i + 1}"
        )

        print("=" * 60)

        print(
            f"Source  : {chunk['source']}"
        )

        print(
            f"Section : {chunk['section']}"
        )

        print(
            chunk["content"]
        )

    # Extract only the text for embeddings
    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    print(
        "\nCreating embeddings..."
    )

    embeddings = create_embeddings(
        texts
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print(
        f"Vectors stored in FAISS: "
        f"{index.ntotal}"
    )

    # Save FAISS index
    faiss.write_index(
        index,
        VECTOR_INDEX_PATH
    )

    # Save chunk metadata
    with open(
        CHUNKS_PATH,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    print(
        "\nVector store successfully created."
    )

    print(
        f"FAISS index : {VECTOR_INDEX_PATH}"
    )

    print(
        f"Metadata    : {CHUNKS_PATH}"
    )


if __name__ == "__main__":
    build_vector_store()