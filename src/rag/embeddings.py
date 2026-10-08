from sentence_transformers import SentenceTransformer


# ============================================================
# EMBEDDING MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD MODEL
# ============================================================

model = SentenceTransformer(MODEL_NAME)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(texts):

    """
    Convert a list of text documents into vector embeddings.

    Args:
        texts: list[str]

    Returns:
        numpy.ndarray
    """

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

def create_query_embedding(query):

    """
    Convert a single user query into an embedding.
    """

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding