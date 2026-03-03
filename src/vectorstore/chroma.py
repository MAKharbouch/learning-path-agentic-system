"""ChromaDB collection factory for course embeddings.

Provides functions to create and access a persistent ChromaDB collection
for storing and querying course embeddings.

Supports OpenAI and local (SentenceTransformers) embedding providers
via the EMBEDDING_PROVIDER env var.

IMPORTANT: ChromaDB requires IDs to be strings, not integers.
Always pass string IDs when adding documents to a collection.
"""

from pathlib import Path
from typing import Any

import chromadb


def get_embedding_function():
    """Return an embedding function based on the configured provider.

    Reads EMBEDDING_PROVIDER from config:
    - "openai": Uses OpenAI text-embedding-3-small (requires OPENAI_API_KEY)
    - "local": Uses SentenceTransformers all-MiniLM-L6-v2 (downloads ~80MB on first run)

    Returns:
        A ChromaDB-compatible embedding function.

    Raises:
        ValueError: If EMBEDDING_PROVIDER is not "openai" or "local".
    """
    from config import EMBEDDING_PROVIDER

    provider = EMBEDDING_PROVIDER.lower()

    if provider == "openai":
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

        from config import EMBEDDING_MODEL, OPENAI_API_KEY

        return OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY, model_name=EMBEDDING_MODEL
        )

    if provider == "local":
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        return SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{provider}'. Use 'openai' or 'local'."
    )


def get_chroma_client(persist_path: Path | None = None) -> Any:
    """Return a persistent ChromaDB client.

    Args:
        persist_path: Directory for ChromaDB storage.
            If None, uses config.CHROMA_PATH.

    Returns:
        A ChromaDB PersistentClient instance.
    """
    if persist_path is None:
        from config import CHROMA_PATH

        persist_path = CHROMA_PATH

    persist_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_path))


def get_collection(persist_path: Path | None = None) -> chromadb.Collection:
    """Return the course embeddings collection with the configured embedding function.

    Uses the embedding provider configured via EMBEDDING_PROVIDER env var.

    IMPORTANT: ChromaDB IDs must be strings, not integers.

    Args:
        persist_path: Directory for ChromaDB storage.
            If None, uses config.CHROMA_PATH.

    Returns:
        A ChromaDB Collection with embeddings configured.
    """
    from config import CHROMA_COLLECTION_NAME

    client = get_chroma_client(persist_path)
    embedding_fn = get_embedding_function()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_fn,  # type: ignore[arg-type]
    )


def get_collection_no_embeddings(
    persist_path: Path | None = None,
) -> chromadb.Collection:
    """Return the course embeddings collection without an embedding function.

    For testing and inspection only. Embeddings must be provided manually
    when adding documents.

    Args:
        persist_path: Directory for ChromaDB storage.
            If None, uses config.CHROMA_PATH.

    Returns:
        A ChromaDB Collection without an embedding function.
    """
    from config import CHROMA_COLLECTION_NAME

    client = get_chroma_client(persist_path)
    return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
