"""Document indexing pipeline: chunking, embedding, and vector storage.

Supports two embedding providers:
- FLP: Free Law Project's fine-tuned ModernBERT model (free, local, legal-domain)
- OpenAI: text-embedding-3-small (API-based, ~$0.02/1M tokens)

Uses ChromaDB for local vector storage.
"""

import logging
from typing import Optional

import chromadb

import config
from classifier import classify_document
from courtlistener import BankruptcyCase, DocketEntry, RecapDocument

logger = logging.getLogger(__name__)

# Lazy-loaded embedding model
_flp_model = None


def _get_flp_model():
    """Lazy-load the FLP ModernBERT model."""
    global _flp_model
    if _flp_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading FLP embedding model: {config.FLP_MODEL_NAME}")
        _flp_model = SentenceTransformer(config.FLP_MODEL_NAME)
        logger.info("Model loaded.")
    return _flp_model


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.
        is_query: If True, prefix with query instruction (FLP model).

    Returns:
        List of embedding vectors.
    """
    if config.EMBEDDING_PROVIDER == "flp":
        model = _get_flp_model()
        # FLP's ModernBERT uses prefixes for asymmetric search
        prefix = "search_query: " if is_query else "search_document: "
        prefixed = [prefix + t for t in texts]
        embeddings = model.encode(prefixed, show_progress_bar=True)
        return embeddings.tolist()

    elif config.EMBEDDING_PROVIDER == "openai":
        import openai

        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        # OpenAI handles batches up to 2048
        import time as _time

        import time as _time

        all_embeddings = []
        batch_size = 128
        for i in range(0, len(texts), batch_size):
            if i > 0:
                _time.sleep(1)
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(
                model=config.OPENAI_EMBEDDING_MODEL, input=batch
            )
            all_embeddings.extend([d.embedding for d in resp.data])
        return all_embeddings

    else:
        raise ValueError(f"Unknown embedding provider: {config.EMBEDDING_PROVIDER}")


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks by token-approximate word count.

    Uses a simple word-based approximation (1 token ≈ 0.75 words).

    Args:
        text: Full document text.
        chunk_size: Target chunk size in approximate tokens.
        overlap: Number of overlap tokens between chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    # Approximate: 1 token ≈ 0.75 words, so chunk_size tokens ≈ chunk_size * 0.75 words
    words = text.split()
    word_chunk_size = max(1, int(chunk_size * 0.75))
    word_overlap = max(0, int(overlap * 0.75))

    if len(words) <= word_chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + word_chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += word_chunk_size - word_overlap

    return chunks


def build_chunk_id(docket_id: int, doc_id: int, chunk_idx: int) -> str:
    """Create a unique ID for a chunk."""
    return f"d{docket_id}_doc{doc_id}_c{chunk_idx}"


class CaseIndex:
    """Manages the vector index for a single bankruptcy case."""

    def __init__(self, docket_id: int):
        self.docket_id = docket_id
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection_name = f"case_{docket_id}"

    def exists(self) -> bool:
        """Check if this case has already been indexed."""
        try:
            col = self.client.get_collection(self.collection_name)
            return col.count() > 0
        except Exception:
            return False

    def delete(self):
        """Delete the index for this case."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted index for case {self.docket_id}")
        except Exception:
            pass

    def index_case(self, case: BankruptcyCase) -> int:
        """Index all available documents in a case.

        Args:
            case: A loaded BankruptcyCase object with documents.

        Returns:
            Number of chunks indexed.
        """
        # Delete existing index if any
        self.delete()

        collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        all_chunks = []
        all_metadatas = []
        all_ids = []

        doc_count = 0
        for entry in case.entries:
            for doc in entry.documents:
                if not doc.plain_text or not doc.plain_text.strip():
                    continue

                doc_count += 1
                doc_type = classify_document(doc.description or entry.description)
                chunks = chunk_text(doc.plain_text)

                for i, chunk_text_str in enumerate(chunks):
                    chunk_id = build_chunk_id(case.docket_id, doc.id, i)
                    metadata = {
                        "docket_entry_id": entry.id,
                        "entry_number": entry.entry_number or 0,
                        "ecf_number": doc.ecf_number or f"Entry {entry.entry_number or '?'}",
                        "description": (doc.description or entry.description or "")[:500],
                        "doc_type": doc_type.value,
                        "date_filed": doc.date_filed or entry.date_filed or "",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "doc_id": doc.id,
                        "source": "document",
                    }
                    all_chunks.append(chunk_text_str)
                    all_metadatas.append(metadata)
                    all_ids.append(chunk_id)

        # Index docket entry descriptions
        entry_count = 0
        for entry in case.entries:
            if not entry.description or not entry.description.strip():
                continue
            entry_count += 1
            doc_type = classify_document(entry.description)
            chunk_id = f"d{case.docket_id}_entry{entry.id}_desc"
            metadata = {
                "docket_entry_id": entry.id,
                "entry_number": entry.entry_number or 0,
                "ecf_number": f"ECF No. {entry.entry_number}" if entry.entry_number else "Unnumbered",
                "description": entry.description[:500],
                "doc_type": doc_type.value,
                "date_filed": entry.date_filed or "",
                "chunk_index": 0,
                "total_chunks": 1,
                "doc_id": 0,
                "source": "docket_entry",
            }
            all_chunks.append(entry.description)
            all_metadatas.append(metadata)
            all_ids.append(chunk_id)

        if not all_chunks:
            logger.warning("No content to index!")
            return 0

        logger.info(
            f"Indexing {len(all_chunks)} chunks from {doc_count} documents "
            f"and {entry_count} docket entry descriptions..."
        )

        # Embed in batches
        embeddings = embed_texts(all_chunks, is_query=False)

        # Add to ChromaDB in batches (max 41666 per call)
        batch_size = 5000
        for i in range(0, len(all_chunks), batch_size):
            end = min(i + batch_size, len(all_chunks))
            collection.add(
                ids=all_ids[i:end],
                embeddings=embeddings[i:end],
                documents=all_chunks[i:end],
                metadatas=all_metadatas[i:end],
            )

        logger.info(f"Indexed {len(all_chunks)} chunks successfully.")
        return len(all_chunks)

    def index_single_document(
        self, case: BankruptcyCase, entry: DocketEntry, doc: RecapDocument
    ) -> int:
        """Incrementally index a single document into the existing collection.

        Args:
            case: The BankruptcyCase (for docket_id).
            entry: The DocketEntry this document belongs to.
            doc: The RecapDocument with plain_text to index.

        Returns:
            Number of chunks added.

        Raises:
            RuntimeError: If no existing index is found.
        """
        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception:
            raise RuntimeError(
                f"No existing index for case {self.docket_id}. "
                "Run index_case() first."
            )

        if not doc.plain_text or not doc.plain_text.strip():
            logger.warning(f"Document {doc.id} has no text to index")
            return 0

        doc_type = classify_document(doc.description or entry.description)
        chunks = chunk_text(doc.plain_text)

        all_chunks = []
        all_metadatas = []
        all_ids = []

        for i, chunk_text_str in enumerate(chunks):
            chunk_id = build_chunk_id(case.docket_id, doc.id, i)
            metadata = {
                "docket_entry_id": entry.id,
                "entry_number": entry.entry_number or 0,
                "ecf_number": doc.ecf_number or f"Entry {entry.entry_number or '?'}",
                "description": (doc.description or entry.description or "")[:500],
                "doc_type": doc_type.value,
                "date_filed": doc.date_filed or entry.date_filed or "",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "doc_id": doc.id,
                "source": "document",
            }
            all_chunks.append(chunk_text_str)
            all_metadatas.append(metadata)
            all_ids.append(chunk_id)

        if not all_chunks:
            return 0

        logger.info(f"Indexing {len(all_chunks)} chunks from document {doc.id}...")
        embeddings = embed_texts(all_chunks, is_query=False)

        collection.add(
            ids=all_ids,
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
        )

        logger.info(f"Indexed {len(all_chunks)} chunks for document {doc.id}")
        return len(all_chunks)

    def reindex_descriptions(self, case: BankruptcyCase) -> int:
        """Re-index only docket entry descriptions, preserving document chunks.

        Deletes all existing description chunks (source="docket_entry") and
        re-embeds current descriptions. Much faster than full re-indexing since
        document chunks are left untouched.

        Args:
            case: The BankruptcyCase with updated descriptions.

        Returns:
            Number of description chunks indexed.
        """
        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception:
            logger.warning("No existing index found — falling back to full index_case()")
            return self.index_case(case)

        # Delete all existing description chunks
        existing = collection.get(where={"source": "docket_entry"})
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
            logger.info(f"Deleted {len(existing['ids'])} old description chunks")

        # Re-index descriptions (same logic as index_case lines 199-221)
        all_chunks = []
        all_metadatas = []
        all_ids = []

        entry_count = 0
        for entry in case.entries:
            if not entry.description or not entry.description.strip():
                continue
            entry_count += 1
            doc_type = classify_document(entry.description)
            chunk_id = f"d{case.docket_id}_entry{entry.id}_desc"
            metadata = {
                "docket_entry_id": entry.id,
                "entry_number": entry.entry_number or 0,
                "ecf_number": f"ECF No. {entry.entry_number}" if entry.entry_number else "Unnumbered",
                "description": entry.description[:500],
                "doc_type": doc_type.value,
                "date_filed": entry.date_filed or "",
                "chunk_index": 0,
                "total_chunks": 1,
                "doc_id": 0,
                "source": "docket_entry",
            }
            all_chunks.append(entry.description)
            all_metadatas.append(metadata)
            all_ids.append(chunk_id)

        if not all_chunks:
            logger.warning("No descriptions to index")
            return 0

        logger.info(f"Re-indexing {entry_count} docket entry descriptions...")
        embeddings = embed_texts(all_chunks, is_query=False)

        batch_size = 5000
        for i in range(0, len(all_chunks), batch_size):
            end = min(i + batch_size, len(all_chunks))
            collection.add(
                ids=all_ids[i:end],
                embeddings=embeddings[i:end],
                documents=all_chunks[i:end],
                metadatas=all_metadatas[i:end],
            )

        logger.info(f"Re-indexed {len(all_chunks)} description chunks")
        return len(all_chunks)

    def _build_where_filter(
        self,
        doc_type_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        entry_ids: Optional[list[int]] = None,
    ) -> Optional[dict]:
        """Build a ChromaDB where filter from optional constraints.

        Args:
            doc_type_filter: Filter to a specific doc_type value.
            source_filter: "document" or "docket_entry".
            entry_ids: Restrict to chunks from these docket_entry_ids.

        Returns:
            A ChromaDB where dict, or None if no filters.
        """
        conditions = []

        if doc_type_filter:
            conditions.append({"doc_type": doc_type_filter})
        if source_filter:
            conditions.append({"source": source_filter})
        if entry_ids:
            conditions.append({"docket_entry_id": {"$in": entry_ids}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def query(
        self,
        question: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        doc_type_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        entry_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        """Query the index with a natural language question.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve.
            doc_type_filter: Optional DocType value to filter by.
            source_filter: Optional "document" or "docket_entry" to
                restrict which pool of content to search.
            entry_ids: Optional list of docket_entry_ids to restrict to.

        Returns:
            List of dicts with keys: text, metadata, distance
        """
        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception:
            logger.error(f"No index found for case {self.docket_id}")
            return []

        where_filter = self._build_where_filter(doc_type_filter, source_filter, entry_ids)

        # Embed question
        q_embedding = embed_texts([question], is_query=True)[0]

        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Flatten results
        chunks = []
        if results and results["documents"]:
            for i, doc_text in enumerate(results["documents"][0]):
                chunks.append(
                    {
                        "text": doc_text,
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )

        return chunks

    def query_descriptions(
        self,
        question: str,
        top_k: int = 20,
        doc_type_filter: Optional[str] = None,
    ) -> list[dict]:
        """Search only docket entry descriptions (stage 1 of two-stage retrieval).

        Searches the full docket (all entries, not just those with documents)
        to find relevant entries by their descriptions.

        Args:
            question: The user's question.
            top_k: Number of description matches to return.
            doc_type_filter: Optional DocType value to filter by.

        Returns:
            List of dicts with keys: text, metadata, distance
        """
        return self.query(
            question=question,
            top_k=top_k,
            doc_type_filter=doc_type_filter,
            source_filter="docket_entry",
        )

    def query_documents(
        self,
        question: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        doc_type_filter: Optional[str] = None,
        entry_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        """Search only full document chunks (stage 2 of two-stage retrieval).

        Args:
            question: The user's question.
            top_k: Number of document chunks to return.
            doc_type_filter: Optional DocType value to filter by.
            entry_ids: If provided, only search chunks from these entries.

        Returns:
            List of dicts with keys: text, metadata, distance
        """
        return self.query(
            question=question,
            top_k=top_k,
            doc_type_filter=doc_type_filter,
            source_filter="document",
            entry_ids=entry_ids,
        )
