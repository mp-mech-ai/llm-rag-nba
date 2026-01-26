# utils/vector_store.py
import os
import pickle
import faiss
import numpy as np
import logging
import logfire

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

from mistralai import Mistral
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from nba_assistant.config.config import (
    MISTRAL_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_FILE,
    DOCUMENT_CHUNKS_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logfire.configure()

# ---------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------

class RawDocument(BaseModel):
    page_content: str = Field(min_length=1)
    metadata: Dict[str, Any]


class DocumentChunk(BaseModel):
    id: str
    text: str = Field(min_length=20)
    metadata: Dict[str, Any]


class EmbeddedChunk(DocumentChunk):
    embedding: List[float] = Field(min_items=10)


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------

class VectorStoreInitializationError(Exception):
    pass


class EmbeddingGenerationError(Exception):
    pass


# ---------------------------------------------------------------------
# Vector store manager
# ---------------------------------------------------------------------

class VectorStoreManager:
    """Manages creation, loading, and querying of a Faiss vector index."""

    def __init__(self):
        with logfire.span("vector_store_init"):
            if not MISTRAL_API_KEY:
                raise VectorStoreInitializationError(
                    "MISTRAL_API_KEY is missing"
                )

            self.index: Optional[faiss.Index] = None
            self.document_chunks: List[DocumentChunk] = []
            self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)

            self._load_index_and_chunks()

    # -----------------------------------------------------------------

    def _load_index_and_chunks(self):
        with logfire.span("load_index_and_chunks"):
            if not (
                os.path.exists(FAISS_INDEX_FILE)
                and os.path.exists(DOCUMENT_CHUNKS_FILE)
            ):
                logger.warning("Faiss index or chunks file not found")
                return

            try:
                self.index = faiss.read_index(FAISS_INDEX_FILE)

                with open(DOCUMENT_CHUNKS_FILE, "rb") as f:
                    raw_chunks = pickle.load(f)

                self.document_chunks = [
                    DocumentChunk.model_validate(chunk)
                    for chunk in raw_chunks
                ]

                logfire.info(
                    "Vector store loaded",
                    vectors=self.index.ntotal,
                    chunks=len(self.document_chunks),
                )

            except Exception as e:
                raise VectorStoreInitializationError(
                    "Failed to load vector store"
                ) from e

    # -----------------------------------------------------------------

    def _split_documents_to_chunks(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[DocumentChunk]:

        with logfire.span("document_chunking", documents=len(documents)):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True,
            )

            validated_docs = [
                RawDocument.model_validate(doc)
                for doc in documents
            ]

            chunks: List[DocumentChunk] = []
            doc_counter = 0

            for doc in validated_docs:
                lc_doc = Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                )

                split_docs = splitter.split_documents([lc_doc])

                for i, chunk in enumerate(split_docs):
                    chunk_model = DocumentChunk(
                        id=f"{doc_counter}_{i}",
                        text=chunk.page_content,
                        metadata={
                            **chunk.metadata,
                            "chunk_id_in_doc": i,
                            "start_index": chunk.metadata.get("start_index", -1),
                        },
                    )
                    chunks.append(chunk_model)

                doc_counter += 1

            logfire.info("Chunks created", total_chunks=len(chunks))
            return chunks

    # -----------------------------------------------------------------

    def _generate_embeddings(
        self,
        chunks: List[DocumentChunk],
    ) -> np.ndarray:

        with logfire.span("embedding_generation", chunks=len(chunks)):
            if not chunks:
                raise EmbeddingGenerationError("No chunks to embed")

            all_embeddings: List[List[float]] = []

            for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
                texts = [chunk.text for chunk in batch]

                try:
                    response = self.mistral_client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        inputs=texts,
                    )
                    embeddings = [d.embedding for d in response.data]
                    all_embeddings.extend(embeddings)

                except Exception as e:
                    raise EmbeddingGenerationError(
                        "Embedding generation failed"
                    ) from e

            embeddings_array = np.array(all_embeddings, dtype="float32")

            if embeddings_array.shape[0] != len(chunks):
                raise EmbeddingGenerationError(
                    "Embedding count does not match chunk count"
                )

            logfire.info(
                "Embeddings generated",
                shape=str(embeddings_array.shape),
            )

            return embeddings_array

    # -----------------------------------------------------------------

    def build_index(self, documents: List[Dict[str, Any]]):
        with logfire.span("build_faiss_index"):
            chunks = self._split_documents_to_chunks(documents)
            embeddings = self._generate_embeddings(chunks)

            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)

            self.index = index
            self.document_chunks = chunks

            self._save_index_and_chunks()

    # -----------------------------------------------------------------

    def _save_index_and_chunks(self):
        with logfire.span("save_index_and_chunks"):
            os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
            os.makedirs(os.path.dirname(DOCUMENT_CHUNKS_FILE), exist_ok=True)

            faiss.write_index(self.index, FAISS_INDEX_FILE)

            with open(DOCUMENT_CHUNKS_FILE, "wb") as f:
                pickle.dump(
                    [chunk.model_dump() for chunk in self.document_chunks],
                    f,
                )

            logfire.info(
                "Vector store saved",
                vectors=self.index.ntotal,
                chunks=len(self.document_chunks),
            )

    # -----------------------------------------------------------------

    def search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        with logfire.span("vector_search", k=k):
            if self.index is None or not self.document_chunks:
                return []

            response = self.mistral_client.embeddings.create(
                model=EMBEDDING_MODEL,
                inputs=[query_text],
            )

            query_embedding = np.array(
                [response.data[0].embedding],
                dtype="float32",
            )

            faiss.normalize_L2(query_embedding)

            scores, indices = self.index.search(query_embedding, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                chunk = self.document_chunks[idx]
                results.append(
                    {
                        "score": float(score) * 100,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    }
                )

            logfire.info("Search completed", results=len(results))
            return results


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def create_vector_store_manager() -> VectorStoreManager:
    try:
        return VectorStoreManager()
    except Exception as e:
        raise VectorStoreInitializationError(
            "Vector store initialization failed"
        ) from e
