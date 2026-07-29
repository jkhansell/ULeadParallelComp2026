from abc import ABC, abstractmethod
from typing import List, Any, Optional

class BaseVectorStoreEngine(ABC):
    """
    Abstract Base Class establishing the contract for unified 
    ingestion, tracking, and vector indexing operations.
    """
    @abstractmethod
    def connect(self) -> Any:
        """Establishes an active driver channel to the storage backend."""
        pass

    @abstractmethod
    def ingest_and_index(self, data_dir: str = "./data", breakpoint_percentile: float = 85.0) -> None:
        """
        Scans data directories, processes raw assets incrementally, 
        and updates active storage indexes.
        """
        pass

    @abstractmethod
    def get_retrieval_engine(self, similarity_top_k: int = 3, filters: Optional[Any] = None, **kwargs) -> Any:
        """Returns a non-synthesizing vector search retriever interface."""
        pass