import os
import shutil
from typing import List, Any, Optional
from pypdf import PdfReader

# Core LlamaIndex & Chroma dependencies
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode, Document as LIDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding as LlamaIndexHFEmbedding

from db_base import BaseVectorStoreEngine


class ChromaVectorEngine(BaseVectorStoreEngine):
    """
    ChromaDB vector engine powered directly by LlamaIndex's native IngestionPipeline.
    """
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "quickstart", state_dir: str = "./pipeline_state"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.state_dir = state_dir
        
        os.makedirs(state_dir, exist_ok=True)
        
        # Instantiate EmbeddingGemma 300M
        self.embed_model = LlamaIndexHFEmbedding(
            model_name="google/embeddinggemma-300m",
            query_instruction="task: search result | query: ",
            text_instruction="title: none | text: ",
        )
        
        self.db = None
        self.chroma_collection = None
        self.vector_store = None
        self.pipeline_docstore = None
        
        self.connect()

    def connect(self) -> chromadb.PersistentClient:
        """Initializes storage structures and loads the local pipeline state docstore."""
        self.db = chromadb.PersistentClient(path=self.persist_dir)

        # 2. Get or build collection target, explicitly enforcing Cosine Distance math
        self.chroma_collection = self.db.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Establishes cosine tracking logic
        )
        
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        # Load or create the persistent document tracking store for the pipeline
        docstore_path = os.path.join(self.state_dir, "chroma_pipeline_docstore.json")
        if os.path.exists(docstore_path):
            self.pipeline_docstore = SimpleDocumentStore.from_persist_path(docstore_path)
        else:
            self.pipeline_docstore = SimpleDocumentStore()
            
        return self.db

    def _extract_clean_documents(self, data_dir: str) -> List[LIDocument]:
        """Extracts text components from directory files, dropping headers and footers."""
        documents = []
        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            return documents

        for file in os.listdir(data_dir):
            full_path = os.path.join(data_dir, file)
            
            if file.lower().endswith(".pdf"):
                try:
                    reader = PdfReader(full_path)
                    for page_num, page in enumerate(reader.pages):
                        page_parts = []
                        def visitor_body(text, cm, tm, font_dict, font_size):
                            y = tm[5]
                            if 50 < y < 740:
                                page_parts.append(text)
                        page.extract_text(visitor_text=visitor_body)
                        text = "".join(page_parts).strip()
                        
                        if text:
                            doc_id = f"{file}_page_{page_num + 1}"
                            documents.append(LIDocument(
                                text=text, id_=doc_id, 
                                metadata={"source": file, "page": page_num + 1}
                            ))
                except Exception as e:
                    print(f"Error parsing PDF {file}: {e}")
                    
            elif file.lower().endswith((".txt", ".md")):
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        if text:
                            documents.append(LIDocument(text=text, id_=file, metadata={"source": file}))
                except Exception as e:
                    print(f"Error reading file {file}: {e}")
        return documents

    def ingest_and_index(self, data_dir: str = "./data", breakpoint_percentile: float = 95.0) -> None:
        """Assembles and runs the native IngestionPipeline."""
        raw_docs = self._extract_clean_documents(data_dir)
        if not raw_docs:
            print("No raw text assets found in target paths.")
            return

        # 1. Define the transformation sequence
        semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1, 
            breakpoint_percentile_threshold=breakpoint_percentile, 
            embed_model=self.embed_model
        )

        # 2. Instantiate the pipeline with its tracking store and vector database attached
        pipeline = IngestionPipeline(
            transformations=[semantic_splitter, self.embed_model],
            vector_store=self.vector_store,
            docstore=self.pipeline_docstore,
            docstore_strategy=DocstoreStrategy.UPSERTS
        )

        # 3. Execute the pipeline. Unchanged files will be skipped automatically.
        print("\nExecuting Native LlamaIndex Ingestion Pipeline...")
        processed_nodes = pipeline.run(documents=raw_docs, show_progress=True)
        print(f"Pipeline processed and committed {len(processed_nodes)} nodes.")

        # 4. Save the document hash map to disk
        docstore_path = os.path.join(self.state_dir, "chroma_pipeline_docstore.json")
        self.pipeline_docstore.persist(persist_path=docstore_path)

    def get_retrieval_engine(self, similarity_top_k: int = 3, filters: Optional[Any] = None, **kwargs) -> Any:
        """Returns a non-synthesizing vector search retriever interface."""
        index = VectorStoreIndex.from_vector_store(self.vector_store, embed_model=self.embed_model)
        return index.as_retriever(similarity_top_k=similarity_top_k, filters=filters, **kwargs)



if __name__ == "__main__":
    print("Testing Ingestion Pipeline with Chroma Backend...")
    import uuid
    test_dir = f"./chroma_db_test_{uuid.uuid4().hex[:8]}"
    test_state = f"./pipeline_state_test_{uuid.uuid4().hex[:8]}"
    data = "./data"

    try:
        engine = ChromaVectorEngine(persist_dir=test_dir, state_dir=test_state)
        
        print("\n--- Pipeline Run 1 (Cold Ingestion) ---")
        engine.ingest_and_index(data_dir=data)
        
        print("\n--- Pipeline Run 2 (Hot Ingestion - Should show 0 nodes processed) ---")
        engine.ingest_and_index(data_dir=data)
        
        retriever = engine.get_retrieval_engine(similarity_top_k=1)
        res = retriever.retrieve("What is AlphaEarth Foundations?")
        print(f"\nRetrieved Text Snippet:\n{res[0].text if res else 'None discovered'}")
        
    finally:
        for folder in [test_dir, test_state]:
            if os.path.exists(folder): shutil.rmtree(folder)