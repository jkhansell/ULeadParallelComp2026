import os
import shutil
from typing import List, Any, Optional
from pypdf import PdfReader

# Core LlamaIndex & Columnar LanceDB Storage primitives
import lancedb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode, Document as LIDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding as LlamaIndexHFEmbedding

# Interface Base Contract Rules
from db_base import BaseVectorStoreEngine


class LanceVectorEngine(BaseVectorStoreEngine):
    """
    LanceDB implementation powered natively by LlamaIndex's IngestionPipeline
    for high-throughput columnar vector operations and file-hash deduplication.
    """
    def __init__(self, uri: str = "./lancedb_storage", table_name: str = "quickstart", state_dir: str = "./pipeline_state"):
        self.uri = uri
        self.table_name = table_name
        self.state_dir = state_dir
        
        os.makedirs(state_dir, exist_ok=True)
        
        # Instantiate EmbeddingGemma 300M with explicit instructions
        self.embed_model = LlamaIndexHFEmbedding(
            model_name="google/embeddinggemma-300m",
            query_instruction="task: search result | query: ",
            text_instruction="title: none | text: ",
        )
        
        self.db_connection = None
        self.vector_store = None
        self.pipeline_docstore = None
        
        self.connect()

    def connect(self) -> lancedb.DBConnection:
        """Establishes connections to local LanceDB file matrices and pipeline states."""
        # 1. Connect to the serverless LanceDB backend path
        self.db_connection = lancedb.connect(self.uri)
        
        # 2. Mount into LlamaIndex vector store adapter interface, forcing Cosine distance metric math
        self.vector_store = LanceDBVectorStore(
            uri=self.uri, 
            table_name=self.table_name,
        )

        # 3. Initialize or reload the persistent state document tracker
        docstore_path = os.path.join(self.state_dir, "lance_pipeline_docstore.json")
        if os.path.exists(docstore_path):
            self.pipeline_docstore = SimpleDocumentStore.from_persist_path(docstore_path)
        else:
            self.pipeline_docstore = SimpleDocumentStore()
            
        return self.db_connection

    def _extract_clean_documents(self, data_dir: str) -> List[LIDocument]:
        """Scans the data directory, extracting raw text while removing running headers/footers."""
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
                        
                        # Coordinate checker callback function
                        def visitor_body(text, cm, tm, font_dict, font_size):
                            y = tm[5]  # Absolute vertical point location coordinate
                            if 50 < y < 730:  # Filters out header and footer text
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
                    print(f"Error parsing PDF target {file}: {e}")
                    
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
        """Assembles and runs the native IngestionPipeline to update LanceDB tables incrementally."""
        raw_docs = self._extract_clean_documents(data_dir)
        if not raw_docs:
            print("No raw text assets found in target paths.")
            return

        # 1. Define transformations for the ingestion run
        semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1, 
            breakpoint_percentile_threshold=breakpoint_percentile, 
            embed_model=self.embed_model
        )

        # 2. Instantiate pipeline with the LanceDB vector store and state docstore attached
        pipeline = IngestionPipeline(
            transformations=[semantic_splitter, self.embed_model],
            vector_store=self.vector_store,
            docstore=self.pipeline_docstore,
            docstore_strategy=DocstoreStrategy.UPSERTS
        )

        # 3. Execute the pipeline (unchanged documents are automatically skipped)
        print("\nExecuting Native LlamaIndex Ingestion Pipeline (LanceDB Columnar Engine)...")
        processed_nodes = pipeline.run(documents=raw_docs, show_progress=True)
        print(f"Pipeline processed and committed {len(processed_nodes)} nodes to LanceDB.")

        # 4. Save the updated tracking hashes back to disk
        docstore_path = os.path.join(self.state_dir, "lance_pipeline_docstore.json")
        self.pipeline_docstore.persist(persist_path=docstore_path)

    def get_retrieval_engine(self, similarity_top_k: int = 3, filters: Optional[Any] = None, **kwargs) -> Any:
        """Returns a non-synthesizing vector search retriever interface linked to LanceDB."""
        index = VectorStoreIndex.from_vector_store(self.vector_store, embed_model=self.embed_model)
        return index.as_retriever(similarity_top_k=similarity_top_k, filters=filters, **kwargs)

if __name__ == "__main__":
    print("Testing Ingestion Pipeline with LanceDB Columnar Backend...")
    import uuid
    test_uri = f"./lancedb_test_{uuid.uuid4().hex[:8]}"
    test_state = f"./pipeline_state_test_{uuid.uuid4().hex[:8]}"
    dummy_data = "./test_data_dir"
    
    os.makedirs(dummy_data, exist_ok=True)
    with open(os.path.join(dummy_data, "sample.txt"), "w") as f:
        f.write("High-Performance Computing configurations require balancing memory channels.\n"
                "Warp divergence tracking handles serialized execution paths across grid blocks.")

    try:
        engine = LanceVectorEngine(uri=test_uri, state_dir=test_state)
        
        print("\n--- Pipeline Run 1 (Cold Ingestion) ---")
        engine.ingest_and_index(data_dir=dummy_data)
        
        print("\n--- Pipeline Run 2 (Hot Ingestion - Should show 0 nodes processed) ---")
        engine.ingest_and_index(data_dir=dummy_data)
        
        retriever = engine.get_retrieval_engine(similarity_top_k=1)
        res = retriever.retrieve("What tracks serialized execution paths?")
        print(f"\nRetrieved Text Snippet:\n{res[0].text if res else 'None discovered'}")
        
    finally:
        for folder in [test_uri, test_state, dummy_data]:
            if os.path.exists(folder): 
                shutil.rmtree(folder)