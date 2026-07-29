import os
import time
from tabulate import tabulate # pip install tabulate (for crisp console logging)

# Import the completed self-contained engines
from db_chroma import ChromaVectorEngine
from db_lance import LanceVectorEngine


def run_benchmark(data_directory: str = "./bench_data", query_string: str = "Qué derechos tienen los sindicatos?"):
    print("=" * 60)

    os.makedirs("./speedtest", exist_ok=True)
    
    # Validate target directory state before proceeding
    if not os.path.exists(data_directory) or not os.listdir(data_directory):
        print(f"Error: Target data folder '{data_directory}' is missing or empty.")
        print("Please place your tutorial PDFs/files inside it before running diagnostics.")
        return

    print(f"Starting Vector Database Performance Diagnostics on: '{data_directory}'")
    print("=" * 60)

    # 1. Instantiate both engines with fresh, temporary disk footprints
    chroma_engine = ChromaVectorEngine(persist_dir="./speedtest/chroma_bench_db", collection_name="bench_table", state_dir="./speedtest/chroma_bench_state")
    lance_engine = LanceVectorEngine(uri="./speedtest/lance_bench_db", table_name="bench_table", state_dir="./speedtest/lance_bench_state")

    results = {}

    # ==========================================
    # BENCHMARK SYSTEM A: CHROMADB
    # ==========================================
    print("\n[Executing Chroma DB Benchmarks...]")
    
    # Cold Ingestion (Parse + Semantic Split + Embed + Write)
    t0 = time.perf_counter()
    chroma_engine.ingest_and_index(data_dir=data_directory)
    chroma_cold_time = time.perf_counter() - t0
    
    # Hot Ingestion (State-checked Hash Verification loop)
    t0 = time.perf_counter()
    chroma_engine.ingest_and_index(data_dir=data_directory)
    chroma_hot_time = time.perf_counter() - t0
    
    # Query Retrieval Latency
    chroma_retriever = chroma_engine.get_retrieval_engine(similarity_top_k=3)
    t0 = time.perf_counter()
    chroma_nodes = chroma_retriever.retrieve(query_string)
    chroma_query_time = time.perf_counter() - t0

    results["ChromaDB"] = {
        "cold_ingest": chroma_cold_time,
        "hot_ingest": chroma_hot_time,
        "query_latency": chroma_query_time,
        "nodes_returned": len(chroma_nodes)
    }

    # ==========================================
    # BENCHMARK SYSTEM B: LANCEDB
    # ==========================================
    print("\n[Executing LanceDB Benchmarks...]")
    
    # Cold Ingestion (Parse + Semantic Split + Embed + Write)
    t0 = time.perf_counter()
    lance_engine.ingest_and_index(data_dir=data_directory)
    lance_cold_time = time.perf_counter() - t0
    
    # Hot Ingestion (State-checked Hash Verification loop)
    t0 = time.perf_counter()
    lance_engine.ingest_and_index(data_dir=data_directory)
    lance_hot_time = time.perf_counter() - t0
    
    # Query Retrieval Latency
    lance_retriever = lance_engine.get_retrieval_engine(similarity_top_k=3)
    t0 = time.perf_counter()
    lance_nodes = lance_retriever.retrieve(query_string)
    lance_query_time = time.perf_counter() - t0

    results["LanceDB"] = {
        "cold_ingest": lance_cold_time,
        "hot_ingest": lance_hot_time,
        "query_latency": lance_query_time,
        "nodes_returned": len(lance_nodes)
    }

    # ==========================================
    # REPORTING LOG MATRIX GENERATION
    # ==========================================
    print("\n" + "=" * 65)
    print("                FINAL PERFORMANCE COMPARISON MATRIX")
    print("=" * 65)
    
    table_data = [
        [
            "Cold Ingestion (s)\n[Total Data Load]", 
            f"{results['ChromaDB']['cold_ingest']:.4f} s", 
            f"{results['LanceDB']['cold_ingest']:.4f} s"
        ],
        [
            "Hot Ingestion (s)\n[Deduplication Skip]", 
            f"{results['ChromaDB']['hot_ingest']:.4f} s", 
            f"{results['LanceDB']['hot_ingest']:.4f} s"
        ],
        [
            "Vector Query Retrieval (s)\n[Top-K Nearest Neighbor Search]", 
            f"{results['ChromaDB']['query_latency']:.4f} s", 
            f"{results['LanceDB']['query_latency']:.4f} s"
        ],
        [
            "Retrieved Node Integrity (count)", 
            f"{results['ChromaDB']['nodes_returned']} nodes", 
            f"{results['LanceDB']['nodes_returned']} nodes"
        ]
    ]
    
    print(tabulate(table_data, headers=["Metric Parameter", "ChromaDB Backend", "LanceDB Backend"], tablefmt="grid"))
    print("=" * 65)

if __name__ == "__main__":
    run_benchmark()