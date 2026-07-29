import os
from typing import List, Dict, Any

# LangChain core prompt and communication utilities
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Local data cluster dependencies
from db_chroma import ChromaVectorEngine
from agent import llm

class RAGPipeline:
    """
    Production-grade execution pipeline that coordinates context retrieval
    from our local vector database with generation via an online LLM.
    """
    def __init__(self, persist_dir: str = "./chroma_db", state_dir: str = "./pipeline_state", top_k: int = 3):
        print("Initializing Core RAG Coordination Layer...")
        
        # 1. Initialize our self-contained, pipeline-backed Chroma backend
        self.db_engine = ChromaVectorEngine(persist_dir=persist_dir, state_dir=state_dir)
        self.retriever = self.db_engine.get_retrieval_engine(similarity_top_k=top_k)
        
        # 2. Build out the online generation engine
        # We use a standard ChatOpenAI wrapper here as an example—swap this with
        # your target online model provider class (e.g., ChatAnthropic, ChatDeepSeek, etc.)
        self.llm = llm
        
        # 3. Formulate custom, technically accurate generation templates
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert technical AI research assistant. Your task is to provide authoritative, "
                "rigorously accurate answers based strictly on the provided context fragments.\n\n"
                "If the context contains equations, code parameters, execution limits, or specific hardware setups, "
                "replicate them exactly. If the answer cannot be found in the context, explicitly state that you "
                "do not have sufficient information.\n\n"
                "--- CONTEXT BACKGROUND ---\n{context}"
            )),
            ("human", "{question}")
        ])

    @staticmethod
    def _format_nodes_with_metadata(nodes) -> str:
        """
        Collects, parses, and explicitly formats matching text nodes alongside 
        their source PDF files and page metadata boundaries.
        """
        formatted_fragments = []
        for idx, node in enumerate(nodes):
            # Extract tracking metadata dictionary values safely from the LlamaIndex node structure
            source_file = node.metadata.get("source", "Unknown Document File")
            page_number = node.metadata.get("page", "N/A")
            
            # Formulate an un-ignorable header boundary around the raw string content chunk
            node_block = (
                f"=== CONTEXT FRAGMENT {idx + 1} ===\n"
                f"[SOURCE FILE]: {source_file}\n"
                f"[SOURCE PAGE]: {page_number}\n"
                f"[CONTENT]:\n{node.text}\n"
                f"================================="
            )
            formatted_fragments.append(node_block)
            
        return "\n\n".join(formatted_fragments)

    def ask(self, query_str: str) -> Dict[str, Any]:
        """
        Executes an end-to-end RAG cycle: retrieves context from Chroma,
        builds the metadata-enriched prompt, and returns the response alongside 
        the full raw context block for logging or debugging output.
        """
        # Retrieve context nodes from our database wrapper
        raw_nodes = self.retriever.retrieve(query_str)
        
        # Format the nodes containing the metadata headers
        context_block = self._format_nodes_with_metadata(raw_nodes)
        
        # Map tracking references cleanly for internal state audits
        sources_manifest = [
            {"source": n.metadata.get("source"), "page": n.metadata.get("page", "N/A")} 
            for n in raw_nodes
        ]
        
        # Execute the LangChain chain execution line
        chain = (
            {
                "context": lambda x: context_block,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        raw_response = chain.invoke(query_str)
        
        return {
            "answer": raw_response,
            "sources": sources_manifest,
            "raw_context": context_block  # Returned directly to allow external printing
        }


if __name__ == "__main__":
    # Ensure active online API keys exist before running verification checks
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: 'OPENAI_API_KEY' environment variable missing.")

    pipeline = RAGPipeline(top_k=10)
    
    # Synchronize local data directory structures incrementally
    print("\nSynchronizing local data directory structures incrementally...")
    pipeline.db_engine.ingest_and_index(data_dir="./data")
    
    test_query = "What are the downsides of the AlphaEarth Foundations model?"
    print(f"\nSending Query to Metadata-Aware RAG Pipeline: '{test_query}'\n")
    
    try:
        output = pipeline.ask(test_query)
        print("=== LLM SYNTHESIZED RESPONSE WITH INLINE SOURCE CITATIONS ===")
        print(output["answer"])
        print("\n" + "="*50)
        
        print("\n=== RAW RETRIEVED CONTEXT USED FOR THE RESPONSE ===")
        print(output["raw_context"])
        print("="*50 + "\n")
        
        print("=== SYSTEM RETURNED AUDIT SOURCES MANIFEST ===")
        print(output["sources"])
    except Exception as e:
        print(f"Generation loop execution failed: {e}")