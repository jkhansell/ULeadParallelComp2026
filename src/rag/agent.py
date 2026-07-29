from langchain_openai import ChatOpenAI

# =========================
# 1. LLM (vLLM endpoint)
# =========================
llm = ChatOpenAI(
    base_url="http://localhost:9001/v1",
    api_key="EMPTY",
    model="Qwen3.5",
    temperature=0.1,
)

