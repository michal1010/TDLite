from langchain_ollama import ChatOllama

from .config import MODEL_NAME, NUM_CTX


def get_llm(temperature: float = 0.2) -> ChatOllama:
    '''Changed model to qwen2.5-coder:3b, takes a bit longer usually, but no more 5 min+ thinking.
    Change model back to model="qwen3:0.6b" if your laptop cannot handle.
    '''
    return ChatOllama(
        model="qwen2.5-coder:7b",
        temperature=temperature,
        num_ctx=NUM_CTX,
    )