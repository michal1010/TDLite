from langchain_ollama import ChatOllama


def get_llm() -> ChatOllama:
    '''Changed model to qwen2.5-coder:3b, takes a bit longer usually, but no more 5 min+ thinking.
    Change model back to model="qwen3:0.6b" if your laptop cannot handle.
    '''
    return ChatOllama(
        model="qwen2.5-coder:3b",
        temperature=0.2,
    )