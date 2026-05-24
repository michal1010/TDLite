from langchain_openai import ChatOpenAI  # works with any OpenAI-compatible endpoint


def get_llm() -> ChatOpenAI:
    '''Changed model to qwen2.5-coder:3b, takes a bit longer usually, but no more 5 min+ thinking.
    Change model back to model="qwen3:0.6b" if your laptop cannot handle.
    '''
    return ChatOpenAI(
        model="qwen2.5-coder:3b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0.2,
    )