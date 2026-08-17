import os
from langsmith import Client as LangSmithClient
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="anthropic/claude-haiku-4.5",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    max_tokens=1024,
    temperature=0
    )
prompt = LangSmithClient().pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)

generation_chain = prompt | llm | StrOutputParser()