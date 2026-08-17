import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

class RouteQuery(BaseModel):
    """
    Route a user query to the most relevant datasource
    """

    datasource: Literal["vectorstore","websearch"] = Field(...,
                                                           description="Given a user question choose to route it"
                                                                       " to web search or a vectorstore"
                                                           )

llm = ChatOpenAI(model="anthropic/claude-haiku-4.5",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    max_tokens=1024,
    temperature=0
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

system_prompt = """
You are an expert at routing an user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering and adversial attacks on llms.
Use the vectorstore for questions on these topics. For all else, use web search
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human","{question}")
    ]
)

question_router = route_prompt | structured_llm_router
