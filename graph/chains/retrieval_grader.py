import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    """

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'.")

llm = ChatOpenAI(model="anthropic/claude-haiku-4.5",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    max_tokens=1024,
    temperature=0
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_prompt = """
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human","Retrieved document: {document} User question: {question}")
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
