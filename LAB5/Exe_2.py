from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


class ResearchExtraction(BaseModel):
    research_question: str = Field(description="The main question investigated")
    method: str = Field(description="The research method used")
    key_finding: str = Field(description="The most important result")


step1_parser = PydanticOutputParser(pydantic_object=ResearchExtraction)

step1_prompt = PromptTemplate(
    template="""
Extract the research question, method, and key finding from this paper abstract.

Abstract:
{abstract_text}

{format_instructions}
""",
    input_variables=["abstract_text"],
    partial_variables={
        "format_instructions": step1_parser.get_format_instructions()
    },
)

step2_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a science communicator explaining research to a general audience.",
    ),
    (
        "human",
        "Write a clear, accurate plain-language summary using only the structured "
        "research information below. Do not assume details that are not provided.\n\n"
        "Research question: {research_question}\n"
        "Method: {method}\n"
        "Key finding: {key_finding}",
    ),
])

step1_chain = step1_prompt | model | step1_parser
step2_chain = step2_prompt | model | StrOutputParser()


def abstract_to_layperson_summary(abstract_text: str) -> dict:
    """Extract an abstract's main points and explain them for non-experts."""
    structured_extraction = step1_chain.invoke({"abstract_text": abstract_text})
    layperson_summary = step2_chain.invoke(structured_extraction.model_dump())
    return {
        "structured_extraction": structured_extraction,
        "layperson_summary": layperson_summary,
    }


if __name__ == "__main__":
    abstract_text = input("Enter paper abstract: ")
    result = abstract_to_layperson_summary(abstract_text)

    print("\nStep 1 - Structured extraction:")
    print(result["structured_extraction"].model_dump_json(indent=2))
    print("\nStep 2 - Layperson summary:")
    print(result["layperson_summary"])