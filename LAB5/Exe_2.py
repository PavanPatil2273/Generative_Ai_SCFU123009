from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

abstract = input("Enter paper abstract: ")


prompt1 = PromptTemplate.from_template(
    """
    Read the following research paper abstract.

    Find:
    1. Research question
    2. Research method
    3. Key finding

    Give the answer clearly.

    Abstract:
    {abstract}
    """
)

chain1 = prompt1 | model | StrOutputParser()

result1 = chain1.invoke({
    "abstract": abstract
})

print(result1)

prompt2 = PromptTemplate.from_template(
    """
    Explain the following research information
    in simple language that a normal person can understand.

    Research information:
    {research}

    Do not add information that is not given.
    """
)

chain2 = prompt2 | model | StrOutputParser()

result2 = chain2.invoke({
    "research": result1
})
print(result2)