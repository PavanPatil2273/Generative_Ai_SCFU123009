from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model1 = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

class Review(BaseModel):
    Research_question: str = Field(description="Enter Research_question")
    method : str = Field(description="method")
    key_finding: str = Field(description="Key finding")

parser1 = PydanticOutputParser(pydantic_object=Review)
prompt1 = PromptTemplate(
template="""
Extract the Research question,method,and key finding from a papaer abstract as structured data
Review:
{review}
{format_instructions}
""",
    input_variables=["review"],
    partial_variables={
        "format_instructions": parser1.get_format_instructions()
    }
)


review = input("Enter customer review: ")
p1 = prompt1.invoke({
    "review": review
})
response1 = model1.invoke(p1)
data = parser1.invoke(response1)
print("\nStep 1:")
print(data)

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are a support ticket writer."),
    ("human", """
      Create a short support ticket using only this information:

Complaint: {complaint}
Product: {product}
Sentiment: {sentiment}
""")
])
p2 = prompt2.invoke({
    "complaint": data.complaint,
    "product": data.product_or_feature,
    "sentiment": data.sentiment
})

response2 = model2.invoke(p2)
parser2 = StrOutputParser()
ticket = parser2.invoke(response2)
print(ticket)