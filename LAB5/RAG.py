from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


model  = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature=0
)

prompt = PromptTemplate(
    template = 'Write a Summary on the following {poem} ',
    input_variables=['poem']
)

parser = StrOutputParser()
0

pdf_loader = PyPDFLoader('Esports.pdf')

docs_1 = pdf_loader.load()

chain  = prompt | model | parser
responce = chain.invoke({'poem' : docs_1[0].page_content})
print(responce)
