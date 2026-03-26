from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# LLM
chat_mistral = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
    task="text-generation",
    temperature=0.3,
)

model = ChatHuggingFace(llm=chat_mistral)

prompt = PromptTemplate(
    template = "Summarize the following document in a concise manner:\n{document}", 
    input_variables=["document"]
)

parser = StrOutputParser()

loader = TextLoader("/home/jaimin.rana@agshealth.com/Documents/Projects/Generative AI/current_affairs.txt", encoding="utf8")

docs = loader.load()

print(docs[0].page_content)

chain = prompt | model | parser

response = chain.invoke({"document": docs[0].page_content})

print(response)
