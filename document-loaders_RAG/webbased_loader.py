from langchain_community.document_loaders import WebBaseLoader
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

parser = StrOutputParser()

prompt = PromptTemplate(
    template = "Please answer the following question:\n{question} based on the following document:\n{document}",
    input_variables=["question", "document"]
)

loader = WebBaseLoader("https://www.amazon.in/Apple-MacBook-13-inch-10-core-Unified/dp/B0DZDDV7GC/ref=sr_1_1?adgrpid=160979644943&dib=eyJ2IjoiMSJ9.uAnD2_MaxCWehr2EK-w2CIHbmuuiKMwcnm0E3pAE0twsnbXJHDM0oHNMNonL6BxLzVDf8maSlG4NdoRFHeuSc3SUGwlFafaYqzrBrEE_t7E3lLSheg3sl35jAk_H37y8SHU_t6hY5d_pYJYkEk6qP2bp8HxySYxN6MLp9vMiACx7RgzRR19qDNPGwVQXdMh--5V8o4m-bV0nx4cSwAq4BviYvxH7mO1yNHrZXXCu4Zc.vProA_WwvRDnOry71PrNjQsGAwWodnoEEkAUkXtzT1g&dib_tag=se&gad_source=1&hvadid=699292596438&hvdev=c&hvlocphy=9061768&hvnetw=g&hvqmt=e&hvrand=122338563643026087&hvtargid=kwd-2179240178396&hydadcr=24573_2265459&keywords=m4%2Bmacbook&mcid=f0c9dcf321c33e58a05cbba75cfd0251&qid=1773402049&sr=8-1&th=1")

docs = loader.load()

# print(docs)

chain = prompt | model | parser

response = chain.invoke({"question": "What are the advantages of the product?", "document": docs[0].page_content})

print(response)