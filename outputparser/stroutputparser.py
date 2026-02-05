from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
    task="text-generation",
    temperature=0.3
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a report on the following topic: {topic}",  
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text: \n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

response = chain.invoke({"topic": "Black holes"})

print(response)

# chain = template1 | model | template2 | model

# response = chain.invoke({"topic": "Artificial Intelligence in Healthcare"})

# print(result2.content)
    