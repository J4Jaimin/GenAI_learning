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

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Please generate a detail report on this topic: {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Summarize the following report in a concise manner:\n{report}",
    input_variables=["report"]
)

chain = prompt1 | model | parser | prompt2 | model | parser

response = chain.invoke({"topic": "Bad sleep habits and their effects on health"})

print(response)

chain.get_graph().print_ascii()