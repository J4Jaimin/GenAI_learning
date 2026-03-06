from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

# LLM
chat_mistral = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=1.2,
)

model = ChatHuggingFace(llm=chat_mistral)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Write a joke on this topic: {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Write a explanation of this joke: {response}",
    input_variables=["response"]
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({"topic": "AI"}))