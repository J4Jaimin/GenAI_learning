from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

chat_mistral = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=1.2,
)

model = ChatHuggingFace(llm=chat_mistral)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Write a tweet on this topic with some emojis: {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Write a short LinkedIn post on this topic with some emojis: {topic}",
    input_variables=["topic"]
)   

parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model, parser),
    "linkedin_post": RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({"topic": "Langchain Runnables"})

print(result['tweet'])
print("\n\n----------------------------------------------------\n\n")
print(result['linkedin_post'])