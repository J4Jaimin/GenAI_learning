from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

chat_mistral = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=1.2,
)

model = ChatHuggingFace(llm=chat_mistral)

parser = StrOutputParser()

pass_through = RunnablePassthrough()

# print(pass_through.invoke({"input": "This is a test input that should be passed through without any changes."}))

prompt1 = PromptTemplate(
    template = "Write a joke on this topic with some emojis: {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Write a explanation of joke: {joke}",
    input_variables=["joke"]
) 

joke_gen_chain = RunnableSequence(prompt1, model, parser)

joke = joke_gen_chain.invoke({"topic": "Adults working in an office"})

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({"topic": "Adults working in an office"})

print("Joke: ", result['joke'])
print("\n\n----------------------------------------------------\n\n")
print("Explanation: ", result['explanation'])