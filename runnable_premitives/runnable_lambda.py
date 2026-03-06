from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

def word_counter(text):
    return len(text.split())

# count_words_runnable = RunnableLambda(word_counter)

# result = count_words_runnable.invoke("Hello, how are you doing today?")

# print(result)

chat_mistral = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=1.2,
)

model = ChatHuggingFace(llm=chat_mistral)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Write a joke on this topic with some emojis: {topic}",
    input_variables=["topic"]
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_counter)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({"topic": "Adults working in an office"})

print(result["joke"])
print(result["word_count"])
