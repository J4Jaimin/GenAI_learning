from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# LLM
chat_mistral = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
    task="text-generation",
    temperature=0.3,
)

chat_meta = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.3
)

model1 = ChatHuggingFace(llm=chat_mistral)
model2 = ChatHuggingFace(llm=chat_meta)

parser = StrOutputParser()

prompt_notes = PromptTemplate(
    template = "Please generate around 5 lines summary on this topic: {topic}",
    input_variables=["topic"]
)

prompt_quiz = PromptTemplate(
    template = "Based on the following topic generate 3 questions with answers for reading:\n{topic}",
    input_variables=["topic"]
)

prompt_merge = PromptTemplate(
    template = "Combine the following notes and quiz into a single study guide:\nNotes:\n{notes}\nQuiz:\n{quiz}",
    input_variables=["notes", "quiz"]
)

chain_parallel = RunnableParallel(
    notes = prompt_notes | model1 | parser,
    quiz = prompt_quiz | model2 | parser
)

chain_merge = prompt_merge | model2 | parser

final_chain = chain_parallel | chain_merge

response = final_chain.invoke({"topic": "The impact of climate change on global ecosystems"})

print(response)

final_chain.get_graph().print_ascii()

