from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
    task="text-generation",
    temperature=1.5
)

chat = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant that provides concise answers."),
    HumanMessage(content="What is the capital of France?")
]

response = chat.invoke(messages)
messages.append(AIMessage(content=response.content)
                )
print(response.content)

print("Chat History:")
print(messages)