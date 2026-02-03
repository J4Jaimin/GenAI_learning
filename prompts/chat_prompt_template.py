from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ("system", "you are a helpful {domain} expert."),
    ("human", "explain the following concept in a {style} manner: {concept}")
])

prompt = chat_template.invoke({
    "domain": "physics",
    "style": "concise",
    "concept": "quantum entanglement"
})

print(prompt)