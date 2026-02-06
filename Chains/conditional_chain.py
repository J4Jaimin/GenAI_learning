from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
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


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback")

pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

prompt_sentiment = PromptTemplate(
    template = "Classify the sentiment of the following feedbback into positive or negative: \n{feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
)

prompt_positive = PromptTemplate(
    template = "Please provide appropriate response to the following positive feedback: \n{feedback}",
    input_variables=["feedback"]
)

prompt_negative = PromptTemplate(
    template = "Please provide appropriate response to the following negative feedback: \n{feedback}",
    input_variables=["feedback"]
)

classifier_chain = prompt_sentiment | model | pydantic_parser

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt_positive | model | parser),
    (lambda x: x.sentiment == "negative", prompt_negative | model | parser),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = classifier_chain | branch_chain

response = chain.invoke({"feedback": "The product quality is outstanding and I am very satisfied with my purchase!"})

print(response)

chain.get_graph().print_ascii()