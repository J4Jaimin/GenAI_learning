from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
    task="text-generation",
    temperature=0.3
)

model = ChatHuggingFace(llm=llm)

# JSON Schema
review_schema = {
    "title": "ReviewAnalysis",
    "type": "object",
    "properties": {
        "keythemes": {
            "type": "array",
            "items": {"type": "string"}
        },
        "summary": {
            "type": "string"
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
        },
        "pros": {
            "type": "array",
            "items": {"type": "string"},
            "nullable": True
        },
        "cons": {
            "type": "array",
            "items": {"type": "string"},
            "nullable": True
        },
        "name": {
            "type": "string",
            "nullable": True
        }
    },
    "required": ["keythemes", "summary", "sentiment"],
    "additionalProperties": False
}

# Parser
parser = JsonOutputParser(json_schema=review_schema)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract a structured review from the user text.\n"
            "Return ONLY valid JSON matching this schema:\n{format_instructions}"
        ),
        ("user", "{review_text}")
    ]
).partial(format_instructions=parser.get_format_instructions())

# Chain
chain = prompt | model | parser

# Invoke
response = chain.invoke(
    {
        "review_text": """
            Customer Name: Rahul Mehta

            I have been using these wireless Bluetooth headphones for almost 3 weeks now.
            The sound quality is excellent with deep bass.
            Battery lasts more than 20 hours and is very comfortable.
            Mic quality is average and bluetooth disconnects sometimes.

            Pros:
            - Excellent sound quality
            - Long battery life
            - Comfortable fit

            Cons:
            - Average microphone
            - Occasional connectivity issues
        """
    }
)

print(response)
