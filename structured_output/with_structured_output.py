from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import Literal, Optional
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
    task="text-generation",
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

class Review(BaseModel):
    keythemes: list[str] = Field(description="Write down all the key themes discussed in the review in a list.")
    summary: str = Field(description="A brief summary of review.")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Return sentiment of the review either positive, neutral or negative")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list.")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list.")
    name: Optional[str] = Field(default=None, description="Write down name of the reviewer.")

parser = PydanticOutputParser(pydantic_object=Review)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Extract a structured review from the user text. {format_instructions}"),
        ("user", "{review_text}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

structured_output_chain = prompt | model | parser

response = structured_output_chain.invoke(
    {"review_text": """
        🧑 Customer Name: Rahul Mehta
        ⭐ Rating: 4.5 / 5
        🛍️ Product: Wireless Bluetooth Headphones
        📝 Customer Review

        I have been using these wireless Bluetooth headphones for almost 3 weeks now, and overall, I am very satisfied with the purchase.

        The sound quality is excellent, especially the bass, which feels deep and punchy without overpowering the vocals. I mostly use them for music, movies, and occasional office calls, and they perform really well in all scenarios.

        One thing that really impressed me is the battery life. On a single full charge, the headphones easily last more than 20 hours, which is perfect for long workdays and travel. I only need to charge them once every few days.

        In terms of comfort, the ear cushions are soft and breathable. I wear them for long hours while working, and they don’t cause any ear pain or discomfort. The build quality also feels premium and sturdy for the price.

        However, the product is not perfect. The microphone quality is just average, so during calls, my voice sometimes sounds slightly muffled to the other person. Also, while the Bluetooth connection is mostly stable, it occasionally disconnects when switching between multiple devices.

        Overall, considering the price, sound quality, comfort, and battery performance, I feel this is a great value-for-money product. I would definitely recommend it to anyone looking for good wireless headphones for daily use, especially for music and entertainment.
     
        ✅ Pros

        Excellent sound quality with strong bass

        Long battery life (20+ hours)

        Very comfortable for long usage

        Premium build quality

        Good value for money

        ❌ Cons

        Microphone quality could be better

        Occasional Bluetooth connection issues

        Not ideal for professional calling
    """}
)

print(response)
