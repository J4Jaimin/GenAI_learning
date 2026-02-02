from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv

# import google.generativeai as genai
# import os

load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# models = genai.list_models()

# for model in models:
#     print(model.name, model.supported_generation_methods)


model = ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0.5)

result = model.invoke("What is the capital of France ?")

print(result.content)