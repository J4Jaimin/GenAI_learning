# from langchain_huggingface import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# text = "Gandhinagar is the capital of gujarat."

# vector = embeddings.embed_query(text)

# print(str(vector))

from sentence_transformers import SentenceTransformer

sentences = ["Gandhinagar is the capital of gujarat.", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

vector = model.encode(sentences)

print(str(vector))