from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

cricketers = [
    "Virat Kohli is one of the finest batsmen of modern cricket, known for his aggressive attitude and consistency. He has led the Indian team in all formats and achieved many batting records. His fitness culture has inspired many young cricketers.",

    "Sachin Tendulkar is famously called the God of Cricket due to his legendary career. He is the highest run-scorer in international cricket and the first player to score 100 international centuries. Sachin remains a role model for millions of cricket lovers.",

    "MS Dhoni is known for his calm nature and brilliant leadership skills. Under his captaincy, India won all major ICC trophies. He is also regarded as one of the best finishers and wicketkeepers in cricket history.",

    "Rohit Sharma is admired for his elegant batting style and big-hitting ability. He holds the record for the highest individual score in ODI cricket. Rohit has also proven himself as a successful captain at both international and IPL levels.",

    "Jasprit Bumrah is India’s leading fast bowler with a unique bowling action. He is especially effective in death overs with his accurate yorkers. Bumrah plays a key role in India’s success across all formats."
]

query = "Who is the best scorer in cricket history?"

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

cricketers_embeddings = model.encode(cricketers)
query_embedding = model.encode(query)

best_match_idx, best_match_score = np.argmax(cosine_similarity([query_embedding], cricketers_embeddings)), np.max(cosine_similarity([query_embedding], cricketers_embeddings))

print(query)
print("Best matching cricketer description:", cricketers[best_match_idx])
print("Best match score:", best_match_score)