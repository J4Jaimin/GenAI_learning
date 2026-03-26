from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/home/jaimin.rana@agshealth.com/Documents/Projects/Generative AI/tech_current_affairs_2026.pdf")

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)