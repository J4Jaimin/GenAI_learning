from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="/home/jaimin.rana@agshealth.com/Documents/Projects/Generative AI/",
    glob="**/*.pdf",
    show_progress=True,
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(len(docs))
print(docs[2].page_content)
print(docs[2].metadata)

