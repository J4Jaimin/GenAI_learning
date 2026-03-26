from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="/home/jaimin.rana@agshealth.com/Documents/Projects/Generative AI/",
    glob="**/*.pdf",
    show_progress=True,
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)