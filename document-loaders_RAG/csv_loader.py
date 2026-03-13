from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("/home/jaimin.rana@agshealth.com/Documents/mask_summary.csv")

docs = loader.lazy_load()

for doc in docs:
    print(doc.page_content)