from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
    Artificial Intelligence (AI) is transforming industries across the globe. From healthcare to finance, AI systems are being used to automate tasks, analyze large volumes of data, and make intelligent decisions.
    In healthcare, AI-powered tools assist doctors in diagnosing diseases, predicting patient outcomes, and recommending treatments. For example, machine learning models can analyze medical images such as X-rays and MRIs with high accuracy.
    However, implementing AI comes with challenges. Data privacy, ethical concerns, and bias in algorithms are major issues that organizations must address. Poorly trained models can lead to incorrect predictions, which can have serious consequences.
    Another important aspect is scalability. Systems need to handle increasing loads efficiently without compromising performance. Cloud computing platforms like AWS, Azure, and Google Cloud are often used to scale AI applications.
    Text splitting plays a crucial role in Retrieval-Augmented Generation (RAG). When documents are too large, they must be broken into smaller chunks so that embedding models can process them effectively.
    Character-based text splitting is one of the simplest methods. It divides text into chunks based on a fixed number of characters. For example, a chunk size of 200 characters with an overlap of 50 characters ensures context is preserved across chunks.
    Despite its simplicity, character splitting may break sentences awkwardly. Therefore, it is often combined with more advanced techniques like recursive or semantic splitting.
    Understanding these techniques is essential for building efficient and accurate RAG pipelines.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(len(chunks))