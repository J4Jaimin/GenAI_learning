from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from dotenv import load_dotenv

    load_dotenv()

    # LLM
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
        task="text-generation",
        temperature=0.3
    )

    model = ChatHuggingFace(llm=llm)

    parser = StrOutputParser()

    prompt = PromptTemplate(
        template = "Give me five interesting facts about {topic}",
        input_variables=["topic"]
    )

    chain = prompt | model | parser

    response = chain.invoke({"topic": "Football"})

    print(response)

    chain.get_graph().print_ascii()
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(chunks[1])
