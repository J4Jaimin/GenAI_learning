from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template=(
        "Write a {length} explanation of the paper '{paper}' "
        "in a {style} style with the use of code examples where applicable."
    ),
    input_variables=["paper", "style", "length"],
    validate_template=True
)

template.save("template.json")