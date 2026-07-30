from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_template(

"""
You are an AI assistant.

Answer the question using ONLY the context.

If the answer is not present in context,
say:
"I don't know based on the provided documents."


Context:
{context}


Question:
{question}


Answer:

"""

)