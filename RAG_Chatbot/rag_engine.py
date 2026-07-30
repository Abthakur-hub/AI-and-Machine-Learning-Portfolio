from langchain_chroma import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_groq import ChatGroq

from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

from config import (
    EMBEDDING_MODEL,
    CHROMA_DB_DIR,
    LLM_MODEL,
    GROQ_API_KEY,
    TOP_K,
    TEMPERATURE
)

from rag_prompt import prompt



def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )



def load_rag_chain():


    embeddings = GoogleGenerativeAIEmbeddings(

        model=EMBEDDING_MODEL

    )


    db = Chroma(

        persist_directory=CHROMA_DB_DIR,

        embedding_function=embeddings

    )


    retriever = db.as_retriever(

        search_kwargs={
            "k": TOP_K
        }

    )


    llm = ChatGroq(

        model=LLM_MODEL,

        temperature=TEMPERATURE,

        groq_api_key=GROQ_API_KEY

    )


    chain = (

        {
            "context":
            retriever | format_docs,


            "question":
            RunnablePassthrough()

        }

        | prompt

        | llm

        | StrOutputParser()

    )


    return chain