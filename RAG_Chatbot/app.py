from dotenv import load_dotenv

from langchain_chroma import Chroma


from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings
)


from langchain_groq import (
    ChatGroq
)


from langchain_core.output_parsers import (
    StrOutputParser
)


from langchain_core.runnables import (
    RunnablePassthrough
)


from config import *

from rag_prompt import prompt



load_dotenv()



# Embedding

embeddings = GoogleGenerativeAIEmbeddings(

    model=EMBEDDING_MODEL

)



# Database

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





def format_docs(docs):

    return "\n\n".join(

        doc.page_content

        for doc in docs

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





def main():


    print(
        """
===============================
       RAG CHATBOT
===============================
Type exit to quit
"""
    )


    while True:


        question = input(
            "\nYou: "
        )


        if question.lower()=="exit":

            break



        answer = chain.invoke(
            question
        )


        print(
            "\nBot:",
            answer
        )





if __name__=="__main__":

    main()