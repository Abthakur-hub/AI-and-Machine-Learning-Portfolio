import os

from dotenv import load_dotenv


from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)


from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings
)


from langchain_chroma import Chroma


from config import (
    EMBEDDING_MODEL,
    CHROMA_DB_DIR
)


load_dotenv()


DATA_PATH = "data"



def load_documents():

    documents = []


    for file in os.listdir(DATA_PATH):

        path = os.path.join(
            DATA_PATH,
            file
        )


        if file.endswith(".pdf"):

            loader = PyPDFLoader(
                path
            )


        elif file.endswith(".docx"):

            loader = Docx2txtLoader(
                path
            )


        elif file.endswith(".txt"):

            loader = TextLoader(
                path
            )


        else:

            continue


        docs = loader.load()

        documents.extend(
            docs
        )


    return documents





def split_documents(docs):


    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200

    )


    chunks = splitter.split_documents(
        docs
    )


    chunks = [

        c for c in chunks

        if c.page_content.strip()

    ]


    return chunks






def create_database(chunks):


    embeddings = GoogleGenerativeAIEmbeddings(

        model=EMBEDDING_MODEL

    )


    Chroma.from_documents(

        documents=chunks,

        embedding=embeddings,

        persist_directory=CHROMA_DB_DIR

    )





def main():


    print("\nLoading documents...")


    docs = load_documents()


    print(
        "Documents:",
        len(docs)
    )


    chunks = split_documents(
        docs
    )


    print(
        "Chunks:",
        len(chunks)
    )


    create_database(
        chunks
    )


    print(
        "\nChroma database created successfully"
    )




if __name__ == "__main__":

    main()