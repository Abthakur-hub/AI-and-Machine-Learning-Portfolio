import os
import shutil


from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
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



def process_document(file_path):


    # Remove old database

    if os.path.exists(CHROMA_DB_DIR):

        shutil.rmtree(
            CHROMA_DB_DIR
        )


    os.makedirs(
        CHROMA_DB_DIR,
        exist_ok=True
    )



    # Load file

    if file_path.endswith(".pdf"):

        loader = PyPDFLoader(
            file_path
        )


    elif file_path.endswith(".docx"):

        loader = Docx2txtLoader(
            file_path
        )


    elif file_path.endswith(".txt"):

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )


    else:

        raise ValueError(
            "Unsupported file"
        )



    documents = loader.load()



    # Split

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200

    )


    chunks = splitter.split_documents(
        documents
    )



    chunks = [

        chunk

        for chunk in chunks

        if chunk.page_content.strip()

    ]



    if len(chunks)==0:

        raise Exception(
            "No text extracted"
        )



    # Embeddings

    embeddings = GoogleGenerativeAIEmbeddings(

        model=EMBEDDING_MODEL

    )



    # Create new database

    Chroma.from_documents(

        documents=chunks,

        embedding=embeddings,

        persist_directory=CHROMA_DB_DIR

    )


    return len(chunks)