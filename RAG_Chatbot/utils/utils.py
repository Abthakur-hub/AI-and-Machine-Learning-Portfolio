import os
import time
import logging


from colorama import (
    Fore,
    Style
)


from config import LOG_FILE



# -----------------------------
# Logging Setup
# -----------------------------


def setup_logger():

    os.makedirs(
        "logs",
        exist_ok=True
    )


    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format=
        "%(asctime)s | %(message)s"
    )



# -----------------------------
# Terminal Banner
# -----------------------------


def banner():

    print(
        Fore.CYAN
        +
        "="*60
    )

    print(
        "              RAG CHATBOT"
    )

    print(
        "="*60
        +
        Style.RESET_ALL
    )



# -----------------------------
# Timer
# -----------------------------


def start_timer():

    return time.time()



def end_timer(start):

    return round(
        time.time()-start,
        2
    )



# -----------------------------
# Document Formatter
# -----------------------------


def format_docs(docs):

    result=[]


    for i,doc in enumerate(
        docs,
        start=1
    ):

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "-"
        )


        result.append(
f"""
Document {i}

Source : {source}

Page : {page}


{doc.page_content}

"""
        )


    return "\n".join(result)



# -----------------------------
# Source Display
# -----------------------------


def show_sources(docs):

    print(
        Fore.YELLOW
        +
        "\nSources"
        +
        Style.RESET_ALL
    )


    for doc in docs:

        print(
            "-"
            *
            40
        )

        print(
            doc.metadata
        )