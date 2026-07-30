import os
import time
import streamlit as st

from rag_engine import load_rag_chain

from document_processor import process_document



# =====================================
# Page Configuration
# =====================================

st.set_page_config(

    page_title="DocuMind AI",

    page_icon="🤖",

    layout="wide"

)



# =====================================
# Custom CSS
# =====================================

st.markdown(
"""
<style>

.main-title {

    font-size:45px;
    font-weight:800;
    text-align:center;

}


.sub-title {

    text-align:center;
    color:#94A3B8;
    font-size:18px;

}


</style>

""",

unsafe_allow_html=True
)



# =====================================
# Header
# =====================================


st.markdown(

"""
<div class="main-title">
🤖 DocuMind AI
</div>

<div class="sub-title">
Chat with your documents using RAG + Gemini + Llama 3.3
</div>

<br>
""",

unsafe_allow_html=True

)



# =====================================
# Initialize Chat
# =====================================

if "messages" not in st.session_state:

    st.session_state.messages = []



if "rag_chain" not in st.session_state:

    st.session_state.rag_chain = None



# =====================================
# Sidebar
# =====================================

with st.sidebar:


    st.title("⚙ Control Panel")



    st.subheader(
        "📄 Upload Document"
    )


    uploaded_file = st.file_uploader(

        "Upload PDF / DOCX / TXT",

        type=[
            "pdf",
            "docx",
            "txt"
        ]

    )



    if uploaded_file:


        if st.button(
            "🚀 Process Document"
        ):


            os.makedirs(

                "data",

                exist_ok=True

            )


            file_path = os.path.join(

                "data",

                uploaded_file.name

            )



            with open(

                file_path,

                "wb"

            ) as f:


                f.write(

                    uploaded_file.getbuffer()

                )



            with st.spinner(

                "Creating knowledge base..."

            ):


                chunks = process_document(
                    file_path
                )


                # remove old cached chain

                st.session_state.rag_chain = None


                # reload fresh database

                st.session_state.rag_chain = load_rag_chain()


            st.success(

                f"✅ Document processed\n\nChunks created: {chunks}"

            )



    st.divider()



    st.subheader(
        "🧠 Model Information"
    )


    st.info(

"""
LLM:
Groq Llama 3.3


Embedding:
Google Gemini


Vector DB:
ChromaDB


Architecture:
RAG Pipeline
"""

    )



    if st.button(
        "🗑 Clear Chat"
    ):

        st.session_state.messages = []

        st.rerun()



# =====================================
# Load RAG Chain
# =====================================


if "rag_chain" not in st.session_state:

    st.session_state.rag_chain = None



# =====================================
# Display Previous Chat
# =====================================


for message in st.session_state.messages:


    with st.chat_message(

        message["role"]

    ):

        st.write(

            message["content"]

        )



# =====================================
# Chat Input
# =====================================


question = st.chat_input(

    "Ask something about your document..."

)



if question:


    st.session_state.messages.append(

        {

            "role":"user",

            "content":question

        }

    )



    with st.chat_message(

        "user"

    ):

        st.write(

            question

        )



    if st.session_state.rag_chain:


        with st.chat_message(

            "assistant"

        ):


            with st.spinner(

                "Searching documents..."

            ):


                start = time.time()



                response = (
                    st.session_state
                    .rag_chain
                    .invoke(question)
                )



                end = time.time()



                st.write(

                    response

                )


                st.caption(

                    f"Response time: {round(end-start,2)} seconds"

                )



        st.session_state.messages.append(

            {

                "role":"assistant",

                "content":response

            }

        )


    else:


        st.error(

            "Please upload and process a document first."

        )