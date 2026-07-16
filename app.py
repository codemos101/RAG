import os
import warnings

import streamlit as st

warnings.filterwarnings(
    "ignore",
    message=r".*langchain-community.*",
    category=DeprecationWarning,
)

# Prefer the newer loader package when available; otherwise fall back to the old one.
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:  # pragma: no cover - fallback for older environments
    from pypdf import PdfReader as PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


def list_available_models():
    try:
        import google.generativeai as genai

        names = []
        for model in genai.list_models():
            if "generatecontent" in getattr(model, "supported_generation_methods", []):
                names.append(model.name.replace("models/", ""))
        return names
    except Exception:
        return []


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-flash-latest"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3


def build_vector_store(pdf_path, embedding_model=EMBEDDING_MODEL):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db


def answer_question(vector_db, llm, question):
    relevant_chunks = vector_db.similarity_search(question, k=TOP_K)
    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)

    prompt = f"""
    You are a helpful assistant. Answer the question based on the context below.
    If the question cannot be answered from the context, say "I don't know".
    Context: {context}
    Question: {question}
    Answer:
    """
    response = llm.invoke(prompt)
    answer_text = getattr(response, "content", str(response))

    if isinstance(answer_text, list):
        answer_text = "\n".join(
            block.get("text", "") for block in answer_text if isinstance(block, dict)
        )

    return answer_text, relevant_chunks


def main():
    st.set_page_config(page_title="PDF Q&A (RAG)", page_icon=":books:")
    st.title("Chat with your PDF :books:")
    st.caption(
        "A simple RAG app - Load -> chunk -> embed -> store -> retrieve -> augment -> answer"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Setup")
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Get your API key from https://developers.google.com/generative-ai",
        )
        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.success("Chat history cleared.")

        available = list_available_models()
        if available:
            if LLM_MODEL in available:
                default_index = available.index(LLM_MODEL)
            else:
                flash = [m for m in available if "flash" in m.lower()]
                default_index = available.index(flash[0]) if flash else 0
            chosen_model = st.selectbox("Gemini model", available, index=default_index)
        else:
            chosen_model = LLM_MODEL
            st.caption(f"Couldn't list available models automatically; using {LLM_MODEL}")

    if not api_key:
        st.info("Please set your Google API key in the sidebar to begin.")
        return

    os.environ["GOOGLE_API_KEY"] = api_key

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is None:
        st.info("👆 Upload a PDF file to get started.")
        return

    pdf_path = f"temp_{uploaded_file.name}"
    with open(pdf_path, "wb") as file_handle:
        file_handle.write(uploaded_file.getbuffer())

    with st.spinner("Reading and indexing your PDF... (This runs once)"):
        vector_db = build_vector_store(pdf_path)
    st.success("✅ Ready")

    llm = ChatGoogleGenerativeAI(model=chosen_model, temperature=0)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"🔍 See the {len(msg['sources'])} chunks used"):
                    for i, chunk in enumerate(msg["sources"], start=1):
                        page = chunk.metadata.get("page", "?")
                        st.markdown(f"Chunk {i} (page {page}):")
                        st.write(chunk.page_content)
                        st.divider()

    question = st.chat_input("Ask a question from your PDF")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF and thinking..."):
                answer, sources = answer_question(vector_db, llm, question)
                st.write(answer)
            with st.expander(f"See the {len(sources)} chunks used"):
                for i, chunk in enumerate(sources, start=1):
                    page = chunk.metadata.get("page", "?")
                    st.markdown(f"Chunk {i} (page {page}):")
                    st.write(chunk.page_content)
                    st.divider()

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
