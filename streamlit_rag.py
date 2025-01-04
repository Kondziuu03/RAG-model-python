import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def get_embedding_function():
    return OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    st.success("Database cleared successfully!")
        
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_function
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")
        
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def populate_database(files):
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
    
    documents = []

    for uploaded_file in files:
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    chunks = split_documents(documents)
    add_to_chroma(chunks)
    
    st.success("Database populated successfully!")

def query_rag(query):
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
        
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    
    results = db.similarity_search_with_score(query, k=5)
    
    results = sorted(results, key=lambda x: x[1])
    
    top_docs = [doc.page_content for doc, _ in results]

    prompt_template = ChatPromptTemplate.from_template(
        """
        Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Provide a right answer with a short explanation.
        """
    )

    context = "\n\n---\n\n".join(top_docs)
    prompt = prompt_template.format(context=context, question=query)

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content, [doc.metadata for doc, _ in results]

# Streamlit App
st.title("RAG-powered Document Assistant")

# Sidebar Navigation
st.sidebar.title("Navigation")
options = ["Upload & Populate Database", "Query Database"]
choice = st.sidebar.radio("Go to", options)

if choice == "Upload & Populate Database":
    st.header("Upload Documents")

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Reset Database"):
        clear_database()

    if st.button("Populate Database") and uploaded_files:
        with st.spinner("Processing files and populating the database..."):
            populate_database(uploaded_files)

elif choice == "Query Database":
    st.header("Query Interface")

    query_text = st.text_input("Enter your query:")
    if st.button("Submit Query") and query_text:
        with st.spinner("Retrieving information..."):
            response, sources = query_rag(query_text)
        
        st.subheader("Response")
        st.write(response)

        st.subheader("Sources")
        for source in sources:
            st.write(source)
