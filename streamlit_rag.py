import os
import shutil
import streamlit as st
import sqlite3
import json
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from langchain_ollama import OllamaLLM
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
#from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFLoader

DATABASE_PATH = "chat_history.db"

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chroma"
DATA_PATH = "data"

#RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#reranker = CrossEncoder(RERANKER_MODEL)

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_name TEXT,
                    query TEXT,
                    response TEXT,
                    sources TEXT,
                    model TEXT
                )''')
    conn.commit()
    conn.close()

def save_chat_history():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_sessions")
    for session_name, history in st.session_state.chat_sessions.items():
        for entry in history:
            c.execute("INSERT INTO chat_sessions (session_name, query, response, sources, model) VALUES (?, ?, ?, ?, ?)",
                      (session_name, entry['query'], entry['response'], json.dumps(entry['sources']), entry['model']))
    conn.commit()
    conn.close()

def load_chat_history():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("SELECT session_name, query, response, sources, model FROM chat_sessions")
    rows = c.fetchall()
    chat_sessions = {}
    for row in rows:
        session_name, query, response, sources, model = row
        if session_name not in chat_sessions:
            chat_sessions[session_name] = []
        chat_sessions[session_name].append({
            "query": query,
            "response": response,
            "sources": json.loads(sources),
            "model": model
        })
    conn.close()
    return chat_sessions

def delete_session(session_name):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_sessions WHERE session_name = ?", (session_name,))
    conn.commit()
    conn.close()


def get_embedding_function(provider):
    if provider == "OpenAI":
        return OpenAIEmbeddings(model="text-embedding-ada-002", 
                              openai_api_key=OPENAI_API_KEY)
    elif provider == "Ollama":
        return OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")

def clear_database(provider="OpenAI"):
    try:
        collection_name = f"documents_{provider.lower()}"

        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function(provider),
            collection_name=collection_name
        )

        db.delete_collection()
        
        st.success(f"Successfully cleared {provider} collection!")
    except Exception as e:
        st.error(f"Error clearing {provider} collection: {str(e)}")

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def get_installed_ollama_models():
    try:
        import requests
        response = requests.get('http://ollama:11434/api/tags')
        if response.status_code == 200:
            models = response.json()
            return [model['name'] for model in models['models']]
        return []
    except:
        return []

def add_to_chroma(chunks: list[Document], provider="OpenAI"):
    try:
        if not chunks:
            st.warning("No documents to add to the database.")
            return
        
        embedding_function = get_embedding_function(provider)
        collection_name = f"documents_{provider.lower()}"  # Create separate collections
        
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embedding_function,
            collection_name=collection_name
        )

        chunks_with_ids = calculate_chunk_ids(chunks)
        chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
        
        db.add_documents(chunks_with_ids, ids=chunk_ids)
        
        # Verify documents were added
        collection_size = len(db.get()['ids'])
        st.info(f"Added {collection_size} documents to the {provider} collection.")
        
    except Exception as e:
        st.error(f"Error adding documents to database: {str(e)}")
        raise e

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

def populate_database(files, provider="OpenAI"):
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    documents = []

    for uploaded_file in files:
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    chunks = split_documents(documents)

    add_to_chroma(chunks, provider)
    
    st.success("Database populated successfully!")
    
def get_reranked_documents(query: str, provider="OpenAI"):
    initial_results = get_similar_documents(query, provider);
    
    return initial_results

    """
    query_doc_pairs = [(query, doc.page_content) for doc, _ in initial_results]
    scores = reranker.predict(query_doc_pairs)

    # Sort by the re-ranked scores (descending)
    reranked_results = sorted(zip([doc for doc, _ in initial_results], scores),
                              key=lambda x: x[1], reverse=True)
    
    return [(doc, score) for doc, score in reranked_results]
    """
    
def get_similar_documents(query: str, provider="OpenAI"):
    collection_name = f"documents_{provider.lower()}"
    
    db = Chroma(
         persist_directory=CHROMA_PATH,
         embedding_function=get_embedding_function(provider),
         collection_name=collection_name
    )
    
    return db.similarity_search_with_score(query, k=10)

def query_rag(query, provider="OpenAI", model="GPT-4o"):
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
        
    #results = db.similarity_search_with_score(query, k=5)
    #results = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    
    reranked_results = get_reranked_documents(query, provider)

    top_results = reranked_results[:5]

    prompt_template = ChatPromptTemplate.from_template(
        """
        Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Provide a right answer with a short explanation.
        """
    )

    context = "\n\n---\n\n".join([doc.page_content for doc, _ in top_results])
    prompt = prompt_template.format(context=context, question=query)

    client = None
    response = None

    if model == "OpenAI GPT-4o":
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content
    elif provider == "Ollama":
        client = OllamaLLM(model=model, base_url="http://ollama:11434")
        response = client.invoke(prompt)
    
    return response, [doc.metadata for doc, _ in top_results]

# Streamlit RAG
st.title("RAG-powered Document Assistant")

if "chat_sessions" not in st.session_state:
    init_db()
    st.session_state.chat_sessions = load_chat_history()
    st.session_state.current_session = None

# Sidebar for Session Management
def manage_sessions():
    st.header("Manage Chat Sessions")
    
    new_session_name = st.text_input("New session name")
    if st.button("Create session"):
        if new_session_name and new_session_name not in st.session_state.chat_sessions:
            st.session_state.chat_sessions[new_session_name] = []
            st.session_state.current_session = new_session_name
            st.success(f"Session '{new_session_name}' created and activated!")
        elif new_session_name in st.session_state.chat_sessions:
            st.warning("Session already exists!")
        else:
            st.warning("Please enter a valid session name.")

    if st.session_state.chat_sessions:
        session_selector = st.selectbox("Select session", st.session_state.chat_sessions.keys())
        if st.button("Switch to selected session"):
            st.session_state.current_session = session_selector
            st.success(f"Switched to session '{session_selector}'!")

    if st.session_state.chat_sessions:
        session_to_delete = st.selectbox("Delete session", st.session_state.chat_sessions.keys())
        
        if st.button("Delete selected session"):
            delete_session(session_to_delete)
            
            del st.session_state.chat_sessions[session_to_delete]
            
            if st.session_state.current_session == session_to_delete:
                st.session_state.current_session = None
                
            st.success(f"Session '{session_to_delete}' deleted!")
            st.rerun()

    if st.session_state.current_session:
        st.write(f"### Active Session: {st.session_state.current_session}")
    else:
        st.write("No session currently active.")

    if st.session_state.current_session:
        save_chat_history()

def upload_files():
    st.header("Upload Documents")

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    provider = st.selectbox("Provider", ["OpenAI", "Ollama"]) 
    
    if st.button("Reset Database"):
        clear_database(provider)

    if st.button("Populate Database") and uploaded_files:
        populate_database(uploaded_files, provider)

def query_database():
    st.header("Query")

    if not st.session_state.current_session:
        st.warning("No active session. Please create or select a session in 'Manage Sessions'.")
    else:
        session_name = st.session_state.current_session

        query_text = st.text_input("Enter your query:")
        provider = st.selectbox("Provider", ["OpenAI", "Ollama"])

        if provider == "OpenAI":
            model = st.selectbox("Model", ["OpenAI GPT-4o"])
        elif provider == "Ollama":
            installed_models = get_installed_ollama_models()
            if not installed_models:
                st.error("No Ollama models found. Please ensure Ollama is running and has models installed.")
                return
            model = st.selectbox("Model", installed_models)

        if st.button("Submit Query") and query_text:
            with st.spinner("Retrieving information..."):
                response, sources = query_rag(query_text, provider, model)
            modelInfo = f"Provider: {provider}, Model: {model}"
            
            st.session_state.chat_sessions[session_name].append({
                "query": query_text,
                "response": response,
                "sources": sources,
                "model": modelInfo
            })
            
            save_chat_history()

        st.subheader(f"Chat History for session: {session_name}")
        chat_history = st.session_state.chat_sessions[session_name]
        if chat_history:
            for idx, entry in enumerate(chat_history):
                st.write(f"**Q{idx+1}:** {entry['query']}")
                st.write(f"**A{idx+1} ({entry['model']}):** {entry['response']}")
                
                # Use expander to show/hide sources
                with st.expander(f"Sources #{idx+1}"):
                    for source in entry['sources']:
                        st.write(source)
                    
                st.markdown("---")
        else:
            st.write("No chat history for this session.")

pg = st.navigation([st.Page(manage_sessions, title="Manage sessions"), st.Page(upload_files, title="Upload files"), st.Page(query_database, title="Query")])
pg.run()